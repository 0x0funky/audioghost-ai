"""
Celery tasks - audio separation worker.

The heavy lifting lives in ``workers/sam_lite.py``; this module handles the job
lifecycle: input decoding, model caching, progress, cancellation and outputs.
Heavy imports (torch, sam_audio) happen inside the task so the API process,
which imports this module to enqueue jobs, stays light.
"""
from __future__ import annotations

import gc
import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

from celery.exceptions import Ignore
from celery.signals import worker_init, worker_ready, worker_shutdown

from settings import OUTPUT_DIR, read_hf_token
from workers import queue_state
from workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# One model in memory at a time: (cache key, LiteModel)
_MODEL_CACHE: dict = {}
_MODEL_LOCK = threading.Lock()
#: The loaded model's repo id, for the heartbeat thread: it must never wait for
#: _MODEL_LOCK, which a model download holds for minutes.
_LOADED_REPO = {"value": None}

QUALITY_PRESETS = ("fast", "balanced", "high")


def _precision(use_float32: bool, device) -> str:
    """
    fp32 on request and on CPU. CUDA: AUDIOGHOST_PRECISION (fp16 default, or
    bf16). Apple MPS (experimental): fp32 unless AUDIOGHOST_PRECISION says otherwise.
    """
    if use_float32 or device.type == "cpu":
        return "fp32"
    preferred = os.environ.get("AUDIOGHOST_PRECISION", "").strip().lower()
    if device.type == "mps":
        return preferred if preferred in ("fp16", "bf16", "fp32") else "fp32"
    return preferred if preferred in ("fp16", "bf16") else "fp16"


def _check_model_requirement(model_size: str, variant: str, use_float32: bool, token) -> None:
    """Fail early when the model must be downloaded but can't be yet."""
    import model_license
    from workers import model_files

    precision = _precision(use_float32, sam_lite.resolve_device())
    need = model_files.missing_requirement(model_size, variant, precision, token=token,
                                           license_accepted=model_license.accepted())
    if need == "token":
        raise RuntimeError("Meta's precision (-tv) checkpoints need a Hugging Face account with access "
                           "to them. Connect it in the app first.")
    if need == "license":
        raise RuntimeError("Accept the SAM-Audio license in the app first; it is shown before the first "
                           "model download.")


def _get_model(model_size: str, variant: str, use_float32: bool, token, progress):
    """Return the cached lite model, evicting a different one first."""
    from workers import sam_lite

    repo_id = sam_lite.model_repo_id(model_size, variant)
    device = sam_lite.resolve_device()
    precision = _precision(use_float32, device)
    key = (repo_id, str(device), precision)
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        for old_key in list(_MODEL_CACHE):
            logger.info("Unloading %s", old_key[0])
            _LOADED_REPO["value"] = None
            sam_lite.unload(_MODEL_CACHE.pop(old_key))

        sam_lite.ensure_downloaded(
            repo_id, token, lambda frac, msg: progress(8 + int(frac * 17), msg), precision=precision
        )
        progress(26, f"Loading {repo_id.split('/')[-1]} ({precision})...")
        lite = sam_lite.load_lite_model(repo_id, device=device, precision=precision, token=token)
        _MODEL_CACHE[key] = lite
        _LOADED_REPO["value"] = repo_id
        return lite


_HEARTBEAT_INTERVAL_SECONDS = 15
_heartbeat_stop = threading.Event()
_worker_state = {"value": "starting"}


def _worker_info() -> dict:
    """Device, memory and loaded model for the UI (only once torch is imported)."""
    import sys

    torch = sys.modules.get("torch")
    sam_lite = sys.modules.get("workers.sam_lite")
    if _worker_state["value"] != "ready" or torch is None or sam_lite is None:
        return {}
    info = {}
    repo = _LOADED_REPO["value"]
    if repo:
        info["model"] = repo.split("/")[-1].replace("sam-audio-", "")
    try:
        device = sam_lite.resolve_device()
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            info.update(
                device="cuda",
                gpu=torch.cuda.get_device_name(device),
                vram_total_gb=round(total / 1024**3, 1),
                vram_used_gb=round((total - free) / 1024**3, 1),
            )
        else:
            info.update(device=device.type, gpu="Apple GPU" if device.type == "mps" else "CPU")
    except Exception as err:  # noqa: BLE001 - informational only
        logger.debug("device info unavailable: %s", err)
    return info


def _heartbeat_loop() -> None:
    """
    Tell the API this worker is alive. A ``--pool=solo`` worker can't answer
    Celery pings while a job runs, and the warm-up imports hold the GIL long
    enough to miss them too; this thread keeps a Redis key fresh instead.
    """
    while True:
        queue_state.worker_heartbeat(_worker_state["value"], _worker_info())
        if _heartbeat_stop.wait(_HEARTBEAT_INTERVAL_SECONDS):
            return


def _recover_interrupted_tracks() -> None:
    """
    Celery acknowledges a job when it starts, so a job that was running when
    the worker stopped (window closed, crash, out of memory) is gone. Fail its
    tracks instead of leaving them "running" forever. Runs before the worker
    takes its first job.
    """
    try:
        import sessions

        from settings import INPROCESS

        dead = {job for job in [queue_state.running_task()] if job}
        queue_state.clear_running()
        failed = sessions.recover_interrupted(dead, all_active=INPROCESS)
        if failed:
            logger.warning("Marked %d interrupted track(s) as failed", failed)
    except Exception as err:  # noqa: BLE001 - best effort
        logger.warning("Could not check for interrupted tracks: %s", err)


@worker_init.connect
def _start_heartbeat(**_kwargs):
    # As early as possible: the worker needs ~20 s more to connect and warm up,
    # and the UI should say "starting" rather than "not running" meanwhile.
    _recover_interrupted_tracks()
    _heartbeat_stop.clear()
    threading.Thread(target=_heartbeat_loop, name="audioghost-heartbeat", daemon=True).start()


@worker_shutdown.connect
def _stop_heartbeat(**_kwargs):
    _heartbeat_stop.set()
    queue_state.worker_gone()


@worker_ready.connect
def _warm_up(**_kwargs):
    """
    Importing SAM-Audio's dependencies takes ~20 s (timm, transformers, ...).
    Do it in the background as soon as the worker starts so the first job
    doesn't pay for it. Set AUDIOGHOST_PRELOAD_MODEL=small|base|large to also
    load a model into VRAM up front.
    """

    def run():
        try:
            import torch  # noqa: F401

            from workers import sam_lite  # installs platform shims first
            import sam_audio  # noqa: F401

            logger.info("Warm-up: SAM-Audio libraries imported")
            _worker_state["value"] = "ready"
            queue_state.worker_heartbeat("ready", _worker_info())
            preload = os.environ.get("AUDIOGHOST_PRELOAD_MODEL", "").strip().lower()
            token = read_hf_token()
            if preload in sam_lite.MODEL_SIZES and token:
                _get_model(preload, "default", False, token, lambda *_: None)
                logger.info("Warm-up: preloaded sam-audio-%s", preload)
        except Exception as err:  # noqa: BLE001 - warm-up is best effort
            logger.warning("Warm-up skipped: %s", err)
        finally:
            if _worker_state["value"] != "ready":
                _worker_state["value"] = "ready"
                queue_state.worker_heartbeat("ready")

    threading.Thread(target=run, name="audioghost-warmup", daemon=True).start()


@celery_app.task(bind=True)
def separate_audio_task(
    self,
    audio_path: str,
    description: str,
    mode: str = "extract",
    anchors: Optional[List] = None,
    model_size: str = "base",
    chunk_duration: float = 25.0,
    use_float32: bool = False,
    is_video: bool = False,
    quality: str = "balanced",
    stereo: bool = False,
    model_variant: str = "default",
    original_filename: Optional[str] = None,
    seed: Optional[int] = None,
    batch_size: int = 0,
):
    """
    Separate ``description`` from an audio or video file.

    Outputs (32-bit float WAV, model sample rate):
      ``{task_id}.original.wav`` - the decoded input
      ``{task_id}.ghost.wav``    - the described sound ("isolated")
      ``{task_id}.clean.wav``    - everything else ("without")
    """
    from workers import audio_io, sam_lite

    task_id = self.request.id
    started = time.perf_counter()
    last_reported = {"progress": -1, "message": None, "time": 0.0}

    def progress(value: int, message: str):
        now = time.monotonic()
        # Throttle Redis writes, but always send new messages and big jumps.
        if (
            message == last_reported["message"]
            and value - last_reported["progress"] < 2
            and now - last_reported["time"] < 1.0
        ):
            return
        last_reported.update(progress=value, message=message, time=now)
        self.update_state(state="PROGRESS", meta={"progress": int(value), "message": message})
        queue_state.touch_running(task_id)

    def should_cancel() -> bool:
        return queue_state.is_cancelled(task_id)

    queue_state.mark_started(task_id)
    try:
        if should_cancel():
            raise sam_lite.SeparationCancelled()
        description = (description or "").strip()
        if not description:
            raise ValueError("Please describe the sound to separate.")
        if quality not in QUALITY_PRESETS:
            quality = "balanced"
        steps = sam_lite.QUALITY_STEPS[quality]
        chunk_duration = max(5.0, min(60.0, float(chunk_duration)))
        anchor_list = sam_lite.normalize_anchors(anchors)

        token = read_hf_token()
        _check_model_requirement(model_size, model_variant, use_float32, token)

        progress(5, "Preparing...")
        lite = _get_model(model_size, model_variant, use_float32, token, progress)
        if should_cancel():
            raise sam_lite.SeparationCancelled()

        progress(28, "Reading audio..." if not is_video else "Reading the video's audio track...")
        audio = audio_io.load_audio(audio_path, lite.sample_rate, stereo=stereo)
        num_channels, num_samples = audio.shape
        audio_duration = num_samples / lite.sample_rate
        logger.info(
            "Task %s: %.1fs, %d ch, %s, quality=%s, anchors=%s",
            task_id, audio_duration, num_channels, lite.repo_id, quality, anchor_list,
        )

        def separation_progress(fraction: float, message: str):
            progress(30 + int(fraction * 62), message)

        result = sam_lite.separate(
            lite,
            audio,
            description,
            chunk_seconds=chunk_duration,
            steps=steps,
            anchors=anchor_list,
            batch_size=batch_size,
            seed=seed,
            progress=separation_progress,
            should_cancel=should_cancel,
        )

        progress(93, "Saving results...")
        base = OUTPUT_DIR / task_id
        original_path = base.with_suffix(".original.wav")
        ghost_path = base.with_suffix(".ghost.wav")
        clean_path = base.with_suffix(".clean.wav")
        audio_io.save_audio(original_path, audio, lite.sample_rate)
        audio_io.save_audio(ghost_path, result.target, lite.sample_rate)
        audio_io.save_audio(clean_path, result.residual, lite.sample_rate)

        processing_time = time.perf_counter() - started
        logger.info(
            "Task %s done in %.1fs (%.1fx realtime, %d chunks, batch %d)",
            task_id, processing_time, audio_duration / max(processing_time, 1e-6),
            result.chunks, result.batch_size,
        )
        output = {
            "original_path": str(original_path),
            "ghost_path": str(ghost_path),
            "clean_path": str(clean_path),
            "description": description,
            "mode": mode,
            "audio_duration": round(audio_duration, 2),
            "processing_time": round(processing_time, 2),
            "model_size": model_size,
            "model_variant": model_variant,
            "model": lite.repo_id,
            "precision": lite.precision,
            "quality": quality,
            "steps": steps,
            "stereo": num_channels > 1,
            "channels": num_channels,
            "sample_rate": lite.sample_rate,
            "chunks": result.chunks,
            "batch_size": result.batch_size,
            "seed": result.seed,
            "anchors": [list(a) for a in anchor_list],
            "original_filename": original_filename,
            "is_video": bool(is_video),
        }
        if is_video:
            output["video_path"] = str(audio_path)
        progress(100, "Complete!")
        return output

    except sam_lite.SeparationCancelled:
        # Custom state: Celery decodes REVOKED results as exceptions, a plain
        # meta dict there breaks AsyncResult. Ignore() keeps this state.
        self.update_state(state="CANCELLED", meta={"progress": 0, "message": "Cancelled"})
        raise Ignore()
    except Exception as err:
        logger.exception("Task %s failed", task_id)
        raise RuntimeError(f"Separation failed: {err}") from None
    finally:
        queue_state.mark_finished(task_id)
        gc.collect()
        sam_lite.empty_device_cache()


def _friendly_error(err: BaseException) -> str:
    text = str(err).strip() or type(err).__name__
    return text if len(text) <= 300 else text[:297] + "..."


@celery_app.task(bind=True)
def separate_session_task(self, session_id: str, track_ids: List[str], options: dict):
    """
    Separate the queued prompts of a session, one after another. Each finished
    sound is saved right away (the UI can play it while the next one runs), with
    the model's own "without this sound" and a new "everything else": a chain, in
    which the model removes this sound from what was left after the earlier ones.
    Everything played back is model output; nothing is subtracted.
    """
    import sessions
    from workers import audio_io, sam_lite

    job_id = self.request.id
    queue_state.mark_started(job_id)
    quality = options.get("quality") if options.get("quality") in QUALITY_PRESETS else "balanced"
    steps = sam_lite.QUALITY_STEPS[quality]
    chunk = max(5.0, min(60.0, float(options.get("chunk_duration") or 25.0)))
    seed = options.get("seed")

    def job_cancelled() -> bool:
        return queue_state.is_cancelled(job_id)

    def safely(action, *args, **kwargs):
        try:
            return action(*args, **kwargs)
        except sessions.SessionNotFound:  # deleted while we were working
            return None

    def open_tracks() -> List[str]:
        data = safely(sessions.load, session_id)
        if data is None:
            return []
        return [t["id"] for t in data["tracks"]
                if t["id"] in track_ids and t["status"] in sessions.ACTIVE]

    try:
        token = read_hf_token()
        _check_model_requirement(options.get("model_size", "base"), options.get("variant", "default"),
                                 bool(options.get("use_float32")), token)

        last_note = {"time": 0.0}

        def note(_value, message):  # model download / loading, shown on the first track
            now = time.monotonic()
            if now - last_note["time"] < 0.5:
                return
            last_note["time"] = now
            pending = open_tracks()
            if pending:
                safely(sessions.set_track, session_id, pending[0], message=message)
            queue_state.touch_running(job_id)

        lite = _get_model(
            options.get("model_size", "base"), options.get("variant", "default"),
            bool(options.get("use_float32")), token, note,
        )
        data = sessions.load(session_id)
        source = sessions.session_dir(session_id) / data["source"]["file"]
        stereo = bool(options.get("stereo")) and data["source"]["channels"] > 1
        audio = audio_io.load_audio(source, lite.sample_rate, stereo=stereo)

        for tid in track_ids:
            if job_cancelled():
                break
            data = safely(sessions.load, session_id)
            track = sessions.find_track(data, tid) if data else None
            if track is None or track["status"] != "queued":
                continue
            safely(sessions.set_track, session_id, tid, status="running", progress=0.0, message="Starting")
            started = time.perf_counter()
            last = {"time": 0.0}
            chain_from = dict(data["rest"]) if data.get("rest") else None

            def progress(fraction, message, tid=tid, last=last):
                now = time.monotonic()
                if now - last["time"] < 0.5 and fraction < 1.0:
                    return
                last["time"] = now
                safely(sessions.set_track, session_id, tid, progress=round(fraction, 3), message=message)
                queue_state.touch_running(job_id)

            def should_cancel(tid=tid) -> bool:
                return job_cancelled() or queue_state.is_cancelled(f"{job_id}:{tid}")

            def stage(lo, hi, progress=progress):
                return lambda fraction, message: progress(lo + (hi - lo) * fraction, message)

            try:  # one bad prompt (or a failed save) shouldn't stop the rest
                common = dict(chunk_seconds=chunk, steps=steps, seed=seed, should_cancel=should_cancel,
                              anchors=sam_lite.normalize_anchors(track.get("anchors")))
                result = sam_lite.separate(lite, audio, track["prompt"],
                                           progress=stage(0.0, 0.5 if chain_from else 1.0), **common)
                rest = result.residual
                if chain_from:
                    # The chain: the model removes this sound from what the earlier ones left.
                    previous = audio_io.load_audio(sessions.session_dir(session_id) / chain_from["file"],
                                                   lite.sample_rate, stereo=stereo)
                    rest = sam_lite.separate(lite, previous, track["prompt"], progress=stage(0.5, 1.0),
                                             decode_target=False, **common).residual
                elapsed = time.perf_counter() - started
                safely(sessions.finish_track, session_id, tid, result.target, result.residual, rest,
                       lite.sample_rate, elapsed, chain_from)
            except sam_lite.SeparationCancelled:
                safely(sessions.drop_track, session_id, tid)
                continue
            except Exception as err:  # noqa: BLE001
                logger.exception("Session %s, track %s failed", session_id, tid)
                safely(sessions.set_track, session_id, tid, status="failed", message="",
                       error=_friendly_error(err), finished_at=time.time())
                continue
            logger.info("Session %s: %r done in %.1fs", session_id, track["prompt"], elapsed)

    except Exception as err:  # noqa: BLE001 - setup failed: report it on every open track
        logger.exception("Session job %s failed", job_id)
        message = _friendly_error(err)
        for tid in open_tracks():
            safely(sessions.set_track, session_id, tid, status="failed", message="",
                   error=message, finished_at=time.time())
    finally:
        for tid in open_tracks():  # cancelled before they started
            safely(sessions.drop_track, session_id, tid)
        safely(sessions.purge_deleted, session_id)
        queue_state.mark_finished(job_id)
        gc.collect()
        try:
            from workers import sam_lite

            sam_lite.empty_device_cache()
        except Exception:  # noqa: BLE001
            pass


@celery_app.task(bind=True)
def match_pattern_task(self, audio_path: str, sample_path: str, threshold: float = 0.85,
                       model_size: str = "base"):
    """Placeholder: find and remove sounds similar to a sample (planned)."""
    return {"status": "not_implemented", "message": "Pattern matching is on the roadmap."}


def cleanup_old_files(max_age_hours: float) -> int:
    """Delete uploads/outputs older than ``max_age_hours``. Returns files removed."""
    from settings import UPLOAD_DIR

    if max_age_hours <= 0:
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for directory in (UPLOAD_DIR, OUTPUT_DIR):
        for path in Path(directory).iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
    return removed
