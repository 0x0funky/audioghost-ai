"""
v1.4: takes of a sound, parts of a file, rebuilding "everything else", mixes that
change over time, kept sessions and batches. Synthetic audio and a stand-in for
the model; no GPU or Redis needed.

Run from the repo root:  python -m pytest backend/tests -q
"""
import io
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sessions  # noqa: E402
from test_sessions import SR, as_audio, client, finish, make_session, read, store, tone, upload  # noqa: E402,F401
from workers import queue_state  # noqa: E402


def add(sid, *prompts, job="job"):
    return sessions.add_tracks(sid, [p if isinstance(p, dict) else {"text": p} for p in prompts], {}, job)


def active_ids(sid):
    return [t["id"] for t in sessions.active_takes(sessions.load(sid))]


def full(sid, name):
    """A session's audio file as the mixes and the browser get it: at full length, mono."""
    return sessions.read_tensor(sessions.load(sid), name, stereo=False)[0].numpy()


# --------------------------------------------------------------------------- #
# Takes
# --------------------------------------------------------------------------- #


def test_single_sound_everything_else_is_its_own_without(store):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    folder = sessions.session_dir(sid)
    [first] = add(sid, "low")
    assert sessions.load(sid)["tracks"][0]["sound"] == first
    finish(session, first, a, b + c)
    rest = sessions.load(sid)["rest"]
    assert rest["file"] == f"{first}.without.wav" and rest["tracks"] == [first]
    assert not list(folder.glob("rest.*.wav"))  # no copy of the same audio
    assert (folder / f"{first}.without.peaks.json").exists()

    # Another take of the same sound: it plays as soon as it is finished, and "everything else" follows it.
    [second] = add(sid, {"text": "low", "sound": first})
    assert sessions.chain_need(sessions.load(sid), sessions.find_track(sessions.load(sid), second)) == "alias"
    finish(session, second, 0.5 * a, 0.5 * a + b + c)
    data = sessions.load(sid)
    assert active_ids(sid) == [second]
    assert [t["active"] for t in data["tracks"]] == [False, True]
    assert data["rest"]["file"] == f"{second}.without.wav" and data["rest"]["rev"] == 2
    assert data["rest"]["tracks"] == [first]  # the sound, not the take
    assert (folder / f"{first}.without.wav").exists()  # the first take is kept, to compare

    sessions.activate_take(sid, first)
    assert active_ids(sid) == [first]
    assert sessions.load(sid)["rest"]["file"] == f"{first}.without.wav"
    plan = sessions.mix_plan(sessions.load(sid), {first: 0.0, "rest": 1.0})
    assert [(p.name, g) for p, g in plan] == [(f"{first}.without.wav", 1.0)]

    sessions.remove_done_track(sid, first)  # the take that plays: the other one takes over
    data = sessions.load(sid)
    assert active_ids(sid) == [second] and data["rest"]["file"] == f"{second}.without.wav"
    assert not (folder / f"{first}.wav").exists() and not (folder / f"{first}.without.wav").exists()
    sessions.remove_done_track(sid, second)
    assert sessions.load(sid)["rest"] is None


def test_another_take_leaves_the_chain_alone(store):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    low, mid = add(sid, "low", "mid")
    finish(session, low, a, b + c)
    finish(session, mid, b, a + c, c)
    chain = sessions.load(sid)["rest"]
    assert chain["file"] == "rest.2.wav" and chain["tracks"] == [low, mid]

    [again] = add(sid, {"text": "low", "sound": low})
    data = sessions.load(sid)
    assert sessions.chain_need(data, sessions.find_track(data, again)) == "none"
    outcome = sessions.finish_track(sid, again, as_audio(0.9 * a), as_audio(0.1 * a + b + c), SR, 1.0)
    assert outcome == "done"
    data = sessions.load(sid)
    assert data["rest"] == chain and not sessions.rest_is_stale(data)
    assert set(active_ids(sid)) == {again, mid}
    # Mixes and exports use the take that plays.
    plan = sessions.mix_plan(data, {again: 0.0, mid: 1.0, "rest": 1.0})
    assert [(p.name, g) for p, g in plan] == [(f"{again}.without.wav", 1.0)]
    plan = sessions.mix_plan(data, {again: 0.0, mid: 0.0, "rest": 1.0})
    assert [(p.name, g) for p, g in plan] == [("rest.2.wav", 1.0)]

    # Removing one take of a sound that has another keeps the sound out of "everything else".
    sessions.remove_done_track(sid, again)
    data = sessions.load(sid)
    assert data["rest"]["file"] == "rest.2.wav" and set(active_ids(sid)) == {low, mid}


def test_a_take_over_another_range_asks_for_a_rebuild(store):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    folder = sessions.session_dir(sid)
    low, mid = add(sid, {"text": "low", "range": [0.5, 1.5]}, "mid")
    assert sessions.load(sid)["tracks"][0]["range"] == [0.5, 1.5]
    finish(session, low, a, b + c)
    finish(session, mid, b, a + c, c)
    assert sessions.load(sid)["rest"]["ranges"] == {low: [0.5, 1.5], mid: None}

    [whole] = add(sid, {"text": "low", "sound": low})  # the same sound, now over the whole file
    data = sessions.load(sid)
    assert sessions.chain_need(data, sessions.find_track(data, whole)) == "rebuild"
    assert sessions.finish_track(sid, whole, as_audio(a), as_audio(b + c), SR, 1.0) == "rebuild"
    assert sessions.rest_is_stale(sessions.load(sid))

    # The worker runs the chain again and publishes it; a stale version is not kept to compare with.
    assert sessions.publish_rest(sid, as_audio(0.5 * c), SR, [low, mid], seed=5)
    data = sessions.load(sid)
    assert data["rest"]["file"] == "rest.3.wav" and data["rest"]["seed"] == 5
    assert data["rest"]["ranges"] == {low: None, mid: None} and not sessions.rest_is_stale(data)
    assert data["rest_alts"] == [] and not (folder / "rest.2.wav").exists()


def test_rebuilt_everything_else_keeps_the_version_it_replaces(store):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    folder = sessions.session_dir(sid)
    low, mid = add(sid, "low", "mid")
    finish(session, low, a, b + c)
    finish(session, mid, b, a + c, c)
    assert sessions.publish_rest(sid, as_audio(0.5 * c), SR, [mid, low], seed=9)
    data = sessions.load(sid)
    assert data["rest"]["rev"] == 3 and data["rest"]["tracks"] == [mid, low]
    assert [alt["rev"] for alt in data["rest_alts"]] == [2] and (folder / "rest.2.wav").exists()

    sessions.activate_rest(sid, 2)  # the first version sounded better after all
    data = sessions.load(sid)
    assert data["rest"]["file"] == "rest.2.wav" and [alt["rev"] for alt in data["rest_alts"]] == [3]
    with pytest.raises(sessions.TrackNotFound):
        sessions.activate_rest(sid, 7)
    sessions.remove_rest_alt(sid, 3)
    assert sessions.load(sid)["rest_alts"] == [] and not (folder / "rest.3.wav").exists()

    # A new sound changes what "everything else" is: versions made before it can't be compared any more.
    sessions.publish_rest(sid, as_audio(0.4 * c), SR, [low, mid], seed=1)
    assert len(sessions.load(sid)["rest_alts"]) == 1
    [high] = add(sid, "high")
    finish(session, high, c, a + b, np.zeros_like(c))
    data = sessions.load(sid)
    assert data["rest_alts"] == [] and data["rest"]["tracks"] == [low, mid, high]
    assert sorted(p.name for p in folder.glob("rest.*.wav")) == [data["rest"]["file"]]

    # Sounds changed while a rebuild ran: nothing is published.
    assert not sessions.publish_rest(sid, as_audio(c), SR, [low, mid], seed=2)
    assert sessions.load(sid)["rest"]["tracks"] == [low, mid, high]
    assert not list(folder.glob("rest-build-*"))


def test_summary_lists_each_sound_once_with_its_size(store):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    [low] = add(sid, "low")
    finish(session, low, a, b + c)
    add(sid, {"text": "low", "sound": low}, "mid")
    [item] = sessions.list_sessions()
    assert item["tracks"] == [{"prompt": "low", "status": "queued"}, {"prompt": "mid", "status": "queued"}]
    assert item["size_bytes"] == sessions.folder_size(sid) > 3 * SR * 4
    assert item["expires_at"] is None  # kept until deleted


def test_sessions_are_kept_until_deleted(store):
    import settings

    assert settings.RETENTION_HOURS == 0
    session, _ = make_session(store)
    data = sessions.load(session["id"])
    data["updated_at"] -= 400 * 24 * 3600
    sessions._write(session["id"], data)
    assert sessions.cleanup(settings.RETENTION_HOURS) == 0
    assert [s["id"] for s in sessions.list_sessions()] == [session["id"]]


def test_interrupted_rebuild_is_marked_failed(store):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    sessions.set_rest_job(sid, {"job_id": "job-r", "status": "running", "progress": 0.3, "message": "", "error": None})
    sessions.set_rest_job(sid, progress=0.5)
    assert sessions.load(sid)["rest_job"]["progress"] == 0.5
    sessions.recover_interrupted()
    job = sessions.load(sid)["rest_job"]
    assert job["status"] == "failed" and job["error"] == sessions.INTERRUPTED_REBUILD
    sessions.set_rest_job(sid)
    assert "rest_job" not in sessions.load(sid)


# --------------------------------------------------------------------------- #
# Mixes that change over time ("only here")
# --------------------------------------------------------------------------- #


def test_mix_can_change_over_time(store):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    low, mid = add(sid, "low", "mid")
    finish(session, low, a, b + c)
    finish(session, mid, b, a + c, c)
    data = sessions.load(sid)
    everything = {low: 1.0, mid: 1.0, "rest": 1.0}
    segments = [{"start": 1.0, "end": 2.0, "gains": {low: 0.0, mid: 1.0, "rest": 1.0}}]  # no "low" from 1 s to 2 s
    pieces = sessions.mix_pieces(data, everything, segments)
    assert [(s, e, [p.name for p, _ in src]) for s, e, src in pieces] == [
        (0, SR, ["original.wav"]), (SR, 2 * SR, [f"{low}.without.wav"]), (2 * SR, 3 * SR, ["original.wav"])]

    path, name = sessions.export_mix(sid, everything, "wav", segments)
    out = read(path)[:, 0]
    assert name == "My Song [mix].wav" and len(out) == 3 * SR
    original = a + b + c
    fade = int(sessions.CROSSFADE_SECONDS * SR)
    np.testing.assert_allclose(out[: SR - fade], original[: SR - fade], atol=2e-6)
    np.testing.assert_allclose(out[SR + fade: 2 * SR - fade], (b + c)[SR + fade: 2 * SR - fade], atol=2e-6)
    np.testing.assert_allclose(out[2 * SR + fade:], original[2 * SR + fade:], atol=2e-6)
    # The crossfade adds up to 1: what both sides share (b + c) passes through untouched.
    middle = slice(SR - fade, SR + fade)
    assert np.abs(out[middle] - (b + c)[middle]).max() <= np.abs(a).max() + 1e-6
    np.testing.assert_allclose(out[SR], (0.5 * a + b + c)[SR], atol=2e-3)

    # A change that plays nothing anywhere is still refused; segments with nothing to say are the plain mix.
    with pytest.raises(ValueError):
        sessions.export_mix(sid, {low: 0.0, mid: 0.0, "rest": 0.0}, "wav",
                            [{"start": 0.0, "end": 3.0, "gains": {low: 0.0, mid: 0.0, "rest": 0.0}}])
    assert len(sessions.mix_pieces(data, everything, [{"start": 0.5, "end": 0.9, "gains": everything}])) == 1


# --------------------------------------------------------------------------- #
# The worker, with a stand-in for the model
# --------------------------------------------------------------------------- #

SHARE = {"low": 0.5, "mid": 0.25, "high": 0.2}


@pytest.fixture
def worker(store, monkeypatch):
    """Runs the real session tasks with a model that takes a fixed share of whatever it is given."""
    from workers import sam_lite, tasks

    calls = []

    def separate(lite, audio, description, *, anchors=None, seed=None, progress=None, decode_target=True, **_):
        calls.append({"prompt": description, "samples": audio.shape[-1], "seed": seed, "anchors": list(anchors or []),
                      "target": decode_target})
        share = SHARE[description]
        if progress:
            progress(1.0, "chunk 1/1")
        return sam_lite.SeparationOutput(target=audio * share if decode_target else None, residual=audio * (1 - share),
                                         sample_rate=SR, chunks=1, batch_size=1,
                                         seed=11 if seed is None else seed, steps=1)

    monkeypatch.setattr(sam_lite, "separate", separate)
    monkeypatch.setattr(tasks, "read_hf_token", lambda: None)
    monkeypatch.setattr(tasks, "_check_model_requirement", lambda *args: None)
    monkeypatch.setattr(tasks, "_get_model", lambda *args: SimpleNamespace(sample_rate=SR))

    def run(sid, ids, **options):
        result = tasks.separate_session_task.apply(args=[sid, ids, options], task_id=f"job-{len(calls)}")
        assert not result.failed(), result.result
        return sessions.load(sid)

    def rebuild(sid, **options):
        result = tasks.rebuild_rest_task.apply(args=[sid, options], task_id=f"rebuild-{len(calls)}")
        assert not result.failed(), result.result
        return sessions.load(sid)

    return SimpleNamespace(run=run, rebuild=rebuild, calls=calls)


def test_worker_runs_one_pass_for_a_take_and_two_for_a_new_sound(store, worker):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    folder = sessions.session_dir(sid)
    mix = a + b + c
    [low] = sessions.add_tracks(sid, [{"text": "low", "seed": 3}], {"seed": None}, "job-0")
    data = worker.run(sid, [low])
    assert [call["target"] for call in worker.calls] == [True]  # the only sound: no chain step
    assert data["tracks"][0]["status"] == "done" and data["tracks"][0]["settings"]["seed"] == 3
    np.testing.assert_allclose(read(folder / f"{low}.wav")[:, 0], 0.5 * mix, atol=1e-6)
    assert data["rest"]["file"] == f"{low}.without.wav"

    [mid] = add(sid, "mid")
    data = worker.run(sid, [mid])
    assert [call["target"] for call in worker.calls[1:]] == [True, False]  # the take, then the chain step
    assert worker.calls[1]["seed"] is None and worker.calls[2]["seed"] == 11  # the step repeats the take's seed
    np.testing.assert_allclose(read(folder / data["rest"]["file"])[:, 0], 0.5 * 0.75 * mix, atol=1e-6)

    [again] = add(sid, {"text": "low", "sound": low, "seed": 4})
    data = worker.run(sid, [again])
    assert len(worker.calls) == 4 and worker.calls[3]["seed"] == 4  # one pass; the chain is left alone
    assert data["rest"]["file"] == "rest.2.wav" and "rest_job" not in data
    assert set(active_ids(sid)) == {again, mid}


def test_worker_separates_only_a_part_of_the_file(store, worker):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    folder = sessions.session_dir(sid)
    mix = a + b + c
    [low] = add(sid, {"text": "low", "range": [1.0, 2.0], "anchors": [["+", 0.5, 1.25], ["-", 1.5, 2.5], ["+", 2.2, 2.6]]})
    data = worker.run(sid, [low])
    call = worker.calls[0]
    assert call["samples"] == SR  # only that second went through the model
    assert [(k, round(s, 3), round(e, 3)) for k, s, e in call["anchors"]] == [("+", 0.0, 0.25), ("-", 0.5, 1.0)]
    # Only the part is stored: a 20-second try on an hour-long file must not cost an hour of audio on disk.
    assert len(read(folder / f"{low}.wav")) == SR and len(read(folder / f"{low}.without.wav")) == SR
    assert data["tracks"][0]["part"] == [SR, 2 * SR]
    target, without = full(sid, f"{low}.wav"), full(sid, f"{low}.without.wav")
    assert len(target) == len(without) == 3 * SR  # read back at full length
    fade = int(0.03 * SR)
    assert not target[:SR].any() and not target[2 * SR:].any()  # silent outside its part
    np.testing.assert_allclose(target[SR + fade: 2 * SR - fade], 0.5 * mix[SR + fade: 2 * SR - fade], atol=1e-6)
    np.testing.assert_array_equal(without[:SR], mix[:SR].astype(np.float32))  # the original itself, untouched
    np.testing.assert_array_equal(without[2 * SR:], mix[2 * SR:].astype(np.float32))
    np.testing.assert_allclose(without[SR + fade: 2 * SR - fade], 0.5 * mix[SR + fade: 2 * SR - fade], atol=1e-6)
    np.testing.assert_allclose(target + without, mix, atol=1e-5)  # the crossfade loses nothing
    assert data["rest"]["ranges"] == {low: [1.0, 2.0]}
    # Its overviews cover the whole file, like every other track's.
    import json as _json

    overview = _json.loads((folder / f"{low}.peaks.json").read_text())
    assert abs(overview["duration"] - 3.0) < 0.01
    third = len(overview["peaks"]) // 3
    assert max(overview["peaks"][:third - 2]) == 0 and max(overview["peaks"][third + 2:2 * third - 2]) > 0.1
    # The mix reads it through its window: without "low" the file is the original outside the part.
    path, _name = sessions.export_mix(sid, {low: 0.0, "rest": 1.0}, "wav")
    mixed = read(path)[:, 0]
    np.testing.assert_allclose(mixed[:SR], mix[:SR], atol=2e-6)
    np.testing.assert_allclose(mixed[SR + fade: 2 * SR - fade], 0.5 * mix[SR + fade: 2 * SR - fade], atol=2e-6)
    # A single track is exported at full length too, so it lines up with the others.
    zip_path, _name = sessions.export_tracks(sid, [low, "rest"], "wav")
    with zipfile.ZipFile(zip_path) as zf:
        sizes = {name: sf.info(io.BytesIO(zf.read(name))).frames for name in zf.namelist()}
    assert sizes == {"My Song [low].wav": 3 * SR, "My Song [everything else].wav": 3 * SR}

    # A second sound over the same part: its chain step also touches that part only.
    [mid] = add(sid, {"text": "mid", "range": [1.0, 2.0]})
    data = worker.run(sid, [mid])
    rest = read(folder / data["rest"]["file"])[:, 0]
    np.testing.assert_array_equal(rest[:SR], mix[:SR].astype(np.float32))
    np.testing.assert_allclose(rest[SR + fade: 2 * SR - fade], 0.5 * 0.75 * mix[SR + fade: 2 * SR - fade], atol=1e-6)

    # The first sound again, over the whole file: the chain no longer fits and is run again, in order.
    del worker.calls[:]
    [whole] = add(sid, {"text": "low", "sound": low})
    data = worker.run(sid, [whole])
    assert [(call["prompt"], call["samples"], call["target"]) for call in worker.calls] == [
        ("low", 3 * SR, True), ("low", 3 * SR, False), ("mid", SR, False)]
    assert data["rest"]["ranges"] == {low: None, mid: [1.0, 2.0]} and not sessions.rest_is_stale(data)
    assert "rest_job" not in data and data["rest_alts"] == []
    rest = read(folder / data["rest"]["file"])[:, 0]
    np.testing.assert_allclose(rest[: SR - fade], 0.5 * mix[: SR - fade], atol=1e-6)
    np.testing.assert_allclose(rest[SR + fade: 2 * SR - fade], 0.5 * 0.75 * mix[SR + fade: 2 * SR - fade], atol=1e-6)


def test_part_only_files_are_served_at_full_length(client, worker):
    """The browser plays a part-only take like any other file: full length, and it can seek in it."""
    sid = upload(client, seconds=3.0).json()["id"]
    [tid] = sessions.add_tracks(sid, [{"text": "low", "range": [1.0, 2.0]}], {"stereo": True}, "job-1")
    worker.run(sid, [tid], stereo=True)
    folder = sessions.session_dir(sid)
    assert sf.info(str(folder / f"{tid}.wav")).frames == SR  # stored: one second
    for name, outside in ((f"{tid}.wav", "silence"), (f"{tid}.without.wav", "original")):
        res = client.get(f"/api/sessions/{sid}/files/{name}")
        assert res.status_code == 200 and res.headers["accept-ranges"] == "bytes", name
        assert int(res.headers["content-length"]) == len(res.content)
        audio, rate = sf.read(io.BytesIO(res.content), dtype="float32", always_2d=True)
        assert rate == SR and len(audio) == 3 * SR, name
        original = read(folder / "original.wav")
        if outside == "silence":
            assert not audio[:SR].any() and not audio[2 * SR:].any()
        else:
            assert audio.shape[1] == 2  # the original's own channels
            np.testing.assert_array_equal(audio[:SR], original[:SR])
            np.testing.assert_array_equal(audio[2 * SR:], original[2 * SR:])
        assert np.abs(audio[SR + 2000: 2 * SR - 2000]).max() > 0.01
        # Seeking: any byte range is exactly that slice of the whole response.
        for first, last in ((0, 99), (40, 60), (1000, 9001), (len(res.content) - 7, len(res.content) - 1)):
            piece = client.get(f"/api/sessions/{sid}/files/{name}", headers={"Range": f"bytes={first}-{last}"})
            assert piece.status_code == 206 and piece.content == res.content[first:last + 1], (name, first, last)
            assert piece.headers["content-range"] == f"bytes {first}-{last}/{len(res.content)}"
        tail = client.get(f"/api/sessions/{sid}/files/{name}", headers={"Range": "bytes=-64"})
        assert tail.status_code == 206 and tail.content == res.content[-64:]
        assert client.get(f"/api/sessions/{sid}/files/{name}", headers={"Range": f"bytes={len(res.content)}-"}).status_code == 416

    # Removing the part-only sound while another one is still running keeps it, at full length, for the chain.
    [other] = sessions.add_tracks(sid, [{"text": "mid"}], {"stereo": True}, "job-2")
    worker.run(sid, [other], stereo=True)
    [running] = sessions.add_tracks(sid, [{"text": "high"}], {"stereo": True}, "job-3")
    sessions.set_track(sid, running, status="running")
    sessions.remove_done_track(sid, tid)
    assert sf.info(str(folder / f"{tid}.deleted.wav")).frames == 3 * SR
    assert not (folder / f"{tid}.wav").exists() and not (folder / f"{tid}.without.wav").exists()


def test_worker_rebuilds_everything_else_in_another_order(store, worker):
    session, (a, b, c) = make_session(store, channels=1)
    sid = session["id"]
    low, mid = add(sid, "low", "mid")
    worker.run(sid, [low, mid])
    del worker.calls[:]
    sessions.set_rest_job(sid, {"job_id": "x", "status": "queued", "progress": 0.0, "message": "", "error": None})
    data = worker.rebuild(sid, order=[mid, low], seed=21)
    assert [(call["prompt"], call["seed"], call["target"]) for call in worker.calls] == [("mid", 21, False), ("low", 21, False)]
    assert data["rest"]["tracks"] == [mid, low] and data["rest"]["seed"] == 21
    assert [alt["rev"] for alt in data["rest_alts"]] == [2] and "rest_job" not in data


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #


def separate(client, sid, **body):
    body.setdefault("prompts", [{"text": "vocals"}])
    return client.post(f"/api/sessions/{sid}/separate", json=body)


def test_api_descriptions_must_be_english(client):
    sid = upload(client).json()["id"]
    for text in ("狗叫聲", "鼓 drums", "голос"):
        res = separate(client, sid, prompts=[{"text": text}])
        assert res.status_code == 400 and "English" in res.json()["detail"], text
    for text in ("a dog barking", "café noise", "drums (808), hi-hats & claps!"):
        assert separate(client, sid, prompts=[{"text": text}]).status_code == 202, text


def test_api_takes_parts_and_seeds(client):
    sid = upload(client, seconds=4.0).json()["id"]
    res = separate(client, sid, range=[1.0, 2.5])
    assert res.status_code == 202, res.text
    [first] = res.json()["session"]["tracks"]
    assert first["range"] == [1.0, 2.5] and first["sound"] == first["id"] and first["active"] is False
    assert isinstance(first["seed"], int)  # picked now, so the result can be repeated
    assert separate(client, sid, range=[1.0, 1.5]).status_code == 400  # shorter than a second
    whole = separate(client, sid, range=[0.0, 4.0], seed=77).json()["session"]["tracks"][-1]
    assert whole["range"] is None and whole["seed"] == 77

    again = separate(client, sid, prompts=[{"text": "vocals", "retake_of": first["id"]}])
    assert again.status_code == 202
    assert again.json()["session"]["tracks"][-1]["sound"] == first["id"]
    assert separate(client, sid, prompts=[{"text": "vocals", "retake_of": "t00000000"}]).status_code == 404
    queued = client.queued[-1][0]
    assert queued[0] == sid and queued[2]["cleanup"] is False


def test_api_choose_a_take_and_rebuild_everything_else(client):
    sid = upload(client, seconds=2.0).json()["id"]
    a, b = tone(440, 2.0), tone(660, 2.0)
    ids = [separate(client, sid, prompts=[{"text": text}]).json()["track_ids"][0] for text in ("vocals", "drums")]
    session = {"id": sid}
    assert client.post(f"/api/sessions/{sid}/tracks/{ids[0]}/activate").status_code == 409  # not finished yet
    finish(session, ids[0], a, b)
    # One sound: there is no chain to rebuild.
    assert client.post(f"/api/sessions/{sid}/rest/rebuild", json={}).status_code == 409
    finish(session, ids[1], b, a, np.zeros_like(a))
    retake = separate(client, sid, prompts=[{"text": "vocals", "retake_of": ids[0]}]).json()["track_ids"][0]
    finish(session, retake, 0.5 * a, 0.5 * a + b)

    body = client.get(f"/api/sessions/{sid}").json()
    assert {t["id"]: t["active"] for t in body["tracks"]} == {ids[0]: False, ids[1]: True, retake: True}
    assert body["rest"]["stale"] is False and body["rest"]["alts"] == [] and body["rest_job"] is None
    body = client.post(f"/api/sessions/{sid}/tracks/{ids[0]}/activate").json()
    assert {t["id"]: t["active"] for t in body["tracks"]} == {ids[0]: True, ids[1]: True, retake: False}
    assert client.post(f"/api/sessions/{sid}/tracks/t00000000/activate").status_code == 404
    # Sessions made before v1.4 have no overview of a take's "without": it is made when asked for.
    overview = sessions.session_dir(sid) / f"{ids[0]}.without.peaks.json"
    overview.unlink()
    assert client.get(f"/api/sessions/{sid}/files/{overview.name}").status_code == 200

    res = client.post(f"/api/sessions/{sid}/rest/rebuild", json={"order": [ids[1], ids[0]], "quality": "high"})
    assert res.status_code == 202, res.text
    assert res.json()["session"]["rest_job"]["status"] == "queued"
    args, job_id = client.queued[-1]
    assert args[0] == sid and args[1]["order"] == [ids[1], ids[0]] and isinstance(args[1]["seed"], int)
    assert client.post(f"/api/sessions/{sid}/rest/rebuild", json={}).status_code == 409  # one at a time
    assert client.delete(f"/api/sessions/{sid}/rest/job").json()["session"]["rest_job"] is None

    sessions.publish_rest(sid, as_audio(0.1 * a), SR, [ids[1], ids[0]], seed=3)
    body = client.get(f"/api/sessions/{sid}").json()
    [alt] = body["rest"]["alts"]
    assert alt["audio_url"].endswith(f"/files/rest.{alt['rev']}.wav")
    assert client.post(f"/api/sessions/{sid}/rest/{alt['rev']}/activate").json()["rest"]["rev"] == alt["rev"]
    body = client.delete(f"/api/sessions/{sid}/rest/{body['rest']['rev']}").json()
    assert body["rest"]["alts"] == []


def test_api_export_with_changes_in_one_part(client):
    sid = upload(client, seconds=2.0).json()["id"]
    a, b = tone(440, 2.0), tone(660, 2.0)
    tid = separate(client, sid).json()["track_ids"][0]
    finish({"id": sid}, tid, a, b)
    res = client.post(f"/api/sessions/{sid}/export", json={
        "kind": "mix", "format": "flac", "gains": {tid: 1.0, "rest": 1.0},
        "segments": [{"start": 0.5, "end": 1.5, "gains": {tid: 0.0, "rest": 1.0}}]})
    assert res.status_code == 200, res.text
    out, _ = sf.read(io.BytesIO(res.content), dtype="float32", always_2d=True)
    assert len(out) == 2 * SR
    assert abs(float(np.abs(out[SR, :]).max()) - float(np.abs(np.stack([b, b], axis=1)[SR]).max())) < 1e-3


def test_api_lists_sessions_with_sizes(client):
    sid = upload(client).json()["id"]
    body = client.get("/api/sessions").json()
    assert [item["id"] for item in body["sessions"]] == [sid]
    assert body["total_bytes"] == body["sessions"][0]["size_bytes"] > 0
    assert body["retention_hours"] == 0 and body["folder"]


# --------------------------------------------------------------------------- #
# Batches
# --------------------------------------------------------------------------- #


@pytest.fixture
def batch_client(client, tmp_path, monkeypatch):
    import batches

    monkeypatch.setattr(batches, "BATCH_DIR", tmp_path / "batches")
    return client


def clip(seconds=2.0, name="clip.wav"):
    buf = io.BytesIO()
    sf.write(buf, np.stack([tone(440, seconds), tone(660, seconds)], axis=1), SR, format="WAV")
    return {"file": (name, buf.getvalue(), "audio/wav")}


def test_batch_applies_one_recipe_to_many_files(batch_client):
    client = batch_client
    recipe = {"name": "interviews", "prompts": [{"text": "talking", "gain": 0.0}, {"text": "music", "gain": 0.5, "seed": 8}],
              "rest_gain": 1.0, "model_size": "small", "quality": "fast", "kind": "mix", "format": "flac"}
    assert client.post("/api/batches", json=dict(recipe, prompts=[{"text": "說話"}])).status_code == 400
    batch = client.post("/api/batches", json=recipe).json()
    bid = batch["id"]
    assert batch["files"] == 0 and batch["recipe"]["settings"]["model_size"] == "small"

    for name in ("one.wav", "two.wav"):
        res = client.post(f"/api/batches/{bid}/items", files=clip(name=name))
        assert res.status_code == 201, res.text
    body = client.get(f"/api/batches/{bid}").json()
    assert [(item["filename"], item["status"]) for item in body["items"]] == [("one.wav", "queued"), ("two.wav", "queued")]
    assert len(client.queued) == 2
    (sid, track_ids, options), _job = client.queued[0]
    assert options["quality"] == "fast" and len(track_ids) == 2
    tracks = sessions.load(sid)["tracks"]
    assert [t["prompt"] for t in tracks] == ["talking", "music"] and tracks[1]["settings"]["seed"] == 8
    assert client.get(f"/api/batches/{bid}/items/{sid}/export").status_code == 409  # not finished
    assert client.post(f"/api/batches/{bid}/export").status_code == 409

    a, b = tone(440, 2.0), tone(660, 2.0)
    finish({"id": sid}, track_ids[0], a, b)
    assert client.get(f"/api/batches/{bid}").json()["items"][0]["status"] == "running"  # one sound of two
    finish({"id": sid}, track_ids[1], 0.5 * b, a + 0.5 * b, 0.5 * b)
    body = client.get(f"/api/batches/{bid}").json()
    assert body["items"][0]["status"] == "done" and body["counts"] == {"done": 1, "queued": 1}
    import batches

    gains = batches.gains_for(batches.load(bid), sid)
    assert gains == {"rest": 1.0, track_ids[0]: 0.0, track_ids[1]: 0.5}

    res = client.get(f"/api/batches/{bid}/items/{sid}/export")
    assert res.status_code == 200 and "one%20%5Bmix%5D.flac" in res.headers["content-disposition"].replace(" ", "%20")
    res = client.post(f"/api/batches/{bid}/export")
    assert res.status_code == 200
    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        assert zf.namelist() == ["one [mix].flac"]

    sessions.set_track(client.queued[1][0][0], client.queued[1][0][1][0], status="failed", error="Out of memory")
    body = client.get(f"/api/batches/{bid}").json()
    assert body["items"][1]["status"] == "failed" and body["items"][1]["error"] == "Out of memory"
    assert [b["id"] for b in client.get("/api/batches").json()["batches"]] == [bid]

    assert client.delete(f"/api/batches/{bid}?sessions=true").status_code == 200
    assert client.get(f"/api/batches/{bid}").status_code == 404
    assert sessions.list_sessions() == []
