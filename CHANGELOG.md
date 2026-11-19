# Changelog

## Unreleased

### New
- **No Hugging Face account needed.** The SAM-Audio models download from AudioGhost's own copies
  ([audioghostai](https://huggingface.co/audioghostai)), so there is no access request, no waiting for Meta's
  approval and no token. Meta's SAM License, which comes with the weights, is shown once and accepted in the app
  (`--accept-license` in the CLI).
- **Smaller downloads, same results.**
  - Small is 1.4 GB, Base 2.7 GB and Large 6.3 GB, against 5.1, 7.7 and 14.9 GB for Meta's checkpoints.
  - The transformer is stored in fp16, with no vision encoder and the audio codec shared by all sizes, as safetensors.
  - Every parameter loads bit-identically to Meta's checkpoints. bf16 and fp32 runs use an fp32 copy.
  - `model_hub/` builds and publishes the files.
- Meta's checkpoints already in the Hugging Face cache are used as they are. Downloaded models load offline, with no
  token. `AUDIOGHOST_MODEL_DIR` points to a local folder of model files for machines without internet.
- **Desktop app for Windows** (`desktop/`, Tauri): a one-click installer that sets up its own Python, PyTorch, FFmpeg
  and SAM-Audio, with no conda, Redis or git, and runs the same UI. See `desktop/README.md`.
- The backend can run in **desktop mode** (`AUDIOGHOST_INPROCESS=1`), with no Redis or Celery worker, and serve the
  UI itself (`AUDIOGHOST_STATIC_DIR`).

### Changed
- Meta's `-tv` precision checkpoints are the only models that still need a Hugging Face account.
- A token saved with Hugging Face's CLI (`hf auth login`) is used when none is saved in the app.

## v1.2.0 - 2026-11

A new workspace: several sounds per run, each on its own track, a mixer, and exports of exactly what you hear.

### New
- **Sessions**: an upload and everything separated from it live in a session that survives page reloads and restarts (the URL carries `?session=`). Recent sessions are listed on the start screen for 24 hours.
- **Several sounds per run**, one track each, processed one after another; finished tracks are playable while the next one runs. Cancel a running track, or remove queued and finished ones.
- **Everything else, made by the model**: each new sound is removed by SAM-Audio from what the earlier ones left (a chain), and each track also keeps the model's own "without this sound". SAM-Audio regenerates what it separates, so its output does not line up with the original sample for sample: on real recordings, subtracting the tracks from the original removed only 0-4 dB of a sound, while the chain removed 5-10 dB and still sounds like the recording.
- **The mix plays model output only**, picked by what you changed: everything at 0 dB plays the original; one track changed plays that track's "without" plus the track at your level; more changes play Everything else plus the tracks. Exports follow the same rule. Removing a finished track adds its sound back to Everything else.
- **Mixer**: volume, mute and solo per track, *hold to hear the original*; every source streams through a WebAudio gain and stays in sync with the original (and the video).
- **Exports**: what you hear as one file, single tracks as a ZIP, or the video with the mix; 24-bit WAV, FLAC or MP3; names like `song [no vocals].wav` or `song [everything else].wav`.
- **Temporal Lock, both ways**: drag on the timeline to mark where a sound is audible or where it is not, per sound (SAM-Audio's positive and negative anchors).
- **Start screen** with upload progress, recent sessions and a setup card (GPU, Hugging Face access, downloaded models); system status moved into a chip in the top bar.
- **Sessions API** (`/api/sessions/...`); the v1.1 endpoints keep working.
- Sounds separated in mono from a stereo file keep their place in the stereo image (each track's left/right
  balance is measured against the original).
- Tracks that were running when the worker stopped (window closed, crash, out of memory) are marked
  "Interrupted" when it starts again, with *Try again*, instead of spinning forever.

### Changed
- The interface was rebuilt around one workspace (multi-track timeline, separate panel, mix bar) with a single design system for dark and light themes; CSS modules replace ~250 inline style blocks, and one track lane replaces the two separate mixers.
- The worker's heartbeat now carries the GPU, its memory and the loaded model.
- Stereo separation is on by default for stereo files (sessions API and UI); a mono pass would collapse everything
  else to mono.
- The header shows the original AudioGhost logo and wordmark again.
- WaveSurfer and Tailwind were removed from the frontend (no longer used); Next's dev badge is hidden.

### Fixed
- WAV exports of a mix were written as 32-bit float instead of 24-bit.
- Old sessions are now cleaned up every 10 minutes, not only when the API starts.
- The worker's heartbeat kept going during a first model download (no false "worker not running").
- A new "everything else" revision could play silently until the mix changed.
- A sound typed but not added was lost when Separate failed.

## v1.1.1 - 2026-11

### Fixed
- The "separation worker is not running yet" warning showed up although the
  worker was fine: right after start-up (the worker is busy importing its
  libraries for ~20 s) and right after pressing Extract (a `--pool=solo`
  worker can't answer pings while it runs a job), and it then stayed on screen.
  The worker now sends a heartbeat from a background thread, and the page
  re-checks every few seconds while something looks wrong and treats a job
  that makes progress as proof that everything is up.

## v1.1.0 - 2026-11

A performance, correctness and install-reliability release. Same SAM-Audio
weights, new inference engine (`backend/workers/sam_lite.py`).

### Faster
- **4-6x faster end to end**: a 4:26 song takes 17 s / 20 s / 30 s with
  Small / Base / Large on an RTX 4090 (v1.0: 95 s / 104 s / 120 s).
- SAM-Audio runs its audio codec with cuDNN switched off, which pushes every
  convolution and the codec's LSTM onto slow fallback kernels (the LSTM ran
  as ~15,000 tiny per-timestep launches per chunk). AudioGhost now runs the
  identical computation with cuDNN (outputs match to ~50 dB SNR) and falls
  back to the original path if cuDNN rejects a shape: decode is ~4x faster
  and uses about half the memory.
- Chunks are batched through the diffusion transformer (auto-sized per model
  and free VRAM, halves automatically on out-of-memory).
- Speed / Quality presets: Fast (8 midpoint steps), Balanced (16, Meta's
  default), High (32).
- Background warm-up of SAM-Audio's heavy imports when the worker starts;
  optional `AUDIOGHOST_PRELOAD_MODEL`.

### Leaner
- New lite loader builds the model **without** the vision encoder, ImageBind,
  CLAP, SAM-Audio-Judge and the span predictor. v1.0 downloaded and built
  them (about 25 GB of extra downloads on first run: ImageBind 4.8 GB, CLAP
  1.9 GB, SAM-Audio-Judge 12.3 GB, PE-A-Frame 5.8 GB) only to delete them again.
- Random weight init is skipped (weights come from the checkpoint), the
  checkpoint is memory-mapped and copied straight into the target dtype:
  model load takes a few seconds once the weights are cached.
- Codec encode is done clip by clip, so peak VRAM no longer grows with the
  batch size.

### More accurate
- Mixed precision rebuilt: transformer in fp16 (bf16/fp32 selectable),
  while T5, the codec, RoPE tables and the ODE state stay in fp32. Against a
  strict fp32 reference (same input, same noise) it reaches 44-57 dB SNR on
  three test clips, vs 25-32 dB for v1.0's default path.
- Fixed: `torch.cuda.amp.autocast()` silently ran fp16 math in both modes, so
  "High Quality (float32)" was not actually fp32.
- Long files: equal-length chunks with 2 s overlap, raised-cosine crossfades
  and one global noise sequence - no seams, nothing dropped.

### Fixed
- **Temporal Lock did nothing**: the selected region was sent to the worker
  but never passed to the model.
- The last chunk was dropped when shorter than 1 s (outputs shorter than the
  input; merged videos lost their ending).
- New installs failed with `TypeError: BaseModel._from_pretrained() missing 2
  required keyword-only arguments: 'proxies' and 'resume_download'` (#9):
  `transformers<5` / `huggingface_hub<1.0` are pinned and the new loader no
  longer goes through that code path.
- Cancel now stops running jobs (the solo-pool worker ignored
  `revoke(terminate=True)`); queued jobs show how many jobs are ahead.
- Downloads are named after the file and prompt (`song [vocals].wav`,
  `song [no vocals].wav`) instead of task UUIDs (#6).
- WebM videos can be re-muxed (Opus audio instead of invalid AAC-in-WebM).
- `check_model_downloaded()` looked in the wrong folder.
- `stop.bat` killed every `node.exe` on the machine and left Redis running.
- Uploads are streamed to disk instead of read into memory; inputs are
  validated; old uploads/results are cleaned up after 24 h.
- The audio stem mixer synced to a track that didn't exist.

### New
- Stereo output (each channel separated with shared noise).
- Meta's `-tv` "precision" checkpoints (better target accuracy).
- WAV (24-bit) / FLAC / MP3 downloads and a one-click ZIP of all stems.
- First-run model download progress in the UI; backend / Redis / worker
  status banner.
- Command-line tool: `python sam_audio_lite.py song.mp3 "vocals"`.
- `scripts/benchmark.py` and unit tests (`backend/tests`).
- Accessibility (#6): labelled controls, switches and sliders, keyboard
  upload, live progress announcements.

### macOS & Linux
- `install.sh` / `start.sh` / `stop.sh` for Linux (x86_64, NVIDIA or CPU) and
  macOS (Apple Silicon), based on community PR #8 by @ShmuelOps: Celery runs
  with `--pool=solo`, every service gets its own process group and
  `stop.sh` only stops what `start.sh` started.
- Apple Silicon: SAM-Audio is installed without decord/xformers (no wheels,
  unused for text prompts; `backend/requirements-lean.txt`) and AudioGhost
  provides a stand-in for xformers' import. Runs on the CPU by default; the
  Apple GPU can be tried with `AUDIOGHOST_DEVICE=mps` (experimental, fp32).

### Install
- `backend/constraints.txt` pins torch 2.9.0 / torchcodec 0.9.1 /
  xformers 0.0.33.post1 for every `pip install`: an unpinned xformers (now
  0.0.34+, built for torch 2.10) made fresh installs replace the CUDA build of
  PyTorch with a CPU build, or fail outright.
- `install.bat` pins SAM-Audio (`bb4c699`), transformers and huggingface_hub,
  picks CUDA 12.8 wheels for RTX 50-series GPUs and verifies the environment
  at the end.
- Windows: "entry point not found ... libintl_bind_textdomain_codeset ...
  gdk_pixbuf-2.0-0.dll" during install (#10) and "Could not load
  libtorchcodec" afterwards. conda-forge's Windows FFmpeg pulls in GTK
  packages, and `libglib` and `libintl` both install an `intl-8.dll`; the one
  that wins lacks the `libintl_*` exports. `install.bat` now downloads a
  self-contained FFmpeg 8.1 shared build (BtbN) into `ffmpeg\`, and
  `start.bat` puts it on PATH.
- Frontend: Next.js 16.4.6 / React 19.2.9 (fixes a critical Next.js
  advisory), unused axios removed.
