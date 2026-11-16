# AudioGhost AI 🎵👻

![AudioGhost Banner](banner.png)

**AI-Powered Object-Oriented Audio Separation**

Describe the sound you want to extract or remove using natural language. Powered by Meta's [SAM-Audio](https://github.com/facebookresearch/sam-audio) model, running locally on your own GPU.

![Version](https://img.shields.io/badge/version-1.1-green) ![Python](https://img.shields.io/badge/python-3.11+-blue) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

## 🆕 What's new in v1.1

| | |
|---|---|
| ⚡ **~4–6× faster** | A 4:26 song with the Small model: **95 s → 17 s** on an RTX 4090 ([benchmarks](#-performance)) |
| 🎯 **More precise** | On our test clips the new mixed precision stays within 44–57 dB SNR of a strict fp32 run (v1.0's default path: 25–32 dB) |
| 💾 **~25 GB less to download** | The first run no longer fetches (or builds) ImageBind, CLAP, SAM-Audio-Judge and PE-A-Frame, which lite mode threw away |
| 🔧 **Installs again** | Fixes the `_from_pretrained() missing ... 'proxies' and 'resume_download'` error ([#9](https://github.com/0x0funky/audioghost-ai/issues/9)); installer picks the CUDA 12.8 build for RTX 50-series |
| 🎚️ **Seamless long files** | Overlapping, crossfaded chunks - no seams, nothing dropped at the end |
| 🕒 **Temporal Lock works** | The selected region was ignored in v1.0 |
| ✨ **New options** | Speed/Quality presets · Stereo · Meta's `-tv` precision models · WAV/FLAC/MP3 + ZIP downloads with readable names · working Cancel · queue position · download progress |
| ♿ **Accessible** | Screen-reader labels, keyboard upload, live progress ([#6](https://github.com/0x0funky/audioghost-ai/issues/6)) |
| 🖥️ **CLI** | `python sam_audio_lite.py song.mp3 "vocals"` |

Full list in the [CHANGELOG](CHANGELOG.md).

## 🎬 Demo

### Audio Separation
https://github.com/user-attachments/assets/49248e25-0c56-46ab-a821-2de7f7016bb6

### Video Upload
https://github.com/user-attachments/assets/6b8c08a8-c84f-4fc3-83ad-5703f474fc1b

## Features

- 🎯 **Text-Guided Separation** - Describe what you want: "vocals", "drums", "a dog barking", "crowd noise"
- 🎬 **Video Support** - Separate a video's soundtrack and download the video with the new audio
- 🕒 **Temporal Lock** - Mark a region where the sound is audible to point the model at the right source
- 🚀 **Consumer GPUs** - Small runs in ~4.2 GB of VRAM; long files are processed in crossfaded chunks
- ⚡ **Speed / Quality presets** - Fast (8 steps), Balanced (16, Meta's default), High (32)
- 🎧 **Stereo** - Optional per-channel separation that keeps the stereo image
- 🎛️ **Stem Mixer** - Play, mute and compare the isolated sound and the rest
- 💾 **Exports** - WAV (24-bit), FLAC, MP3, or everything as a ZIP

## ⚡ Performance

4:26 stereo song (`"singing voice"`), RTX 4090 (24 GB), Windows 11, 25 s chunks, Balanced quality.
"v1.0" is v1.0's inference path run on the same machine with the same weights.

| Model | v1.0 | **v1.1** | v1.1 Fast | v1.1 Stereo | Peak VRAM v1.1 (auto batch / low-VRAM) |
|---|---|---|---|---|---|
| Small | 95 s | **17 s** (16× realtime) | 14 s | 31 s | 6.3 GB / 4.2 GB |
| Base | 104 s | **20 s** (14× realtime) | 15 s | 39 s | 7.5 GB / 5.4 GB |
| Large | 120 s | **30 s** (9× realtime) | 20 s | 60 s | 11.0 GB / 8.9 GB |

- The batch size is chosen from free VRAM; on smaller GPUs AudioGhost automatically runs one chunk at a time and decodes one clip at a time ("low-VRAM" column: ~3× slower than the auto batch, still faster than v1.0). Shorter chunks need even less memory.
- Loading a downloaded model is disk-bound (reading a 5–15 GB checkpoint): a few seconds from a fast SSD with a warm cache, up to ~1.5 min for Large from a slow disk.
- First use of a model also downloads its weights: Small 5.1 GB, Base 7.7 GB, Large 14.9 GB (+0.9 GB text encoder once). v1.0 additionally downloaded ~25 GB of ImageBind (4.8 GB), CLAP (1.9 GB), SAM-Audio-Judge (12.3 GB) and PE-A-Frame (5.8 GB) weights it never used.
- Reproduce on your machine: `python scripts/benchmark.py --file song.mp3 --model base`.

### What changed under the hood

| | v1.0 | v1.1 |
|---|---|---|
| Model loading | `from_pretrained()` builds every sub-model (vision encoder, ImageBind, CLAP, Judge, span predictor), then deletes them | Builds only what text prompting uses, skips random init, memory-maps the checkpoint |
| Audio codec | cuDNN disabled by SAM-Audio → slow fallback convolutions and a per-timestep LSTM loop | Same computation with cuDNN (falls back automatically if cuDNN rejects a shape) |
| Precision | Whole model bf16, wrapped in fp16 `autocast` (so "float32" mode wasn't fp32 either) | fp16 transformer; T5, codec, RoPE and the ODE state in fp32 |
| Chunking | Hard cuts every 25 s, tail < 1 s dropped, one chunk at a time | Equal chunks, 2 s crossfades, shared noise, batched |

## 🗺️ Roadmap

- 🖱️ **Visual Prompting** - click an object in the video to isolate its sound (SAM 2.1 / [SAM 3.1](https://huggingface.co/facebook/sam3.1) masks + SAM-Audio `-tv`)
- 🎼 **Multi-stem in one pass** - e.g. vocals + drums + bass from a single run
- 🎚️ **Specialised engines** for standard music stems (BS-/Mel-RoFormer) next to SAM-Audio for "any sound"
- 🍎 **macOS / Linux scripts** (community PR [#8](https://github.com/0x0funky/audioghost-ai/pull/8))

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Frontend                       │
│             (Next.js + Tailwind v4)             │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│               Backend API                        │
│            (FastAPI + Python)                    │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              Task Queue                          │
│          (Celery + Redis)                        │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│        SAM-Audio Lite engine (sam_lite.py)       │
│  lite loader · mixed precision · batched,        │
│  crossfaded chunks · cuDNN codec                 │
└─────────────────────────────────────────────────┘
```

## Requirements

- **Windows 10/11** (one-click scripts) or Linux (manual setup)
- **NVIDIA GPU**: 6 GB+ for Small, 8 GB+ for Base, 12 GB+ for Large (measured peaks: 4.2 / 5.4 / 8.9 GB with 25 s chunks; shorter chunks need less)
- **Anaconda / Miniconda** and **Node.js 18+**
- A **HuggingFace account** with access to [facebook/sam-audio-large](https://huggingface.co/facebook/sam-audio-large) (gated; approval can take a while)

> 💡 PyTorch, FFmpeg and Redis are installed by `install.bat`. RTX 50-series cards get the CUDA 12.8 build automatically.

## 🚀 One-Click Installation (Recommended)

```bash
# First time: creates the 'audioghost' conda env, installs everything (pinned), downloads Redis
install.bat

# Every day
start.bat     # starts Redis, API, worker and the web UI, then opens http://localhost:3000
stop.bat      # stops only AudioGhost's own windows
```

Updating an existing install: `git pull`, then run `install.bat` again (it re-applies the version pins).

---

## Manual Setup (Advanced)

### 1. Start Redis

`install.bat` downloads Redis to `redis/`. With Docker instead:
```bash
docker-compose up -d
```

### 2. Create the environment

```bash
conda create -n audioghost python=3.11 -y
conda activate audioghost
```

### 3. Install PyTorch 2.9

```bash
# RTX 20/30/40-series (CUDA 12.6)
pip install torch==2.9.0+cu126 torchvision==0.24.0+cu126 torchaudio==2.9.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
# RTX 50-series: replace cu126 with cu128 (all three packages and the index URL)
```

### 4. Install FFmpeg 7 (required by TorchCodec)

```bash
conda install -c conda-forge "ffmpeg=7" -y
```

### 5. Install SAM-Audio (pinned)

```bash
pip install "git+https://github.com/facebookresearch/sam-audio.git@bb4c6999d2677c7402360e426afc01ddfad6dce0"
```

### 6. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt   # also pins transformers<5 and huggingface_hub<1.0
```

### 7. Install frontend dependencies

```bash
cd frontend
npm ci
```

### 8. Start services

**Terminal 1 - Backend API:**
```bash
cd backend
uvicorn main:app --port 8000
```

**Terminal 2 - Celery Worker:**
```bash
conda activate audioghost
cd backend
celery -A workers.celery_app worker --loglevel=info --pool=solo
```

**Terminal 3 - Frontend:**
```bash
cd frontend
npm run dev
```

### 9. Open the app

Navigate to `http://localhost:3000`

### 10. Connect HuggingFace

1. Click **Connect HuggingFace**
2. Request access at https://huggingface.co/facebook/sam-audio-large (if a `-tv` Precision model later returns 403, request access on its page too)
3. Create an access token: https://huggingface.co/settings/tokens
4. Paste the token and connect

## Usage

1. **Upload** an audio or video file (drag & drop, click, or Tab + Enter)
2. **Describe** what to extract or remove: "vocals", "drums", "background music", "a dog barking", "crowd noise", "a person speaking"
3. Optional:
   - **Temporal Lock** (audio files): select a region where the sound is clearly audible - it's a hint, the whole file is still processed
   - **Speed / Quality**: Fast for drafts, High for final exports
   - **Precision model (-tv)**: Meta's checkpoints that pick the right sound more reliably
   - **Keep stereo**: separate left and right channels (about 2× processing)
4. Click **Extract** or **Remove** (both stems are always produced). Cancel any time.
5. **Preview** in the Stem Mixer and **download** single stems or a ZIP as WAV, FLAC or MP3. For videos you can also download the video with the new soundtrack.

### Command line

```bash
python sam_audio_lite.py song.mp3 "singing voice"
python sam_audio_lite.py interview.mp4 "background music" --model base --quality fast -o out/
python sam_audio_lite.py live.flac "crowd noise" --stereo --start 12.5 --end 16 --format flac
```

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `AUDIOGHOST_PRECISION` | `fp16` | Transformer precision when High Quality is off (`fp16` or `bf16`) |
| `AUDIOGHOST_PRELOAD_MODEL` | - | `small` / `base` / `large`: load a model when the worker starts |
| `AUDIOGHOST_DEVICE` | auto | Force `cuda`, `cuda:1`, `cpu` (or `mps`, experimental) |
| `AUDIOGHOST_SAFE_CODEC` | `0` | `1` = use SAM-Audio's original (slow, cuDNN-off) codec path |
| `AUDIOGHOST_RETENTION_HOURS` | `24` | Delete uploads/results older than this (`0` = keep) |
| `AUDIOGHOST_MAX_UPLOAD_MB` | `2048` | Upload size limit |
| `AUDIOGHOST_REDIS_URL` | `redis://localhost:6379/0` | Celery broker / result backend |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | Where the web UI finds the API |

## API Reference

Interactive docs: `http://localhost:8000/docs`

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/separate/` | Create a task (multipart form, see below) |
| `POST` | `/api/separate/batch` | One task per description for the same file (`descriptions` = JSON array) |
| `GET` | `/api/tasks/{task_id}` | Status, progress, queue position, result |
| `DELETE` | `/api/tasks/{task_id}` | Cancel (running tasks stop at the next diffusion step) |
| `GET` | `/api/tasks/{task_id}/download/{original\|ghost\|clean\|video}` | Result file; add `?format=wav\|flac\|mp3` to download with a readable name |
| `GET` | `/api/tasks/{task_id}/download-all?format=wav` | ZIP with all stems |
| `GET` | `/api/tasks/{task_id}/download-video-with-audio/{original\|ghost\|clean}` | Video with the audio track replaced |
| `GET` | `/api/system/status` | Redis / worker status |

`POST /api/separate/` form fields: `file`, `description`, `mode` (`extract`/`remove`), `model_size` (`small`/`base`/`large`), `model_variant` (`default`/`tv`), `quality` (`fast`/`balanced`/`high`), `chunk_duration` (5-60), `use_float32`, `stereo`, optional `start_time` + `end_time` (Temporal Lock) and `seed`.

```json
{ "task_id": "uuid", "status": "pending", "message": "Task submitted successfully" }
```

`ghost` = the described sound, `clean` = everything else.

## Troubleshooting

### `TypeError: BaseModel._from_pretrained() missing 2 required keyword-only arguments: 'proxies' and 'resume_download'`
Your `huggingface_hub` is 1.x. Run `pip install -r backend/requirements.txt` (pins `transformers<5`, `huggingface_hub<1.0`) or re-run `install.bat`.

### `Could not load libtorchcodec` / TorchCodec DLL errors
TorchCodec needs FFmpeg's shared DLLs (FFmpeg 4-7). Inside the env run `conda install -c conda-forge "ffmpeg=7" -y`, or install the FFmpeg 7 **shared** build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add its `bin` folder to `PATH`, then restart the terminal.

### "Entry point not found ... libintl / gdk_pixbuf-2.0-0.dll" during install ([#10](https://github.com/0x0funky/audioghost-ai/issues/10))
Another program's GTK DLLs on `PATH` (GIMP, Inkscape, Git's `mingw64\bin`, ...) clash with conda's. Remove those folders from `PATH` for the install window, or use the gyan.dev FFmpeg build above, then re-run `install.bat`.

### `CUDA error: no kernel image is available for execution on the device`
RTX 50-series GPUs need the CUDA 12.8 PyTorch build - re-run `install.bat`, or reinstall PyTorch with `cu128` (manual setup step 3).

### Out of memory / extremely slow processing
On Windows, running out of VRAM often doesn't crash - the driver spills into system RAM and everything slows down massively. Use the Small model, a shorter max chunk length (e.g. 15 s), turn off Stereo and High Quality, and close other GPU apps. Peak usage is listed in the [performance table](#-performance).

### The page at localhost:3000 is a different app
Something else already uses port 3000, so Next.js picked the next free port (3001, 3002, ...) - check the "AudioGhost Frontend" window for the URL. The API accepts local ports 3000-3099.

### The UI says the worker or Redis isn't running
Use `start.bat`, which starts Redis, the API, the worker and the UI. For manual setups, start Redis before the worker ([#5](https://github.com/0x0funky/audioghost-ai/issues/5)).

### HuggingFace 401 / 403
Reconnect in the UI. 403 means your account hasn't been granted access to that model yet - open the model's page on HuggingFace (the `-tv` checkpoints are separate repos) and request access there.

## Project Structure

```
audioghost-ai/
├── backend/
│   ├── main.py               # FastAPI app
│   ├── settings.py           # paths & env configuration
│   ├── api/
│   │   ├── auth.py           # HuggingFace auth
│   │   ├── separate.py       # create tasks (validation, streaming upload)
│   │   └── tasks.py          # status, cancel, downloads, ZIP, video remux
│   ├── workers/
│   │   ├── celery_app.py     # Celery config
│   │   ├── tasks.py          # job lifecycle (progress, cancel, outputs)
│   │   ├── sam_lite.py       # SAM-Audio Lite engine
│   │   ├── audio_io.py       # decode / encode / export helpers
│   │   └── queue_state.py    # queue position, cancel flags
│   └── tests/                # unit tests (python -m pytest backend/tests)
├── frontend/                 # Next.js app (src/app, src/components, src/lib)
├── scripts/benchmark.py      # performance benchmark
├── sam_audio_lite.py         # command-line tool
├── install.bat / start.bat / stop.bat
├── CHANGELOG.md
└── QUICKSTART.md             # 中文快速指南
```

## License

AudioGhost is licensed under the MIT License. SAM-Audio code and weights are licensed separately by Meta (see the [SAM-Audio repository](https://github.com/facebookresearch/sam-audio) and model cards) - check those terms before commercial use. SAM-Audio's codec embeds an inaudible watermark in generated audio; AudioGhost leaves it intact.

## Credits

- [SAM-Audio](https://github.com/facebookresearch/sam-audio) by Meta AI Research
- **Core Optimization Logic**: Special thanks to [NilanEkanayake](https://github.com/NilanEkanayake) for the initial modifications in [Issue #24](https://github.com/facebookresearch/sam-audio/issues/24) that made VRAM reduction possible, and to [oulianov](https://github.com/oulianov) for the "don't build unused modules" approach in [PR #3](https://github.com/0x0funky/audioghost-ai/pull/3)
- Built with ❤️ using Next.js, FastAPI, and Celery
