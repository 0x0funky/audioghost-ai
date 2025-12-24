# AudioGhost AI 🎵👻

![AudioGhost Banner](banner.png)

**AI-Powered Object-Oriented Audio Separation**

Describe the sound you want to extract or remove using natural language. Powered by Meta's [SAM-Audio](https://github.com/facebookresearch/sam-audio) model.

![Demo](https://img.shields.io/badge/status-v1.1-green) ![Python](https://img.shields.io/badge/python-3.11+-blue) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

## 🎬 Demo

https://github.com/user-attachments/assets/49248e25-0c56-46ab-a821-2de7f7016bb6

## Features

- 🎯 **Text-Guided Separation** - Describe what you want to extract: "vocals", "drums", "a dog barking"
- 🚀 **Memory Optimized** - Lite mode reduces VRAM from ~11GB to ~4GB
- 🎨 **Modern UI** - Glassmorphism design with waveform visualization
- ⚡ **Real-time Progress** - Track separation progress in real-time
- 🎛️ **Stem Mixer** - Preview and compare original, extracted, and residual audio
- 🐳 **Docker Support** - One-command setup for any platform

## 🗺️ Roadmap

- 🎬 **Video Support** - Upload videos and separate audio sources visually
- 🖱️ **Visual Prompting** - Click on video to select sound sources (Integration with [SAM 3](https://github.com/facebookresearch/sam2))

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
│           SAM Audio Lite                         │
│    (Memory-optimized Meta SAM-Audio)            │
└─────────────────────────────────────────────────┘
```

---

## 🐳 Docker Installation (Recommended)

The easiest way to run AudioGhost AI on any platform (Windows, macOS, Linux).

### Prerequisites

- **Docker Desktop** ([Windows/Mac](https://www.docker.com/products/docker-desktop/)) or **Docker Engine** ([Linux](https://docs.docker.com/engine/install/))
- **For GPU mode**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/LunarECL/audioghost-ai.git
cd audioghost-ai

# 2. Copy and configure environment
cp env.example .env
# Edit .env and add your HuggingFace token (HF_TOKEN=hf_xxx)

# 3. Start all services (CPU mode)
docker compose up --build

# 4. Open http://localhost:3000
```

### GPU Mode (Linux / Windows WSL2)

```bash
# Requires NVIDIA Container Toolkit
docker compose --profile gpu up --build
```

### Stopping Services

```bash
docker compose down
```

### Updating

```bash
git pull
docker compose build --no-cache
docker compose up
```

---

## 💻 Local Installation (Advanced)

For development or if you prefer not to use Docker.

### Requirements

- **Python 3.11+**
- **CUDA-compatible GPU** (4GB+ VRAM for lite mode, 12GB+ for full mode)
- **CUDA 12.6** (recommended)
- **Node.js 18+** (for frontend)

> 💡 FFmpeg and Redis are automatically installed by the installer.

### 🚀 One-Click Installation (Windows)

#### First Time Setup

```bash
# Run installer (creates Conda env, downloads Redis, installs all dependencies)
install.bat
```

#### Daily Usage

```bash
# Start all services with one click
start.bat

# Stop all services
stop.bat
```

---

### Manual Setup (All Platforms)

#### 1. Start Redis

Using Docker (recommended):

```bash
docker run -d --name redis -p 6379:6379 redis:alpine
```

Or install locally: [Redis Installation Guide](https://redis.io/docs/getting-started/installation/)

#### 2. Create Anaconda Environment

```bash
# Create new environment (Python 3.11+ required)
conda create -n audioghost python=3.11 -y

# Activate environment
conda activate audioghost
```

#### 3. Install PyTorch

**With CUDA (NVIDIA GPU):**

```bash
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

**With MPS (Apple Silicon):**

```bash
pip install torch==2.4.0 torchaudio==2.4.0
```

**CPU Only:**

```bash
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

#### 4. Install FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (with Conda)
conda install -c conda-forge ffmpeg -y
```

#### 5. Install SAM Audio

```bash
pip install git+https://github.com/facebookresearch/sam-audio.git
```

#### 6. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

#### 7. Install Frontend Dependencies

```bash
cd frontend
npm install
```

#### 8. Start Services

**Terminal 1 - Backend API:**

```bash
cd backend
uvicorn main:app --reload --port 8000
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

#### 9. Open the App

Navigate to `http://localhost:3000`

#### 10. Connect HuggingFace

1. Click "Connect HuggingFace" button
2. Request access at https://huggingface.co/facebook/sam-audio-large
3. Create Access Token: https://huggingface.co/settings/tokens
4. Paste the token and connect

---

## Usage

1. **Upload** an audio file (MP3, WAV, FLAC)
2. **Describe** what you want to extract or remove:
   - "vocals" / "singing voice"
   - "drums" / "percussion"
   - "background music"
   - "a dog barking"
   - "crowd noise"
3. Click **Extract** or **Remove**
4. Wait for processing
5. **Preview** and **download** the results

## Performance Benchmarks

> Tested on RTX 4090 with 4:26 audio (11 chunks @ 25s each)

### VRAM Usage (Lite Mode)

| Model | bfloat16 (Default) | float32 (High Quality) | Recommended GPU                    |
| ----- | ------------------ | ---------------------- | ---------------------------------- |
| Small | **~6 GB**          | **~10 GB**             | RTX 3060 6GB / RTX 3070 8GB        |
| Base  | **~7 GB**          | **~13 GB**             | RTX 3070/4060 8GB / RTX 4070 12GB  |
| Large | **~10 GB**         | **~20 GB**             | RTX 3080/4070 12GB / RTX 4080 16GB |

> 💡 **High Quality Mode (float32)**: Better separation quality but uses +2-3GB more VRAM. Enable via the "High Quality Mode" toggle in the UI.

### Processing Time

| Model | First Run (incl. model load) | Subsequent Runs | Speed          |
| ----- | ---------------------------- | --------------- | -------------- |
| Small | ~78s                         | **~25s**        | ~10x realtime  |
| Base  | ~100s                        | **~29s**        | ~9x realtime   |
| Large | ~130s                        | **~41s**        | ~6.5x realtime |

> 💡 First run includes model download and loading. Subsequent runs use cached models.

### Memory Optimization Details

AudioGhost uses a "Lite Mode" that removes unused model components:

| Component Removed | VRAM Saved |
| ----------------- | ---------- |
| Vision Encoder    | ~2GB       |
| Visual Ranker     | ~2GB       |
| Text Ranker       | ~2GB       |
| Span Predictor    | ~1-2GB     |

**Total Reduction**: Up to **40% less VRAM** compared to original SAM-Audio

This is achieved by:

- Disabling video-related features (not needed for audio-only)
- Using `predict_spans=False` and `reranking_candidates=1`
- Using `bfloat16` precision by default (optional float32 for quality)
- 25-second chunking for long audio files

## Environment Variables

| Variable                 | Description                               | Default     |
| ------------------------ | ----------------------------------------- | ----------- |
| `HF_TOKEN`               | HuggingFace access token                  | -           |
| `DEVICE`                 | Device mode: `auto`, `cpu`, `cuda`, `mps` | `auto`      |
| `REDIS_HOST`             | Redis hostname                            | `localhost` |
| `REDIS_PORT`             | Redis port                                | `6379`      |
| `DEFAULT_MODEL_SIZE`     | Default model: `small`, `base`, `large`   | `base`      |
| `CPU_DEFAULT_MODEL_SIZE` | Model for CPU mode                        | `small`     |

See `env.example` for all available options.

## Project Structure

```
audioghost-ai/
├── backend/
│   ├── main.py           # FastAPI app
│   ├── config.py         # Centralized configuration
│   ├── Dockerfile        # CPU Docker image
│   ├── Dockerfile.gpu    # GPU Docker image
│   ├── api/              # API routes
│   │   ├── auth.py       # HuggingFace auth
│   │   ├── separate.py   # Separation endpoints
│   │   └── tasks.py      # Task status endpoints
│   └── workers/
│       ├── celery_app.py # Celery config
│       └── tasks.py      # SAM Audio Lite worker
├── frontend/
│   ├── Dockerfile        # Frontend Docker image
│   ├── src/
│   │   ├── app/          # Next.js app
│   │   ├── components/   # React components
│   │   └── lib/          # Utilities & API client
│   └── package.json
├── docker-compose.yml    # Docker orchestration
├── env.example           # Environment template
└── README.md
```

## API Reference

### POST /api/separate/

Create a separation task.

**Form Data:**

- `file` - Audio file
- `description` - Text prompt (e.g., "vocals")
- `mode` - "extract" or "remove"
- `model_size` - "small", "base", or "large" (default: "base")
- `chunk_duration` - Chunk size in seconds (default: 25)
- `use_float32` - High quality mode (default: "false")

**Response:**

```json
{
  "task_id": "uuid",
  "status": "pending",
  "message": "Task submitted successfully"
}
```

### GET /api/tasks/{task_id}

Get task status and progress.

### GET /api/tasks/{task_id}/download/{stem}

Download result audio (ghost, clean, or original).

### GET /api/auth/status

Check authentication status and device info.

## Troubleshooting

### CUDA Out of Memory

- Use `model_size: "small"` instead of "base" or "large"
- Ensure lite mode is enabled (check for "Optimizing model for low VRAM" in logs)
- Close other GPU applications

### TorchCodec DLL Error

- Downgrade to FFmpeg 7.x
- Ensure FFmpeg `bin` directory is in PATH

### HuggingFace 401 Error

- Re-authenticate via the UI
- Check that `HF_TOKEN` environment variable is set (Docker) or `.hf_token` exists in `backend/`

### Docker: Container exits immediately

- Check logs: `docker compose logs api`
- Ensure `.env` file exists with valid `HF_TOKEN`
- Wait for model download on first run (can take several minutes)

### Docker: GPU not detected

- Verify NVIDIA Container Toolkit is installed: `nvidia-smi`
- Use `docker compose --profile gpu up` (not default profile)
- Check GPU availability: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`

## License

This project is licensed under the MIT License. SAM-Audio is licensed by Meta under a research license.

## Credits

- [SAM-Audio](https://github.com/facebookresearch/sam-audio) by Meta AI Research
- **Core Optimization Logic**: Special thanks to [NilanEkanayake](https://github.com/NilanEkanayake) for providing the initial code modifications in [Issue #24](https://github.com/facebookresearch/sam-audio/issues/24) that made VRAM inference reduction possible.
- Built with ❤️ using Next.js, FastAPI, and Celery
