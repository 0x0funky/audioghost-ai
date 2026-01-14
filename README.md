# AudioGhost AI 🎵👻

![AudioGhost Banner](banner.png)

**AI-Powered Object-Oriented Audio Separation**

Describe the sound you want to extract or remove using natural language. Powered by Meta's [SAM-Audio](https://github.com/facebookresearch/sam-audio) model.

![Demo](https://img.shields.io/badge/status-MVP%20v1.0-green) ![Python](https://img.shields.io/badge/python-3.11+-blue) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

## 🎬 Demo

### Audio Separation
https://github.com/user-attachments/assets/49248e25-0c56-46ab-a821-2de7f7016bb6

### Video Upload
https://github.com/user-attachments/assets/6b8c08a8-c84f-4fc3-83ad-5703f474fc1b

## Features

- 🎯 **Text-Guided Separation** - Describe what you want to extract: "vocals", "drums", "a dog barking"
- 🎬 **Video Upload Support** - Upload videos and extract/remove audio sources (audio extraction only, not vision-based)
- 🚀 **Memory Optimized** - Lite mode reduces VRAM from ~11GB to ~4GB
- 🎨 **Modern UI** - Glassmorphism design with waveform visualization
- ⚡ **Real-time Progress** - Track separation progress in real-time
- 🎛️ **Stem Mixer** - Preview and compare original, extracted, and residual audio

## 🗺️ Roadmap

- 🖱️ **Visual Prompting** - Click on video to select sound sources visually (Integration with [SAM 2](https://github.com/facebookresearch/sam2))

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

## Requirements

### Windows
- **Python 3.11+**
- **CUDA-compatible GPU** (4GB+ VRAM for lite mode, 12GB+ for full mode)
- **CUDA 12.6** (recommended)
- **Node.js 18+** (for frontend)

### macOS
- **Python 3.11+**
- **Apple Silicon (M1/M2/M3/M4)** or Intel Mac
- **Node.js 18+** (for frontend)
- **Homebrew** (for Redis installation)

> 💡 **Note for Mac users**: The application uses CPU processing to avoid MPS (Metal Performance Shaders) compatibility issues. This is slower but more stable.

### Linux
- **Python 3.11+**
- **CUDA-compatible GPU** (4GB+ VRAM for lite mode, 12GB+ for full mode) or CPU
- **CUDA 12.6** (recommended for GPU)
- **Node.js 18+** (for frontend)

> 💡 FFmpeg and Redis are automatically installed by the installer on all platforms.

## 🚀 One-Click Installation (Recommended)

### Windows

**First Time Setup:**
```bash
# Run installer (creates Conda env, downloads Redis, installs all dependencies)
windows\install.bat
```

**Daily Usage:**
```bash
# Start all services with one click
windows\start.bat

# Stop all services
windows\stop.bat
```

### macOS / Linux

**First Time Setup:**
```bash
# Run installer (creates Conda env, installs Redis, installs all dependencies)
chmod +x unix/*.sh
./unix/install.sh
```

**Daily Usage:**
```bash
# Start all services with one click
./unix/start.sh

# Stop all services
./unix/stop.sh
```

---

## Manual Setup (Advanced)

### 1. Start Redis

Redis is automatically downloaded to `redis/` folder by `install.bat`. If you prefer Docker:
```bash
docker-compose up -d
```

### 2. Create Anaconda Environment

```bash
# Create new environment (Python 3.11+ required)
conda create -n audioghost python=3.11 -y

# Activate environment
conda activate audioghost
```

### 3. Install PyTorch (CUDA 12.6)

```bash
pip install torch==2.9.0+cu126 torchvision==0.24.0+cu126 torchaudio==2.9.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
```

### 4. Install FFmpeg (required by TorchCodec)

```bash
conda install -c conda-forge ffmpeg -y
```

### 5. Install SAM Audio

```bash
pip install git+https://github.com/facebookresearch/sam-audio.git
```

### 6. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 7. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 8. Start Services

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

### 9. Open the App

Navigate to `http://localhost:3000`

### 10. Connect HuggingFace

1. Click "Connect HuggingFace" button
2. Request access at https://huggingface.co/facebook/sam-audio-large
3. Create Access Token: https://huggingface.co/settings/tokens
4. Paste the token and connect



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

| Model | bfloat16 (Default) | float32 (High Quality) | Recommended GPU |
|-------|-------------------|------------------------|-----------------|
| Small | **~6 GB** | **~10 GB** | RTX 3060 6GB / RTX 3070 8GB |
| Base | **~7 GB** | **~13 GB** | RTX 3070/4060 8GB / RTX 4070 12GB |
| Large | **~10 GB** | **~20 GB** | RTX 3080/4070 12GB / RTX 4080 16GB |

> 💡 **High Quality Mode (float32)**: Better separation quality but uses +2-3GB more VRAM. Enable via the "High Quality Mode" toggle in the UI.

### Processing Time

| Model | First Run (incl. model load) | Subsequent Runs | Speed |
|-------|------------------------------|-----------------|-------|
| Small | ~78s | **~25s** | ~10x realtime |
| Base | ~100s | **~29s** | ~9x realtime |
| Large | ~130s | **~41s** | ~6.5x realtime |

> 💡 First run includes model download and loading. Subsequent runs use cached models.

### Memory Optimization Details

AudioGhost uses a "Lite Mode" that removes unused model components:

| Component Removed | VRAM Saved |
|-------------------|------------|
| Vision Encoder | ~2GB |
| Visual Ranker | ~2GB |
| Text Ranker | ~2GB |
| Span Predictor | ~1-2GB |

**Total Reduction**: Up to **40% less VRAM** compared to original SAM-Audio

This is achieved by:
- Disabling video-related features (not needed for audio-only)
- Using `predict_spans=False` and `reranking_candidates=1`
- Using `bfloat16` precision by default (optional float32 for quality)
- 25-second chunking for long audio files

## Project Structure

```
audioghost-ai/
├── backend/
│   ├── main.py           # FastAPI app
│   ├── api/              # API routes
│   │   ├── auth.py       # HuggingFace auth
│   │   └── separate.py   # Separation endpoints
│   └── workers/
│       ├── celery_app.py # Celery config
│       └── tasks.py      # SAM Audio Lite worker (with Mac CPU fix)
├── frontend/
│   ├── src/
│   │   ├── app/          # Next.js app
│   │   └── components/   # React components
│   └── package.json
├── windows/              # Windows-specific scripts
│   ├── install.bat       # One-click installer for Windows
│   ├── start.bat         # Start all services (Windows)
│   └── stop.bat          # Stop all services (Windows)
├── unix/                 # macOS/Linux scripts
│   ├── install.sh        # One-click installer for Unix
│   ├── start.sh          # Start all services (Unix)
│   └── stop.sh           # Stop all services (Unix)
├── sam_audio_lite.py     # Standalone lite version
├── QUICKSTART.md         # Quick setup guide
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

**Response:**
```json
{
  "task_id": "uuid",
  "status": "pending",
  "message": "Task submitted successfully"
}
```

### GET /api/separate/{task_id}/status

Get task status and progress.

### GET /api/separate/{task_id}/download/{stem}

Download result audio (ghost, clean, or original).

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
- Check that `.hf_token` exists in `backend/`

## License

This project is licensed under the MIT License. SAM-Audio is licensed by Meta under a research license.

## Credits

- [SAM-Audio](https://github.com/facebookresearch/sam-audio) by Meta AI Research
- **Core Optimization Logic**: Special thanks to [NilanEkanayake](https://github.com/NilanEkanayake) for providing the initial code modifications in [Issue #24](https://github.com/facebookresearch/sam-audio/issues/24) that made VRAM inference reduction possible.
- Built with ❤️ using Next.js, FastAPI, and Celery
