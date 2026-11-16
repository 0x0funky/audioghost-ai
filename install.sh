#!/usr/bin/env bash
# AudioGhost AI - one-click installer for macOS (Apple Silicon) and Linux (x86_64).
# Windows: use install.bat. Based on community PR #8 by @ShmuelOps.
set -eo pipefail

SAM_AUDIO_COMMIT="bb4c6999d2677c7402360e426afc01ddfad6dce0"
PERCEPTION_MODELS_COMMIT="e72b6810b1133e1c879f2cc965d276eb73803f1f"
TORCH_VERSION="2.9.0"
TORCHVISION_VERSION="0.24.0"
ENV_NAME="audioghost"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
OS="$(uname -s)"
ARCH="$(uname -m)"
CONSTRAINTS="$ROOT/backend/constraints.txt"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           AudioGhost AI - One-Click Installer               ║"
echo "║                  v1.2 (macOS / Linux)                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# --------------------------------------------------------------- Checks ----
if [[ "$OS" == "Darwin" && "$ARCH" != "arm64" ]]; then
    echo "[ERROR] Intel Macs are not supported: PyTorch 2.9 only ships Apple Silicon builds for macOS."
    exit 1
fi
if [[ "$OS" == "Linux" && "$ARCH" != "x86_64" ]]; then
    echo "[ERROR] Linux on $ARCH is not supported yet (TorchCodec / xformers have no wheels for it)."
    exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] Conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
    echo "[ERROR] Node.js (npm) not found. Install Node.js 18+ first: https://nodejs.org"
    exit 1
fi
if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] git not found (needed to install SAM-Audio). Install git first."
    exit 1
fi

# ---------------------------------------------------------------- Redis ----
echo "[1/8] Checking Redis..."
if command -v redis-server >/dev/null 2>&1; then
    echo "      redis-server found: $(command -v redis-server)"
elif [[ "$OS" == "Darwin" ]]; then
    if command -v brew >/dev/null 2>&1; then
        brew install redis
    else
        echo "      [WARN] Homebrew not found. Install Redis yourself (https://brew.sh, then 'brew install redis')"
        echo "             or run it with Docker: docker compose up -d"
    fi
else
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y redis-server
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y redis
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y redis
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --noconfirm redis
    else
        echo "      [WARN] Could not install Redis automatically - install it or run: docker compose up -d"
    fi
fi

# ------------------------------------------------------------ Conda env ----
echo ""
echo "[2/8] Creating conda environment '$ENV_NAME' (Python 3.11)..."
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "      Environment already exists, reusing it"
else
    conda create -n "$ENV_NAME" python=3.11 -y
fi

echo ""
echo "[3/8] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# -------------------------------------------------------------- PyTorch ----
echo ""
echo "[4/8] Installing PyTorch $TORCH_VERSION..."
if [[ "$OS" == "Darwin" ]]; then
    echo "      macOS: runs on the CPU by default (Apple GPU/MPS is experimental, see README)"
    pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCH_VERSION"
else
    TORCH_CUDA="cpu"
    if command -v nvidia-smi >/dev/null 2>&1; then
        TORCH_CUDA="cu126"
        # RTX 50-series (compute capability 12.x) needs the CUDA 12.8 build
        while read -r cap; do
            major="${cap%%.*}"
            if [[ "$major" =~ ^[0-9]+$ ]] && (( major >= 12 )); then TORCH_CUDA="cu128"; fi
        done < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)
    else
        echo "      [WARN] nvidia-smi not found - installing the CPU build (very slow)"
    fi
    echo "      Using $TORCH_CUDA wheels"
    if [[ "$TORCH_CUDA" == "cpu" ]]; then
        pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCH_VERSION" \
            --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
    else
        pip install "torch==$TORCH_VERSION+$TORCH_CUDA" "torchvision==$TORCHVISION_VERSION+$TORCH_CUDA" \
            "torchaudio==$TORCH_VERSION+$TORCH_CUDA" \
            --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" --extra-index-url https://pypi.org/simple
    fi
fi

# --------------------------------------------------------------- FFmpeg ----
echo ""
echo "[5/8] Installing FFmpeg 7 (required by TorchCodec)..."
conda install -c conda-forge "ffmpeg=7" -y

# ------------------------------------------------------------ SAM-Audio ----
echo ""
echo "[6/8] Installing SAM Audio (pinned commit ${SAM_AUDIO_COMMIT:0:7})..."
if [[ "$OS" == "Darwin" ]]; then
    # decord and xformers (pulled in by Meta's perception_models) have no Apple
    # Silicon wheels and aren't used for text prompting, so install SAM-Audio
    # without them. AudioGhost provides a stand-in for xformers at runtime.
    pip install -c "$CONSTRAINTS" -r backend/requirements-lean.txt
    pip install --no-deps "git+https://github.com/facebookresearch/perception_models@$PERCEPTION_MODELS_COMMIT"
    pip install --no-deps "git+https://github.com/facebookresearch/sam-audio.git@$SAM_AUDIO_COMMIT"
else
    pip install -c "$CONSTRAINTS" "git+https://github.com/facebookresearch/sam-audio.git@$SAM_AUDIO_COMMIT"
fi

echo ""
echo "[7/8] Installing backend dependencies (also pins transformers / huggingface_hub)..."
pip install -c "$CONSTRAINTS" -r backend/requirements.txt

echo ""
echo "[8/8] Installing frontend dependencies..."
(cd frontend && (npm ci || npm install))

# --------------------------------------------------------------- Verify ----
echo ""
echo "Verifying the installation..."
if ! (cd backend && python -c "
import torch
from workers import sam_lite  # installs platform shims
import torchcodec, sam_audio
print('  PyTorch', torch.__version__, '| device:', sam_lite.resolve_device())
print('  TorchCodec + FFmpeg: OK')
print('  SAM-Audio: OK')
"); then
    echo ""
    echo "[ERROR] The Python environment is not working yet."
    echo "        'Could not load libtorchcodec' means FFmpeg's shared libraries were not found:"
    echo "        re-run this installer, or install FFmpeg 7 (brew install ffmpeg@7 / your package manager)."
    echo "        See README.md - Troubleshooting."
    exit 1
fi

chmod +x "$ROOT/start.sh" "$ROOT/stop.sh" 2>/dev/null || true

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║             Installation Complete! ✓                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║   Start:  ./start.sh        Stop:  ./stop.sh                ║"
echo "║   The first separation downloads the model (5-15 GB).       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
