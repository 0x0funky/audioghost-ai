#!/bin/bash
set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           AudioGhost AI - One-Click Installer               ║"
echo "║                   v1.0 MVP                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Get the project root directory (parent of the script directory)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda not found. Please install Anaconda or Miniconda first."
    echo "Download from: https://www.anaconda.com/download"
    exit 1
fi

# Detect OS and install Redis if needed
echo "[1/8] Checking Redis..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use Homebrew
    if ! command -v redis-server &> /dev/null; then
        echo "Installing Redis via Homebrew..."
        if ! command -v brew &> /dev/null; then
            echo "[ERROR] Homebrew not found. Install from: https://brew.sh"
            exit 1
        fi
        brew install redis
    else
        echo "Redis already installed, skipping..."
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - use package manager
    if ! command -v redis-server &> /dev/null; then
        echo "Installing Redis..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y redis-server
        elif command -v yum &> /dev/null; then
            sudo yum install -y redis
        else
            echo "[WARN] Could not install Redis automatically. Please install manually."
        fi
    else
        echo "Redis already installed, skipping..."
    fi
fi

echo ""
echo "[2/8] Creating Conda environment 'audioghost' (Python 3.11)..."
conda create -n audioghost python=3.11 -y || echo "[WARN] Environment may already exist, continuing..."

echo ""
echo "[3/8] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate audioghost

echo ""
echo "[4/8] Installing PyTorch..."
echo "This may take several minutes..."

# Detect platform and install appropriate PyTorch
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use CPU/MPS build
    echo "Detected macOS - installing PyTorch with MPS support..."
    pip install torch torchvision torchaudio
else
    # Linux - install CUDA build
    echo "Detected Linux - installing PyTorch with CUDA 12.6..."
    pip install torch==2.9.0+cu126 torchvision==0.24.0+cu126 torchaudio==2.9.0+cu126 \
        --index-url https://download.pytorch.org/whl/cu126 \
        --extra-index-url https://pypi.org/simple
fi

echo ""
echo "[5/8] Installing FFmpeg..."
conda install -c conda-forge ffmpeg -y

echo ""
echo "[6/8] Installing SAM Audio..."
pip install git+https://github.com/facebookresearch/sam-audio.git

echo ""
echo "[7/8] Installing Backend dependencies..."
cd backend
pip install -r requirements.txt
cd ..

echo ""
echo "[8/8] Installing Frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║             Installation Complete! ✓                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║   To start AudioGhost, run:  ./start.sh                     ║"
echo "║                                                              ║"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "║   On macOS, Redis is managed by Homebrew.                   ║"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "║   On Linux, Redis is installed as a system service.         ║"
fi
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
