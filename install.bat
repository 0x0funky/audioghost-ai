@echo off
chcp 65001 >nul
title AudioGhost AI - One-Click Installer

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║           AudioGhost AI - One-Click Installer               ║
echo ║                        v1.1                                  ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

:: Pinned versions (tested together)
set "SAM_AUDIO_COMMIT=bb4c6999d2677c7402360e426afc01ddfad6dce0"
set "TORCH_VERSION=2.9.0"
set "TORCHVISION_VERSION=0.24.0"

:: Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Check if Conda is installed
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda not found. Please install Anaconda or Miniconda first.
    echo Download from: https://www.anaconda.com/download
    pause
    exit /b 1
)

:: Pick the PyTorch CUDA build: RTX 50-series (compute capability 12.x) needs CUDA 12.8
set "TORCH_CUDA=cu126"
for /f "tokens=1 delims=." %%a in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do (
    if %%a GEQ 12 set "TORCH_CUDA=cu128"
)
where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] nvidia-smi not found - no NVIDIA GPU/driver detected. AudioGhost will run on CPU ^(very slow^).
)
echo Using PyTorch %TORCH_VERSION% with %TORCH_CUDA%

echo.
echo [1/8] Downloading Redis for Windows...
if not exist "redis\redis-server.exe" (
    echo Downloading from GitHub...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/tporadowski/redis/releases/download/v5.0.14.1/Redis-x64-5.0.14.1.zip' -OutFile 'redis.zip'"
    echo Extracting...
    powershell -Command "Expand-Archive -Path 'redis.zip' -DestinationPath 'redis' -Force"
    del redis.zip
    echo Redis installed to ./redis/
) else (
    echo Redis already exists, skipping...
)

echo.
echo [2/8] Creating Conda environment 'audioghost' (Python 3.11)...
call conda create -n audioghost python=3.11 -y
if %errorlevel% neq 0 (
    echo [WARN] Environment may already exist, continuing...
)

echo.
echo [3/8] Activating environment...
call conda activate audioghost
if %errorlevel% neq 0 (
    echo [ERROR] Could not activate the 'audioghost' environment.
    pause
    exit /b 1
)

echo.
echo [4/8] Installing PyTorch %TORCH_VERSION% (%TORCH_CUDA%)...
echo This may take several minutes...
pip install torch==%TORCH_VERSION%+%TORCH_CUDA% torchvision==%TORCHVISION_VERSION%+%TORCH_CUDA% torchaudio==%TORCH_VERSION%+%TORCH_CUDA% --index-url https://download.pytorch.org/whl/%TORCH_CUDA% --extra-index-url https://pypi.org/simple

echo.
echo [5/8] Installing FFmpeg 7 (TorchCodec does not support every FFmpeg release)...
call conda install -c conda-forge "ffmpeg=7" -y

echo.
echo [6/8] Installing SAM Audio (pinned commit %SAM_AUDIO_COMMIT:~0,7%)...
pip install "git+https://github.com/facebookresearch/sam-audio.git@%SAM_AUDIO_COMMIT%"

echo.
echo [7/8] Installing Backend dependencies (also pins transformers / huggingface_hub)...
cd backend
pip install -r requirements.txt
cd ..

echo.
echo [8/8] Installing Frontend dependencies...
cd frontend
call npm ci
if %errorlevel% neq 0 call npm install
cd ..

echo.
echo Verifying the installation...
python -c "import torch, torchcodec, sam_audio; print('  PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available()); print('  TorchCodec + FFmpeg: OK'); print('  SAM-Audio: OK')"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The Python environment is not working yet. Common fixes:
    echo   - "Could not load libtorchcodec": FFmpeg DLLs not found. Re-run this installer, or install
    echo     the FFmpeg 7 "shared" build from https://www.gyan.dev/ffmpeg/builds/ and add its bin folder to PATH.
    echo   - "entry point ... libintl / gdk_pixbuf": another program's DLLs are on PATH ^(GTK, Git mingw64^).
    echo     Remove them from PATH for this window and re-run install.bat.
    echo   See README.md - Troubleshooting.
    pause
    exit /b 1
)

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║             Installation Complete! ✓                        ║
echo ╠══════════════════════════════════════════════════════════════╣
echo ║                                                              ║
echo ║   To start AudioGhost, run:  start.bat                      ║
echo ║                                                              ║
echo ║   No Docker required! Redis is included.                    ║
echo ║   The first separation downloads the model (5-15 GB).       ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
pause
