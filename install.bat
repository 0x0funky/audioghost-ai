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
:: FFmpeg 8.1 shared build (DLLs for TorchCodec + ffmpeg.exe), linked from ffmpeg.org
set "FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-lgpl-shared-8.1.zip"

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
    powershell -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://github.com/tporadowski/redis/releases/download/v5.0.14.1/Redis-x64-5.0.14.1.zip' -OutFile 'redis.zip'"
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
echo [5/8] Installing FFmpeg 8 (shared build for TorchCodec)...
:: A self-contained FFmpeg build instead of conda-forge's: its Windows package
:: currently ships two different intl-8.dll files (libglib and libintl), which
:: breaks gdk-pixbuf (the libintl_bind_textdomain_codeset popup of issue #10)
:: and stops TorchCodec from loading FFmpeg at all.
if exist "ffmpeg\bin\avcodec-62.dll" (
    echo FFmpeg already installed, skipping...
) else (
    echo Downloading FFmpeg 8.1 ^(about 80 MB^)...
    powershell -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile 'ffmpeg.zip'"
    powershell -Command "Expand-Archive -Path 'ffmpeg.zip' -DestinationPath 'ffmpeg_tmp' -Force"
    for /d %%D in (ffmpeg_tmp\*) do move "%%D" ffmpeg >nul
    if exist ffmpeg_tmp rmdir /s /q ffmpeg_tmp
    if exist ffmpeg.zip del ffmpeg.zip
)
if not exist "ffmpeg\bin\avcodec-62.dll" (
    echo [WARN] FFmpeg download failed. Install an FFmpeg 8 "shared" build yourself and add its bin folder to PATH:
    echo        https://github.com/BtbN/FFmpeg-Builds/releases or https://www.gyan.dev/ffmpeg/builds/
)
set "PATH=%SCRIPT_DIR%ffmpeg\bin;%PATH%"
echo.
echo [6/8] Installing SAM Audio (pinned commit %SAM_AUDIO_COMMIT:~0,7%)...
:: constraints.txt keeps torch / torchcodec / xformers on the tested versions
pip install -c backend\constraints.txt "git+https://github.com/facebookresearch/sam-audio.git@%SAM_AUDIO_COMMIT%"

echo.
echo [7/8] Installing Backend dependencies (also pins transformers / huggingface_hub)...
cd backend
pip install -c constraints.txt -r requirements.txt
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
    echo   - "Could not load libtorchcodec": FFmpeg DLLs not found. Re-run this installer, or install an
    echo     FFmpeg 8 "shared" build ^(BtbN or gyan.dev^) and add its bin folder to PATH.
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
