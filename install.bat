@echo off
chcp 65001 >nul
title AudioGhost AI - One-Click Installer

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║           AudioGhost AI - One-Click Installer               ║
echo ║                   v1.0 MVP                                   ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

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

echo.
echo [4/8] Installing PyTorch (CUDA 12.6)...
echo This may take several minutes...
pip install torch==2.9.0.dev20250322+cu126 torchaudio==2.6.0.dev20250322+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126

echo.
echo [5/8] Installing FFmpeg...
call conda install -c conda-forge ffmpeg -y

echo.
echo [6/8] Installing SAM Audio...
pip install git+https://github.com/facebookresearch/sam-audio.git

echo.
echo [7/8] Installing Backend dependencies...
cd backend
pip install -r requirements.txt
cd ..

echo.
echo [8/8] Installing Frontend dependencies...
cd frontend
call npm install
cd ..

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║             Installation Complete! ✓                        ║
echo ╠══════════════════════════════════════════════════════════════╣
echo ║                                                              ║
echo ║   To start AudioGhost, run:  start.bat                      ║
echo ║                                                              ║
echo ║   No Docker required! Redis is included.                    ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
pause
