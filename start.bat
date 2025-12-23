@echo off
chcp 65001 >nul
title AudioGhost AI - Launcher

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║              AudioGhost AI - Launcher                        ║
echo ║                   v1.0 MVP                                   ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

:: Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Check if Redis exists
if not exist "redis\redis-server.exe" (
    echo [ERROR] Redis not found. Please run install.bat first.
    pause
    exit /b 1
)

echo [1/4] Starting Redis...
start "AudioGhost Redis" /min cmd /c "cd /d %SCRIPT_DIR%redis && redis-server.exe"
timeout /t 2 /nobreak >nul

echo [2/4] Starting Backend API...
start "AudioGhost Backend" cmd /k "cd /d %SCRIPT_DIR% && conda activate audioghost && cd backend && uvicorn main:app --reload --port 8000"

echo [3/4] Starting Celery Worker...
timeout /t 2 /nobreak >nul
start "AudioGhost Worker" cmd /k "cd /d %SCRIPT_DIR% && conda activate audioghost && cd backend && celery -A workers.celery_app worker --loglevel=info --pool=solo"

echo [4/4] Starting Frontend...
timeout /t 2 /nobreak >nul
start "AudioGhost Frontend" cmd /k "cd /d %SCRIPT_DIR% && cd frontend && npm run dev"

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║              All Services Started! ✓                         ║
echo ╠══════════════════════════════════════════════════════════════╣
echo ║                                                              ║
echo ║   Frontend:  http://localhost:3000                          ║
echo ║   Backend:   http://localhost:8000                          ║
echo ║   API Docs:  http://localhost:8000/docs                     ║
echo ║                                                              ║
echo ║   Four windows opened:                                      ║
echo ║   - AudioGhost Redis (minimized)                            ║
echo ║   - AudioGhost Backend (FastAPI)                            ║
echo ║   - AudioGhost Worker (Celery)                              ║
echo ║   - AudioGhost Frontend (Next.js)                           ║
echo ║                                                              ║
echo ║   Close all windows to stop services.                       ║
echo ║                                                              ║
echo ║   Press any key to open browser...                          ║
echo ╚══════════════════════════════════════════════════════════════╝
pause >nul

:: Open browser
start http://localhost:3000
