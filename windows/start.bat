@echo off
chcp 65001 >nul
title AudioGhost AI - Launcher

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║              AudioGhost AI - Launcher                        ║
echo ║                   v1.0 MVP                                   ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

:: Get the project root directory (parent of the script directory)
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

:: Check if Docker Redis is already running on port 6379
echo [1/4] Checking Redis...
netstat -an | findstr ":6379.*LISTENING" >nul 2>&1
if %ERRORLEVEL%==0 (
    echo       Docker Redis detected - using existing instance
    goto :redis_ready
)

:: Docker Redis not running, try offline Redis
if exist "redis\redis-server.exe" (
    echo       Starting offline Redis...
    start "AudioGhost Redis" /min cmd /c "cd /d %PROJECT_ROOT%\redis && redis-server.exe"
    timeout /t 2 /nobreak >nul
    goto :redis_ready
)

:: Neither available
echo [ERROR] Redis not available. Either:
echo         - Start Docker: docker-compose up -d
echo         - Or run install.bat to download offline Redis
pause
exit /b 1

:redis_ready

echo [2/4] Starting Backend API...
start "AudioGhost Backend" cmd /k "cd /d %PROJECT_ROOT% && conda activate audioghost && cd backend && uvicorn main:app --reload --port 8000"

echo [3/4] Starting Celery Worker...
timeout /t 2 /nobreak >nul
start "AudioGhost Worker" cmd /k "cd /d %PROJECT_ROOT% && conda activate audioghost && cd backend && celery -A workers.celery_app worker --loglevel=info --pool=solo"

echo [4/4] Starting Frontend...
timeout /t 2 /nobreak >nul
start "AudioGhost Frontend" cmd /k "cd /d %PROJECT_ROOT% && cd frontend && npm run dev"

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
echo ║   Run stop.bat or close all windows to stop services.       ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Opening browser in 3 seconds...
timeout /t 3 /nobreak >nul

:: Open browser automatically
start http://localhost:3000
