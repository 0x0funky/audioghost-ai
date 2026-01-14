@echo off
chcp 65001 >nul
title AudioGhost AI - Stop All Services

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║              AudioGhost AI - Shutdown                        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo Stopping all AudioGhost services...
echo.

:: Kill Node.js (Frontend)
echo [1/4] Stopping Frontend...
taskkill /FI "WINDOWTITLE eq AudioGhost Frontend*" /F >nul 2>&1
taskkill /IM "node.exe" /F >nul 2>&1

:: Kill Celery (Worker)
echo [2/4] Stopping Celery Worker...
taskkill /FI "WINDOWTITLE eq AudioGhost Worker*" /F >nul 2>&1

:: Kill Uvicorn (Backend)
echo [3/4] Stopping Backend API...
taskkill /FI "WINDOWTITLE eq AudioGhost Backend*" /F >nul 2>&1

:: Stop Redis
echo [4/4] Stopping Redis...
docker-compose down >nul 2>&1

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║               All Services Stopped! ✓                        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
pause
