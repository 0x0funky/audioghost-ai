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

:: Only the windows opened by start.bat are closed (with their child processes).
:: Other Node.js / Python / Redis processes on this machine are left alone.

echo [1/4] Stopping Frontend...
taskkill /T /F /FI "WINDOWTITLE eq AudioGhost Frontend*" >nul 2>&1

echo [2/4] Stopping Celery Worker...
taskkill /T /F /FI "WINDOWTITLE eq AudioGhost Worker*" >nul 2>&1

echo [3/4] Stopping Backend API...
taskkill /T /F /FI "WINDOWTITLE eq AudioGhost Backend*" >nul 2>&1

echo [4/4] Stopping Redis...
taskkill /T /F /FI "WINDOWTITLE eq AudioGhost Redis*" >nul 2>&1
docker-compose down >nul 2>&1

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║               All Services Stopped! ✓                        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
pause
