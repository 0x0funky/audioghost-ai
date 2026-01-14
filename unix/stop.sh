#!/bin/bash

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              AudioGhost AI - Shutdown                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

echo "Stopping all AudioGhost services..."
echo ""

# Kill Frontend (Node.js/npm)
echo "[1/4] Stopping Frontend..."
pkill -f "npm run dev" 2>/dev/null || true
pkill -f "next-router-worker" 2>/dev/null || true

# Kill Celery Worker
echo "[2/4] Stopping Celery Worker..."
pkill -f "celery.*audioghost" 2>/dev/null || true

# Kill Backend (Uvicorn)
echo "[3/4] Stopping Backend API..."
pkill -f "uvicorn main:app" 2>/dev/null || true

# Stop Redis (Docker)
echo "[4/4] Stopping Redis..."
if command -v docker-compose &> /dev/null; then
    docker-compose down 2>/dev/null || true
fi

# Stop Redis (Homebrew on Mac)
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew services stop redis 2>/dev/null || true
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║               All Services Stopped! ✓                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

