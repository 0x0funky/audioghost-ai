#!/bin/bash
set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              AudioGhost AI - Launcher                        ║"
echo "║                   v1.0 MVP                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Get the project root directory (parent of the script directory)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

# Function to wait for a service to be ready
wait_for_service() {
    local port=$1
    local service_name=$2
    local max_attempts=30
    local attempt=0

    echo "       Waiting for $service_name to be ready..."
    while ! nc -z localhost $port 2>/dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            echo "       [WARN] $service_name didn't start in time, continuing anyway..."
            return 1
        fi
        sleep 1
    done
    echo "       $service_name is ready! ✓"
}

# Check if Redis is running
echo "[1/4] Checking Redis..."
if nc -z localhost 6379 2>/dev/null; then
    echo "       Redis detected - using existing instance"
else
    echo "       Starting Redis..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use Homebrew
        brew services start redis 2>/dev/null || redis-server --daemonize yes
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux - try systemd or docker
        if command -v systemctl &> /dev/null; then
            sudo systemctl start redis-server 2>/dev/null || sudo systemctl start redis 2>/dev/null || docker-compose up -d
        else
            docker-compose up -d
        fi
    fi
    sleep 2
fi

echo "[2/4] Starting Backend API..."
# Start backend in background with proper conda activation
(cd "$PROJECT_ROOT/backend" && eval "$(conda shell.bash hook)" && conda activate audioghost && uvicorn main:app --reload --port 8000 > /tmp/audioghost-backend.log 2>&1) &
BACKEND_PID=$!
echo "       Backend starting (PID: $BACKEND_PID, logs: /tmp/audioghost-backend.log)"
wait_for_service 8000 "Backend API"

echo "[3/4] Starting Celery Worker..."
# Start celery in background with proper conda activation
(cd "$PROJECT_ROOT/backend" && eval "$(conda shell.bash hook)" && conda activate audioghost && celery -A workers.celery_app worker --loglevel=info > /tmp/audioghost-celery.log 2>&1) &
CELERY_PID=$!
echo "       Celery starting (PID: $CELERY_PID, logs: /tmp/audioghost-celery.log)"
sleep 3  # Give Celery time to connect to Redis

echo "[4/4] Starting Frontend..."
# Start frontend in background
(cd "$PROJECT_ROOT/frontend" && npm run dev > /tmp/audioghost-frontend.log 2>&1) &
FRONTEND_PID=$!
echo "       Frontend starting (PID: $FRONTEND_PID, logs: /tmp/audioghost-frontend.log)"
wait_for_service 3000 "Frontend"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              All Services Started! ✓                         ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║   Frontend:  http://localhost:3000                          ║"
echo "║   Backend:   http://localhost:8000                          ║"
echo "║   API Docs:  http://localhost:8000/docs                     ║"
echo "║                                                              ║"
echo "║   Services running in background:                           ║"
echo "║   - Redis:    PID $(lsof -ti :6379 2>/dev/null | head -1 || echo 'N/A')                                          ║"
echo "║   - Backend:  PID $BACKEND_PID                                          ║"
echo "║   - Celery:   PID $CELERY_PID                                          ║"
echo "║   - Frontend: PID $FRONTEND_PID                                          ║"
echo "║                                                              ║"
echo "║   View logs:                                                 ║"
echo "║   - Backend:  tail -f /tmp/audioghost-backend.log           ║"
echo "║   - Celery:   tail -f /tmp/audioghost-celery.log            ║"
echo "║   - Frontend: tail -f /tmp/audioghost-frontend.log          ║"
echo "║                                                              ║"
echo "║   Run ./stop.sh to stop all services.                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Opening browser in 3 seconds..."
sleep 3

# Open browser automatically
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:3000
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:3000 2>/dev/null || echo "Please open http://localhost:3000 in your browser"
fi

echo ""
echo "All services are running! Press Ctrl+C to return to shell."
echo "(Services will continue running in background)"
