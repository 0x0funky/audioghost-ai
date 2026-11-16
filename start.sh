#!/usr/bin/env bash
# AudioGhost AI - start Redis, API, worker and web UI (macOS / Linux).
# Windows: use start.bat. Stop everything with ./stop.sh
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUN_DIR="$ROOT/.audioghost"
LOG_DIR="$ROOT/logs"
ENV_NAME="audioghost"
mkdir -p "$RUN_DIR" "$LOG_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              AudioGhost AI - Launcher (v1.1)                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

wait_for_port() {  # port name seconds
    local i
    for ((i = 0; i < $3; i++)); do
        if port_open "$1"; then echo "      $2 is up"; return 0; fi
        sleep 1
    done
    echo "      [WARN] $2 is not answering on port $1 yet - see $LOG_DIR"
}

# Refuse to start twice
if [[ -f "$RUN_DIR/pids" ]]; then
    while read -r name pid; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "AudioGhost is already running ($name, PID $pid). Run ./stop.sh first."
            exit 1
        fi
    done < "$RUN_DIR/pids"
    rm -f "$RUN_DIR/pids"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found - run ./install.sh first."
    exit 1
fi
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ---------------------------------------------------------------- Redis ----
echo "[1/4] Redis..."
if port_open 6379; then
    echo "      Using the Redis already listening on port 6379"
elif command -v redis-server >/dev/null 2>&1; then
    redis-server --port 6379 --bind 127.0.0.1 --save "" --appendonly no \
        --daemonize yes --pidfile "$RUN_DIR/redis.pid" --logfile "$LOG_DIR/redis.log"
    wait_for_port 6379 Redis 10
elif command -v docker >/dev/null 2>&1; then
    (docker compose up -d || docker-compose up -d) && touch "$RUN_DIR/redis.docker"
    wait_for_port 6379 Redis 30
else
    echo "[ERROR] Redis is not installed. Run ./install.sh, or start it with: docker compose up -d"
    exit 1
fi

# ------------------------------------------------------------- Services ----
# Job control gives every service its own process group, so stop.sh can end
# each one together with its children (npm -> next, etc.) and nothing else.
set -m
start_service() {  # name dir command...
    local name="$1" dir="$2"
    shift 2
    (cd "$dir" && exec "$@") > "$LOG_DIR/$name.log" 2>&1 &
    echo "$name $!" >> "$RUN_DIR/pids"
}

echo "[2/4] Starting Backend API..."
start_service api "$ROOT/backend" python -m uvicorn main:app --host 127.0.0.1 --port 8000
wait_for_port 8000 "Backend API" 60

echo "[3/4] Starting Worker..."
start_service worker "$ROOT/backend" python -m celery -A workers.celery_app worker --loglevel=info --pool=solo

echo "[4/4] Starting Frontend..."
start_service frontend "$ROOT/frontend" npm run dev
# Next.js moves to 3001, 3002, ... if 3000 is taken - wait for the URL it reports.
URL=""
for ((i = 0; i < 90; i++)); do
    URL="$(grep -Eo 'http://localhost:[0-9]+' "$LOG_DIR/frontend.log" 2>/dev/null | head -1 || true)"
    if [[ -n "$URL" ]]; then break; fi
    sleep 1
done
if [[ -n "$URL" ]]; then
    echo "      Frontend is up"
else
    URL="http://localhost:3000"
    echo "      [WARN] The frontend hasn't reported its URL yet - see $LOG_DIR/frontend.log"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              All Services Started! ✓                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "   Web UI:   $URL"
echo "   API docs: http://localhost:8000/docs"
echo "   Logs:     $LOG_DIR  (tail -f logs/worker.log)"
echo "   Stop:     ./stop.sh"
echo ""

if [[ "$(uname -s)" == "Darwin" ]]; then
    open "$URL" || true
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 || true
fi
