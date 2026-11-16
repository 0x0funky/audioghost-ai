#!/usr/bin/env bash
# AudioGhost AI - stop what ./start.sh started (and nothing else).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$ROOT/.audioghost"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              AudioGhost AI - Shutdown                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Only signal a recorded PID if it still runs the program we started there
# (PIDs can be reused after a reboot).
is_ours() {  # name pid
    local cmd
    cmd="$(ps -o command= -p "$2" 2>/dev/null)" || return 1
    case "$1" in
        api) [[ "$cmd" == *uvicorn* ]] ;;
        worker) [[ "$cmd" == *celery* ]] ;;
        frontend) [[ "$cmd" == *npm* || "$cmd" == *next* || "$cmd" == *node* ]] ;;
        *) return 1 ;;
    esac
}

stop_group() {  # signal name pid
    kill "-$1" -- "-$3" 2>/dev/null || kill "-$1" "$3" 2>/dev/null || true
}

if [[ -f "$RUN_DIR/pids" ]]; then
    while read -r name pid; do
        if is_ours "$name" "$pid"; then
            echo "Stopping $name (PID $pid)..."
            stop_group TERM "$name" "$pid"
        fi
    done < "$RUN_DIR/pids"
    sleep 3
    while read -r name pid; do
        if is_ours "$name" "$pid"; then
            stop_group KILL "$name" "$pid"
        fi
    done < "$RUN_DIR/pids"
    rm -f "$RUN_DIR/pids"
else
    echo "Nothing recorded in $RUN_DIR - was AudioGhost started with ./start.sh?"
fi

# Redis: only the instance start.sh launched itself
if [[ -f "$RUN_DIR/redis.pid" ]]; then
    echo "Stopping Redis..."
    kill "$(cat "$RUN_DIR/redis.pid")" 2>/dev/null || true
    rm -f "$RUN_DIR/redis.pid"
fi
if [[ -f "$RUN_DIR/redis.docker" ]]; then
    echo "Stopping Redis (Docker)..."
    (cd "$ROOT" && (docker compose down || docker-compose down)) >/dev/null 2>&1 || true
    rm -f "$RUN_DIR/redis.docker"
fi

echo ""
echo "All AudioGhost services stopped."
echo ""
