#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT/.server.pid"
LOG_FILE="$ROOT/.server.log"
PORT=18846

live_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(<"$PID_FILE")"
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -Fq "$ROOT/server.py"
}

case "${1:-status}" in
  start)
    if live_pid; then echo "already running: $(<"$PID_FILE")"; exit 0; fi
    rm -f "$PID_FILE"
    nohup python3 "$ROOT/server.py" --port "$PORT" >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    echo "started: $(<"$PID_FILE")"
    ;;
  stop)
    if live_pid; then kill "$(<"$PID_FILE")"; fi
    rm -f "$PID_FILE"
    echo "stopped"
    ;;
  status)
    if live_pid; then echo "running: $(<"$PID_FILE")"; else echo "stopped"; fi
    ;;
  *) echo "usage: $0 {start|stop|status}" >&2; exit 2 ;;
esac
