#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${ROBOARM_VOICE_PYTHON:-/Users/dhruvauppaluri/Documents/Codex/2026-10-10/cou/work/sim-env/bin/python}"
SOCKET=/tmp/roboarm-mac-ssh
HOST="${AMD_SSH_TARGET:-amd-genai2}"
if ! ssh -S "$SOCKET" -O check "$HOST" >/dev/null 2>&1; then
  ssh -M -S "$SOCKET" -o ControlPersist=8h -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -fN "$HOST"
fi
set -a
source "$HOME/.config/roboarm/mac-voice.env"
set +a
exec "$PYTHON" "$SCRIPT_DIR/mac_voice.py" --socket "$SOCKET" --host "$HOST" "$@"
