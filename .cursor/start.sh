#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8080}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v ss >/dev/null 2>&1; then
  if ss -ltn | grep -q ":${PORT} "; then
    echo "Static server already listening on port ${PORT}"
    exit 0
  fi
fi

echo "Starting static file server on 0.0.0.0:${PORT} from ${ROOT}"
exec python3 -m http.server "$PORT" --bind 0.0.0.0
