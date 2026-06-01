#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 TRAINING_PID" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAINING_PID=$1
INTERVAL=${MIDI_LLM_BACKUP_INTERVAL_SECONDS:-300}

while kill -0 "$TRAINING_PID" 2>/dev/null; do
  bash "$ROOT/scripts/backup_score_first.sh" || true
  sleep "$INTERVAL"
done

bash "$ROOT/scripts/backup_score_first.sh"
