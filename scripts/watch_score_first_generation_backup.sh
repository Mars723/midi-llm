#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 GENERATION_PID OUTPUT_ROOT" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GENERATION_PID=$1
OUTPUT_ROOT=$2
BACKUP_ROOT=${MIDI_LLM_BACKUP_ROOT:-"$ROOT/training_backups"}
RCLONE_REMOTE=${MIDI_LLM_RCLONE_REMOTE:-}
INTERVAL=${MIDI_LLM_GENERATION_BACKUP_INTERVAL_SECONDS:-120}

sync_progress() {
  local progress_root="$BACKUP_ROOT/generated_progress"
  local partial relative destination
  mkdir -p "$progress_root"
  while IFS= read -r -d '' partial; do
    relative=${partial#"$OUTPUT_ROOT/"}
    destination="$progress_root/$relative"
    mkdir -p "$(dirname "$destination")"
    rsync -a "$partial" "$destination"
  done < <(
    find "$OUTPUT_ROOT" -type f \
      \( -name '*.partial_after_*.score.ir.json' -o -name 'failures.json' \) \
      -print0 2>/dev/null
  )
  if [[ -n "$RCLONE_REMOTE" ]]; then
    command -v rclone >/dev/null
    rclone sync "$progress_root/" "$RCLONE_REMOTE/generated_progress/" \
      --create-empty-src-dirs
  fi
  echo "Generation progress backup complete: $progress_root"
}

while kill -0 "$GENERATION_PID" 2>/dev/null; do
  sync_progress || true
  sleep "$INTERVAL"
done

sync_progress
