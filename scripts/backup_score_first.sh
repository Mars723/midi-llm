#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=${MIDI_LLM_RUN_ROOT:-"$ROOT/training_runs/score_first_intermediate_v2"}
BACKUP_ROOT=${MIDI_LLM_BACKUP_ROOT:-"$ROOT/training_backups"}
RECOVERY_ROOT=${MIDI_LLM_RECOVERY_ROOT:-}
RCLONE_REMOTE=${MIDI_LLM_RCLONE_REMOTE:-}
STAGE=${MIDI_LLM_BACKUP_STAGE:-09_two_staff_refinement}
MIRROR_RUN_ROOT=${MIDI_LLM_BACKUP_MIRROR_RUN_ROOT:-1}
CREATE_ADAPTER_ARCHIVE=${MIDI_LLM_CREATE_ADAPTER_ARCHIVE:-1}
RUN_NAME=$(basename "$RUN_ROOT")
LOCK_DIR="$BACKUP_ROOT/.backup-lock"

mkdir -p "$BACKUP_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Backup already in progress: $LOCK_DIR"
  exit 0
fi
trap 'rmdir "$LOCK_DIR"' EXIT

mkdir -p "$BACKUP_ROOT/metadata"
if [[ "$MIRROR_RUN_ROOT" != 0 ]]; then
  mkdir -p "$BACKUP_ROOT/training_runs/$RUN_NAME"
  rsync -a --delete "$RUN_ROOT/" "$BACKUP_ROOT/training_runs/$RUN_NAME/"
fi

if [[ -d "$BACKUP_ROOT/logs" ]]; then
  mkdir -p "$BACKUP_ROOT/log-snapshots/latest"
  rsync -a --delete "$BACKUP_ROOT/logs/" "$BACKUP_ROOT/log-snapshots/latest/"
fi

if [[ -n "$RECOVERY_ROOT" && -d "$RECOVERY_ROOT" ]]; then
  mkdir -p "$BACKUP_ROOT/recovery"
  rsync -a --exclude 'rclone.conf' "$RECOVERY_ROOT/" "$BACKUP_ROOT/recovery/"
fi

{
  printf 'backup_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'source_root=%s\n' "$ROOT"
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'git_commit=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
} > "$BACKUP_ROOT/metadata/latest.txt"

if [[ "$CREATE_ADAPTER_ARCHIVE" != 0 && -d "$RUN_ROOT/$STAGE/adapter" ]]; then
  mkdir -p "$BACKUP_ROOT/artifacts"
  archive="$BACKUP_ROOT/artifacts/${STAGE}-adapter.tar.gz"
  tmp_archive="$archive.tmp"
  tar -czf "$tmp_archive" -C "$RUN_ROOT/$STAGE" adapter
  mv "$tmp_archive" "$archive"
  sha256sum "$archive" > "$archive.sha256"
fi

if [[ -n "$RCLONE_REMOTE" ]]; then
  command -v rclone >/dev/null
  rclone sync "$RUN_ROOT/" "$RCLONE_REMOTE/training_runs/$RUN_NAME/" \
    --create-empty-src-dirs
  rclone sync "$BACKUP_ROOT/" "$RCLONE_REMOTE/" \
    --create-empty-src-dirs \
    --exclude '/logs/**' \
    --exclude '/training_runs/**' \
    --exclude '/.backup-lock/**'
fi

echo "Backup complete: $BACKUP_ROOT"
