#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 TRAINING_PID RUN_DIR [STAGE]" >&2
  exit 2
fi

TRAINING_PID=$1
RUN_DIR=$2
STAGE=${3:-$(basename "$RUN_DIR")}
LOCAL_ROOT=${MIDI_LLM_NATIVE_LOCAL_BACKUP_ROOT:-/root/midllm-local/native-classical-backups}
PERSIST_ROOT=${MIDI_LLM_NATIVE_BACKUP_ROOT:-/workspace/midllm-backups/native-classical-v1}
RCLONE_REMOTE=${MIDI_LLM_RCLONE_REMOTE:-}
RCLONE=${MIDI_LLM_RCLONE_BIN:-rclone}
RCLONE_TIMEOUT=${MIDI_LLM_NATIVE_RCLONE_TIMEOUT:-600}
RCLONE_LOG_TIMEOUT=${MIDI_LLM_NATIVE_RCLONE_LOG_TIMEOUT:-60}
INTERVAL=${MIDI_LLM_NATIVE_BACKUP_INTERVAL:-15}
LOG=${MIDI_LLM_NATIVE_LOG:-}

if [[ ! "$STAGE" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Stage must contain only letters, numbers, dots, underscores, and hyphens: $STAGE" >&2
  exit 2
fi

LOCAL_STAGE="$LOCAL_ROOT/$STAGE"
PERSIST_STAGE="$PERSIST_ROOT/$STAGE"
mkdir -p "$LOCAL_STAGE" "$PERSIST_STAGE"

copy_stable_file() {
  local source=$1
  local destination="$PERSIST_STAGE/$(basename "$source")"
  local persist_marker="$source.persisted"
  local remote_marker="$source.remote-uploaded"

  if [[ ! -f "$persist_marker" || ! -s "$destination" ]]; then
    cp "$source" "$destination.tmp"
    mv "$destination.tmp" "$destination"
    touch "$persist_marker"
  fi

  if [[ -n "$RCLONE_REMOTE" && ! -f "$remote_marker" ]]; then
    if timeout --foreground "$RCLONE_TIMEOUT" \
      "$RCLONE" copyto "$source" "$RCLONE_REMOTE/$STAGE/$(basename "$source")"; then
      touch "$remote_marker"
    else
      echo "Drive upload failed; will retry: $source" >&2
      return 1
    fi
  fi
}

archive_dir() {
  local source=$1
  local name=$2
  local archive="$LOCAL_STAGE/$name.tar.gz"

  if [[ ! -f "$archive" ]]; then
    rm -f "$archive.tmp"
    tar -czf "$archive.tmp" -C "$(dirname "$source")" "$(basename "$source")"
    mv "$archive.tmp" "$archive"
    (
      cd "$LOCAL_STAGE"
      sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256"
    )
  fi

  copy_stable_file "$archive" || true
  copy_stable_file "$archive.sha256" || true
}

snapshot_checkpoints() {
  local checkpoint
  for checkpoint in "$RUN_DIR"/checkpoint-*; do
    [[ -d "$checkpoint" && -f "$checkpoint/trainer_state.json" ]] || continue
    archive_dir "$checkpoint" "$STAGE-$(basename "$checkpoint")"
  done
}

snapshot_log() {
  local snapshot="$LOCAL_STAGE/$STAGE.log"
  [[ -n "$LOG" && -f "$LOG" ]] || return 0
  cp "$LOG" "$snapshot.tmp"
  mv "$snapshot.tmp" "$snapshot"
  cp "$snapshot" "$PERSIST_STAGE/$(basename "$snapshot").tmp"
  mv "$PERSIST_STAGE/$(basename "$snapshot").tmp" "$PERSIST_STAGE/$(basename "$snapshot")"
  if [[ -n "$RCLONE_REMOTE" ]]; then
    timeout --foreground "$RCLONE_LOG_TIMEOUT" \
      "$RCLONE" copyto "$snapshot" "$RCLONE_REMOTE/$STAGE/$(basename "$snapshot")" || \
      echo "Drive log upload failed; will retry: $snapshot" >&2
  fi
}

snapshot_metadata() {
  local source
  for source in \
    "$RUN_DIR/training_result.json" \
    "$RUN_DIR/training_spec.json" \
    "$RUN_DIR/tokenizer_preflight.json"; do
    [[ -f "$source" ]] || continue
    cp "$source" "$LOCAL_STAGE/$(basename "$source").tmp"
    mv "$LOCAL_STAGE/$(basename "$source").tmp" "$LOCAL_STAGE/$(basename "$source")"
    copy_stable_file "$LOCAL_STAGE/$(basename "$source")" || true
  done
}

while kill -0 "$TRAINING_PID" 2>/dev/null; do
  snapshot_checkpoints
  snapshot_log
  sleep "$INTERVAL"
done

snapshot_checkpoints
snapshot_log
snapshot_metadata
if [[ -d "$RUN_DIR/adapter" ]]; then
  archive_dir "$RUN_DIR/adapter" "$STAGE-final-adapter"
fi

echo "Native classical backup complete: $PERSIST_STAGE"
