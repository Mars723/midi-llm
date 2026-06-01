#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 TRAINING_PID" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAINING_PID=$1
BACKUP_ROOT=${MIDI_LLM_BACKUP_ROOT:-"$ROOT/training_backups"}
POST_LOG_ROOT=${MIDI_LLM_POST_LOG_ROOT:-"$BACKUP_ROOT/logs"}
OUTPUT_ROOT=${MIDI_LLM_POST_OUTPUT_ROOT:-"$ROOT/generated_score_first"}
SEEDS=${MIDI_LLM_POST_SAMPLE_SEEDS:-"90 96 102 108"}

mkdir -p "$BACKUP_ROOT/generated_score_first" "$POST_LOG_ROOT" "$OUTPUT_ROOT"

while kill -0 "$TRAINING_PID" 2>/dev/null; do
  sleep 60
done

bash "$ROOT/scripts/backup_score_first.sh"

selected_output=
for seed in $SEEDS; do
  output="$OUTPUT_ROOT/checkpoint_sample_v4_two_staff_seed${seed}"
  log="$POST_LOG_ROOT/checkpoint_sample_v4_two_staff_seed${seed}.log"
  if MIDI_LLM_SAMPLE_OUTPUT="$output" MIDI_LLM_SAMPLE_SEED="$seed" \
    bash "$ROOT/scripts/generate_score_first_pilot_sample.sh" > "$log" 2>&1; then
    selected_output=$output
    break
  fi
done

if [[ -z "$selected_output" ]]; then
  echo "No post-training sample passed generation validation." >&2
  exit 1
fi

rsync -a "$selected_output/" "$BACKUP_ROOT/generated_score_first/$(basename "$selected_output")/"
cp "$selected_output.tar.gz" "$BACKUP_ROOT/generated_score_first/"
bash "$ROOT/scripts/backup_score_first.sh"

echo "Post-training sample: $selected_output"
