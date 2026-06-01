#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${MIDI_LLM_VENV:-"$ROOT/.venv-score-first"}
OUTPUT_ROOT=${MIDI_LLM_NATIVE_OUTPUT_ROOT:-"$ROOT/generated_native_backbone"}
RUN_NAME=${MIDI_LLM_NATIVE_RUN_NAME:-"upstream_parity_$(date +%Y-%m-%d_%H%M%S)"}
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
PROMPT=${MIDI_LLM_NATIVE_PROMPT:-"A lyrical intermediate solo piano nocturne with a tense middle section and a calm return."}

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Missing GPU environment: $VENV" >&2
  exit 2
fi

source "$VENV/bin/activate"
cd "$ROOT"

python -m midi_llm.native_backbone \
  --prompt "$PROMPT" \
  --output-dir "$OUTPUT_DIR" \
  --n-outputs "${MIDI_LLM_NATIVE_OUTPUTS:-4}" \
  --seed "${MIDI_LLM_NATIVE_SEED:-23}" \
  --max-tokens "${MIDI_LLM_NATIVE_MAX_TOKENS:-2046}"

ARCHIVE="$OUTPUT_ROOT/$RUN_NAME.tar.gz"
tar -czf "$ARCHIVE" -C "$OUTPUT_ROOT" "$RUN_NAME"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"

if [[ -n "${MIDI_LLM_RCLONE_REMOTE:-}" ]]; then
  rclone copyto "$ARCHIVE" "$MIDI_LLM_RCLONE_REMOTE/generated_native_backbone/$RUN_NAME.tar.gz"
  rclone copyto "$ARCHIVE.sha256" "$MIDI_LLM_RCLONE_REMOTE/generated_native_backbone/$RUN_NAME.tar.gz.sha256"
fi

printf 'Native backbone pilot: %s\n' "$OUTPUT_DIR"
printf 'Archive: %s\n' "$ARCHIVE"
