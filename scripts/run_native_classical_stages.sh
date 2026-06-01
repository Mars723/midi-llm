#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${MIDI_LLM_NATIVE_VENV:-/root/.venv-midllm-native}
DATASET=${MIDI_LLM_NATIVE_DATASET:-/root/midllm-local/classical-native-v1/training_manifests/pdmx-classical-core/native_tokens_v2}
RUN_ROOT=${MIDI_LLM_NATIVE_RUN_ROOT:-/root/midllm-local/training_runs/native_classical_v1}

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Missing native GPU environment: $VENV" >&2
  exit 2
fi

source "$VENV/bin/activate"
cd "$ROOT"

dry_run() {
  python -m midi_llm.train_native \
    --dataset-dir "$DATASET" \
    --output-dir "$RUN_ROOT/00_dry_run" \
    --dry-run
}

smoke() {
  python -m midi_llm.train_native \
    --dataset-dir "$DATASET" \
    --output-dir "$RUN_ROOT/01_smoke" \
    --max-segments "${MIDI_LLM_NATIVE_SMOKE_SEGMENTS:-8}" \
    --max-segments-per-work 1 \
    --max-seq-length "${MIDI_LLM_NATIVE_MAX_SEQ_LENGTH:-8192}" \
    --max-steps "${MIDI_LLM_NATIVE_SMOKE_STEPS:-1}" \
    --gradient-accumulation-steps 1 \
    --save-steps 1
}

pilot() {
  python -m midi_llm.train_native \
    --dataset-dir "$DATASET" \
    --output-dir "$RUN_ROOT/02_pilot" \
    --max-segments-per-work "${MIDI_LLM_NATIVE_MAX_SEGMENTS_PER_WORK:-4}" \
    --max-seq-length "${MIDI_LLM_NATIVE_MAX_SEQ_LENGTH:-8192}" \
    --max-steps "${MIDI_LLM_NATIVE_PILOT_STEPS:-100}" \
    --learning-rate "${MIDI_LLM_NATIVE_LEARNING_RATE:-0.00002}" \
    --save-steps "${MIDI_LLM_NATIVE_SAVE_STEPS:-25}"
}

case "${1:-dry-run}" in
  dry-run)
    dry_run
    ;;
  smoke)
    smoke
    ;;
  pilot)
    pilot
    ;;
  *)
    echo "Usage: $0 {dry-run|smoke|pilot}" >&2
    exit 2
    ;;
esac
