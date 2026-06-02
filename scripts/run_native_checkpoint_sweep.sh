#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${MIDI_LLM_NATIVE_VENV:-/root/.venv-midllm-native}
RUN_DIR=${1:?Usage: $0 /path/to/native-training-run [replay-root]}
REPLAY_ROOT=${2:-/root/midllm-local/generated_native_backbone/$(basename "$RUN_DIR")_checkpoint_sweep}
BASELINE=${MIDI_LLM_NATIVE_BASELINE_RUN:-/root/midllm-local/generated_native_backbone/upstream_parity_nocturne_v1}

source "$VENV/bin/activate"
cd "$ROOT"
mkdir -p "$REPLAY_ROOT"

for checkpoint in "$RUN_DIR"/checkpoint-*; do
  [[ -d "$checkpoint" && -f "$checkpoint/trainer_state.json" ]] || continue
  name=$(basename "$checkpoint")
  output="$REPLAY_ROOT/$name"
  if [[ ! -f "$output/manifest.json" ]]; then
    MIDI_LLM_PARITY_OUTPUTS="${MIDI_LLM_SWEEP_OUTPUTS:-4}" \
    MIDI_LLM_PARITY_SEED="${MIDI_LLM_SWEEP_SEED:-23}" \
    MIDI_LLM_PARITY_MAX_TOKENS="${MIDI_LLM_SWEEP_MAX_TOKENS:-2046}" \
      bash scripts/run_native_adapter_parity.sh "$checkpoint" "$output"
  fi
done

python -m midi_llm.native_checkpoint_selection \
  --baseline-run "$BASELINE" \
  --replay-root "$REPLAY_ROOT" \
  --checkpoint-run-root "$RUN_DIR" \
  --output "$REPLAY_ROOT/checkpoint-selection.json"
