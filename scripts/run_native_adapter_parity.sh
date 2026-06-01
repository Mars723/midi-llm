#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${MIDI_LLM_NATIVE_VENV:-/root/.venv-midllm-native}
ADAPTER=${1:?Usage: $0 /path/to/native-adapter [output-dir]}
OUTPUT=${2:-/root/midllm-local/generated_native_backbone/native_adapter_parity}
PROMPT=${MIDI_LLM_PARITY_PROMPT:-A lyrical intermediate solo piano nocturne in the style of Chopin, from the romantic period, with a tense middle section and a calm return.}

source "$VENV/bin/activate"
cd "$ROOT"

python -m midi_llm.native_backbone \
  --prompt "$PROMPT" \
  --adapter "$ADAPTER" \
  --output-dir "$OUTPUT" \
  --n-outputs "${MIDI_LLM_PARITY_OUTPUTS:-2}" \
  --seed "${MIDI_LLM_PARITY_SEED:-23}" \
  --max-tokens "${MIDI_LLM_PARITY_MAX_TOKENS:-2046}" \
  --skip-score-draft
