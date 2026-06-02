#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${MIDI_LLM_NATIVE_VENV:-/root/.venv-midllm-native}
ADAPTER=${1:?Usage: $0 /path/to/native-adapter [output-dir]}
OUTPUT=${2:-/root/midllm-local/generated_native_backbone/native_classical_v1_whole_piece_001}
PROMPT=${MIDI_LLM_WHOLE_PIECE_PROMPT:-A complete lyrical intermediate solo piano nocturne with an introspective opening, a tense contrasting middle section, a recognizable calm return, and a clear quiet ending.}
BLUEPRINT=${MIDI_LLM_WHOLE_PIECE_BLUEPRINT:-ABA nocturne with explicit opening-theme reuse in A prime and a short coda that realizes the future authentic-cadence target.}

source "$VENV/bin/activate"
cd "$ROOT"
mkdir -p "$OUTPUT"

python -m midi_llm.native_backbone \
  --prompt "$PROMPT Shared full-piece blueprint: $BLUEPRINT Compose opening A-section material only." \
  --adapter "$ADAPTER" \
  --output-dir "$OUTPUT/material_a" \
  --n-outputs "${MIDI_LLM_WHOLE_PIECE_MATERIAL_OUTPUTS:-2}" \
  --seed "${MIDI_LLM_WHOLE_PIECE_A_SEED:-31}" \
  --max-tokens "${MIDI_LLM_WHOLE_PIECE_MATERIAL_TOKENS:-1200}" \
  --skip-score-draft

python -m midi_llm.native_backbone \
  --prompt "$PROMPT Shared full-piece blueprint: $BLUEPRINT Compose contrasting tense B-section development material only." \
  --adapter "$ADAPTER" \
  --output-dir "$OUTPUT/material_b" \
  --n-outputs "${MIDI_LLM_WHOLE_PIECE_MATERIAL_OUTPUTS:-2}" \
  --seed "${MIDI_LLM_WHOLE_PIECE_B_SEED:-41}" \
  --max-tokens "${MIDI_LLM_WHOLE_PIECE_MATERIAL_TOKENS:-1200}" \
  --skip-score-draft

python -m midi_llm.native_whole_piece \
  --a-run "$OUTPUT/material_a" \
  --b-run "$OUTPUT/material_b" \
  --output-dir "$OUTPUT/piece" \
  --prompt "$PROMPT" \
  --blueprint "$BLUEPRINT" \
  --title "Native Nocturne Study No. 1" \
  --tempo "${MIDI_LLM_WHOLE_PIECE_TEMPO:-72}" \
  --a-seconds "${MIDI_LLM_WHOLE_PIECE_A_SECONDS:-38}" \
  --b-seconds "${MIDI_LLM_WHOLE_PIECE_B_SECONDS:-42}" \
  --coda-seconds "${MIDI_LLM_WHOLE_PIECE_CODA_SECONDS:-8}"

echo "Native whole-piece sample: $OUTPUT/piece"
