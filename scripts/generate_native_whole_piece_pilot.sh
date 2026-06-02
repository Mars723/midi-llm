#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${MIDI_LLM_NATIVE_VENV:-/root/.venv-midllm-native}
ADAPTER=${1:?Usage: $0 /path/to/native-adapter [output-dir]}
OUTPUT=${2:-/root/midllm-local/generated_native_backbone/native_classical_v1_whole_piece_001}
PROMPT=${MIDI_LLM_WHOLE_PIECE_PROMPT:-A complete lyrical intermediate solo piano nocturne in D minor, in the style of Chopin, 4/4 Andante, with a singing right-hand melody, broken-chord left-hand accompaniment, an introspective opening, a tense contrasting middle section, a recognizable calm return, and a quiet sparse ending.}
BLUEPRINT=${MIDI_LLM_WHOLE_PIECE_BLUEPRINT:-ABA nocturne with shared thematic context, explicit opening-theme reuse in A prime, phrase-aware transitions, and a short coda that avoids a terminal block chord while realizing the future cadence target.}

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

A_MIDI=$(python - <<PY
from midi_llm.native_whole_piece import select_native_material
print(select_native_material("$OUTPUT/material_a", ${MIDI_LLM_WHOLE_PIECE_A_SECONDS:-48}))
PY
)

python -m midi_llm.native_backbone \
  --prompt "$PROMPT Shared full-piece blueprint: $BLUEPRINT Continue from the supplied A-section tail into contrasting tense B-section development material only." \
  --adapter "$ADAPTER" \
  --output-dir "$OUTPUT/material_b" \
  --n-outputs "${MIDI_LLM_WHOLE_PIECE_MATERIAL_OUTPUTS:-2}" \
  --seed "${MIDI_LLM_WHOLE_PIECE_B_SEED:-41}" \
  --max-tokens "${MIDI_LLM_WHOLE_PIECE_MATERIAL_TOKENS:-1200}" \
  --prefix-midi "$A_MIDI" \
  --prefix-tail-seconds "${MIDI_LLM_WHOLE_PIECE_PREFIX_SECONDS:-18}" \
  --skip-score-draft

python -m midi_llm.native_backbone \
  --prompt "$PROMPT Shared full-piece blueprint: $BLUEPRINT Continue from the supplied return-theme tail into a quiet sparse Coda. Avoid a final block chord." \
  --adapter "$ADAPTER" \
  --output-dir "$OUTPUT/material_coda" \
  --n-outputs "${MIDI_LLM_WHOLE_PIECE_MATERIAL_OUTPUTS:-2}" \
  --seed "${MIDI_LLM_WHOLE_PIECE_CODA_SEED:-51}" \
  --max-tokens "${MIDI_LLM_WHOLE_PIECE_CODA_TOKENS:-360}" \
  --prefix-midi "$A_MIDI" \
  --prefix-tail-seconds "${MIDI_LLM_WHOLE_PIECE_PREFIX_SECONDS:-18}" \
  --skip-score-draft

python -m midi_llm.native_whole_piece \
  --a-run "$OUTPUT/material_a" \
  --b-run "$OUTPUT/material_b" \
  --coda-run "$OUTPUT/material_coda" \
  --output-dir "$OUTPUT/piece" \
  --prompt "$PROMPT" \
  --blueprint "$BLUEPRINT" \
  --title "Native Nocturne Study No. 1" \
  --tempo "${MIDI_LLM_WHOLE_PIECE_TEMPO:-72}" \
  --a-seconds "${MIDI_LLM_WHOLE_PIECE_A_SECONDS:-48}" \
  --b-seconds "${MIDI_LLM_WHOLE_PIECE_B_SECONDS:-55}" \
  --coda-seconds "${MIDI_LLM_WHOLE_PIECE_CODA_SECONDS:-10}"

echo "Native whole-piece sample: $OUTPUT/piece"
