#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${MIDI_LLM_VENV:-"$ROOT/.venv-score-first"}
RUN_ROOT=${MIDI_LLM_RUN_ROOT:-"$ROOT/training_runs/score_first_intermediate_v2"}
OUTPUT=${MIDI_LLM_SAMPLE_OUTPUT:-"$ROOT/generated_score_first/checkpoint_sample_001"}
ADAPTER_DIR=${MIDI_LLM_SAMPLE_ADAPTER_DIR:-"$RUN_ROOT/08_balance_refinement/adapter"}

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Missing GPU environment: $VENV. Run scripts/bootstrap_score_first_gpu.sh first." >&2
  exit 2
fi

source "$VENV/bin/activate"
cd "$ROOT"

python -m midi_llm.generate_checkpoint \
  --adapter-dir "$ADAPTER_DIR" \
  --prompt "A lyrical intermediate nocturne with a tense middle section and a calm return." \
  --genre nocturne \
  --form ABA \
  --difficulty intermediate \
  --duration-minutes 3 \
  --candidates "${MIDI_LLM_SAMPLE_CANDIDATES:-2}" \
  --section-candidates "${MIDI_LLM_SAMPLE_SECTION_CANDIDATES:-2}" \
  --temperature "${MIDI_LLM_SAMPLE_TEMPERATURE:-0.8}" \
  --repetition-penalty "${MIDI_LLM_SAMPLE_REPETITION_PENALTY:-1.01}" \
  --skip-musescore \
  --output-dir "$OUTPUT"

tar -czf "$OUTPUT.tar.gz" -C "$(dirname "$OUTPUT")" "$(basename "$OUTPUT")"
echo "Checkpoint-backed sample archive: $OUTPUT.tar.gz"
