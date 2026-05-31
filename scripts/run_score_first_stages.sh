#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${MIDI_LLM_VENV:-"$ROOT/.venv-score-first"}
DATASET=${MIDI_LLM_DATASET:-"$ROOT/training_manifests/pdmx-intermediate/model_dataset"}
RUN_ROOT=${MIDI_LLM_RUN_ROOT:-"$ROOT/training_runs/score_first_intermediate_v2"}
MAX_SEQ_LENGTH=${MIDI_LLM_MAX_SEQ_LENGTH:-81920}

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Missing GPU environment: $VENV. Run scripts/bootstrap_score_first_gpu.sh first." >&2
  exit 2
fi

source "$VENV/bin/activate"
cd "$ROOT"

run_stage() {
  local name=$1
  local tasks=$2
  local epochs=$3
  local resume_adapter=${4:-}
  local args=(
    python -m midi_llm.train_scoredsl
    --dataset-dir "$DATASET"
    --output-dir "$RUN_ROOT/$name"
    --tasks "$tasks"
    --max-seq-length "$MAX_SEQ_LENGTH"
    --epochs "$epochs"
  )
  if [[ -n "$resume_adapter" ]]; then
    args+=(--resume-adapter-dir "$resume_adapter")
  fi
  "${args[@]}"
}

smoke() {
  python -m midi_llm.train_scoredsl \
    --dataset-dir "$DATASET" \
    --output-dir "$RUN_ROOT/00_smoke" \
    --tasks score-dsl-autoencode \
    --max-example-characters "${MIDI_LLM_SMOKE_MAX_EXAMPLE_CHARACTERS:-18000}" \
    --max-examples "${MIDI_LLM_SMOKE_MAX_EXAMPLES:-8}" \
    --max-seq-length "${MIDI_LLM_SMOKE_MAX_SEQ_LENGTH:-8192}" \
    --max-steps "${MIDI_LLM_SMOKE_MAX_STEPS:-1}" \
    --gradient-accumulation-steps 1
}

grammar() {
  run_stage \
    01_grammar \
    score-dsl-autoencode \
    "${MIDI_LLM_GRAMMAR_EPOCHS:-1}"
}

whole_piece() {
  run_stage \
    02_whole_piece \
    ending-complete,whole-piece-generate \
    "${MIDI_LLM_WHOLE_PIECE_EPOCHS:-1}" \
    "$RUN_ROOT/01_grammar/adapter"
}

structure() {
  run_stage \
    03_structure \
    section-expand-16-64,masked-span-inpaint,recapitulation-revise \
    "${MIDI_LLM_STRUCTURE_EPOCHS:-1}" \
    "$RUN_ROOT/02_whole_piece/adapter"
}

case "${1:-pilot}" in
  smoke)
    smoke
    ;;
  grammar)
    grammar
    ;;
  whole-piece)
    whole_piece
    ;;
  pilot)
    grammar
    whole_piece
    ;;
  structure)
    structure
    ;;
  all)
    smoke
    grammar
    whole_piece
    structure
    ;;
  *)
    echo "Usage: $0 {smoke|grammar|whole-piece|pilot|structure|all}" >&2
    exit 2
    ;;
esac
