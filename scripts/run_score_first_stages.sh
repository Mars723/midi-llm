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
  if [[ $# -ge 4 ]]; then
    shift 4
  else
    shift "$#"
  fi
  local args=(
    python -m midi_llm.train_scoredsl
    --dataset-dir "$DATASET"
    --output-dir "$RUN_ROOT/$name"
    --tasks "$tasks"
    --max-seq-length "$MAX_SEQ_LENGTH"
    --epochs "$epochs"
    --save-steps "${MIDI_LLM_SAVE_STEPS:-100}"
  )
  if [[ -n "$resume_adapter" ]]; then
    args+=(--resume-adapter-dir "$resume_adapter")
  fi
  if [[ -n "${MIDI_LLM_RESUME_CHECKPOINT_DIR:-}" ]]; then
    args+=(--resume-checkpoint-dir "$MIDI_LLM_RESUME_CHECKPOINT_DIR")
  fi
  if [[ $# -gt 0 ]]; then
    args+=("$@")
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

whole_piece_focus() {
  run_stage \
    04_whole_piece_focus \
    whole-piece-generate \
    "${MIDI_LLM_WHOLE_PIECE_FOCUS_EPOCHS:-3}" \
    "$RUN_ROOT/03_structure/adapter"
}

v3_grammar() {
  run_stage \
    05_v3_grammar \
    score-dsl-autoencode \
    "${MIDI_LLM_V3_GRAMMAR_EPOCHS:-1}" \
    "$RUN_ROOT/04_whole_piece_focus/adapter"
}

diversity_structure() {
  run_stage \
    06_diversity_structure \
    section-expand-16-64,masked-span-inpaint,recapitulation-revise,section-variation-revise \
    "${MIDI_LLM_DIVERSITY_STRUCTURE_EPOCHS:-1}" \
    "$RUN_ROOT/05_v3_grammar/adapter" \
    --min-variation-score "${MIDI_LLM_MIN_VARIATION_SCORE:-0.55}"
}

diversity_whole_piece() {
  run_stage \
    07_diversity_whole_piece \
    ending-complete,whole-piece-generate,section-variation-revise \
    "${MIDI_LLM_DIVERSITY_WHOLE_PIECE_EPOCHS:-2}" \
    "$RUN_ROOT/06_diversity_structure/adapter" \
    --min-variation-score "${MIDI_LLM_MIN_VARIATION_SCORE:-0.55}" \
    --gradient-accumulation-steps "${MIDI_LLM_DIVERSITY_GRADIENT_ACCUMULATION_STEPS:-4}" \
    --learning-rate "${MIDI_LLM_DIVERSITY_LEARNING_RATE:-0.0001}"
}

balance_refinement() {
  run_stage \
    08_balance_refinement \
    score-dsl-autoencode,section-expand-16-64,masked-span-inpaint,recapitulation-revise,ending-complete,whole-piece-generate,section-variation-revise \
    "${MIDI_LLM_BALANCE_REFINEMENT_EPOCHS:-1}" \
    "$RUN_ROOT/07_diversity_whole_piece/adapter" \
    --min-variation-score "${MIDI_LLM_MIN_VARIATION_SCORE:-0.55}" \
    --learning-rate "${MIDI_LLM_BALANCE_REFINEMENT_LEARNING_RATE:-0.00005}"
}

two_staff_refinement() {
  run_stage \
    09_two_staff_refinement \
    score-dsl-autoencode,section-expand-16-64,masked-span-inpaint,recapitulation-revise,ending-complete,whole-piece-generate,section-variation-revise \
    "${MIDI_LLM_TWO_STAFF_REFINEMENT_EPOCHS:-1}" \
    "$RUN_ROOT/08_balance_refinement/adapter" \
    --min-variation-score "${MIDI_LLM_MIN_VARIATION_SCORE:-0.55}" \
    --min-lower-staff-measure-coverage "${MIDI_LLM_MIN_LOWER_STAFF_MEASURE_COVERAGE:-0.75}" \
    --learning-rate "${MIDI_LLM_TWO_STAFF_REFINEMENT_LEARNING_RATE:-0.00005}"
}

variation_repair_refinement() {
  run_stage \
    10_variation_repair_refinement \
    section-expand-16-64,masked-span-inpaint,recapitulation-revise,section-variation-revise \
    "${MIDI_LLM_VARIATION_REPAIR_REFINEMENT_EPOCHS:-1}" \
    "$RUN_ROOT/09_two_staff_refinement/adapter" \
    --min-variation-score "${MIDI_LLM_VARIATION_REPAIR_MIN_VARIATION_SCORE:-0.70}" \
    --min-lower-staff-measure-coverage "${MIDI_LLM_MIN_LOWER_STAFF_MEASURE_COVERAGE:-0.75}" \
    --gradient-accumulation-steps "${MIDI_LLM_VARIATION_REPAIR_GRADIENT_ACCUMULATION_STEPS:-8}" \
    --learning-rate "${MIDI_LLM_VARIATION_REPAIR_LEARNING_RATE:-0.00003}"
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
  whole-piece-focus)
    whole_piece_focus
    ;;
  diversity)
    v3_grammar
    diversity_structure
    diversity_whole_piece
    balance_refinement
    two_staff_refinement
    ;;
  balance-refinement)
    balance_refinement
    ;;
  two-staff-refinement)
    two_staff_refinement
    ;;
  variation-repair-refinement)
    variation_repair_refinement
    ;;
  all)
    smoke
    grammar
    whole_piece
    structure
    whole_piece_focus
    v3_grammar
    diversity_structure
    diversity_whole_piece
    balance_refinement
    two_staff_refinement
    variation_repair_refinement
    ;;
  *)
    echo "Usage: $0 {smoke|grammar|whole-piece|pilot|structure|whole-piece-focus|diversity|balance-refinement|two-staff-refinement|variation-repair-refinement|all}" >&2
    exit 2
    ;;
esac
