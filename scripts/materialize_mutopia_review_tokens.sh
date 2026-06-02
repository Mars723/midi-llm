#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MANIFEST_ROOT=${MIDI_LLM_MUTOPIA_MANIFEST_ROOT:-"$ROOT/training_manifests/mutopia-piano-v1"}
COMPILE_ROOT=${MIDI_LLM_MUTOPIA_COMPILE_ROOT:-"$MANIFEST_ROOT/compile-audit"}
PROFILE_ROOT=${MIDI_LLM_MUTOPIA_PROFILE_ROOT:-"$MANIFEST_ROOT/midi-profile-audit"}
TOKEN_ROOT=${MIDI_LLM_MUTOPIA_TOKEN_ROOT:-"$MANIFEST_ROOT/native-token-review-view"}
MAX_EVENT_TOKENS=${MIDI_LLM_NATIVE_MAX_EVENT_TOKENS:-7800}

cd "$ROOT"
if [[ -f "$ROOT/.venv-native-preprocess/bin/activate" ]]; then
  source "$ROOT/.venv-native-preprocess/bin/activate"
fi
python3 -m midi_llm.materialize_native_training \
  --manifest "$PROFILE_ROOT/bounded_piece_review_candidates.jsonl" \
  --dataset-root "$COMPILE_ROOT" \
  --output-dir "$TOKEN_ROOT" \
  --max-event-tokens "$MAX_EVENT_TOKENS"
