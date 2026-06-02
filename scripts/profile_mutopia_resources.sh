#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MANIFEST_ROOT=${MIDI_LLM_MUTOPIA_MANIFEST_ROOT:-"$ROOT/training_manifests/mutopia-piano-v1"}
COMPILE_ROOT=${MIDI_LLM_MUTOPIA_COMPILE_ROOT:-"$MANIFEST_ROOT/compile-audit"}
PROFILE_ROOT=${MIDI_LLM_MUTOPIA_PROFILE_ROOT:-"$MANIFEST_ROOT/midi-profile-audit"}

cd "$ROOT"
python3 -m midi_llm.profile_mutopia_midi \
  --manifest "$COMPILE_ROOT/clean_solo_piano_review_candidates.jsonl" \
  --compile-root "$COMPILE_ROOT" \
  --output-dir "$PROFILE_ROOT"
