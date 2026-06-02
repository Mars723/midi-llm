#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_ROOT=${MIDI_LLM_MUTOPIA_ROOT:-"$ROOT/training_resources/mutopia/source"}
MANIFEST_ROOT=${MIDI_LLM_MUTOPIA_MANIFEST_ROOT:-"$ROOT/training_manifests/mutopia-piano-v1"}
COMPILE_ROOT=${MIDI_LLM_MUTOPIA_COMPILE_ROOT:-"$MANIFEST_ROOT/compile-audit"}
PROFILE_ROOT=${MIDI_LLM_MUTOPIA_PROFILE_ROOT:-"$MANIFEST_ROOT/midi-profile-audit"}
GALLERY_ROOT=${MIDI_LLM_MUTOPIA_GALLERY_ROOT:-"$MANIFEST_ROOT/intermediate-review-gallery"}
TOOLCHAIN_ROOT=${MIDI_LLM_TOOLCHAIN_ROOT:-"$ROOT/training_resources/toolchains"}
LILYPOND_VERSION=${MIDI_LLM_LILYPOND_VERSION:-2.26.0}
LILYPOND=${MIDI_LLM_LILYPOND:-"$TOOLCHAIN_ROOT/lilypond-$LILYPOND_VERSION-darwin-x86_64/bin/lilypond"}

cd "$ROOT"
if [[ ! -x "$LILYPOND" ]]; then
  bash scripts/bootstrap_lilypond_runtime.sh
fi
python3 -m midi_llm.render_mutopia_review \
  --manifest "$PROFILE_ROOT/intermediate_proxy_review_candidates.jsonl" \
  --source-root "$SOURCE_ROOT" \
  --compile-root "$COMPILE_ROOT" \
  --output-dir "$GALLERY_ROOT" \
  --lilypond "$LILYPOND"
