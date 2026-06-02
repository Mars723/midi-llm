#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_ROOT=${MIDI_LLM_MUTOPIA_ROOT:-"$ROOT/training_resources/mutopia/source"}
MANIFEST_ROOT=${MIDI_LLM_MUTOPIA_MANIFEST_ROOT:-"$ROOT/training_manifests/mutopia-piano-v1"}
TOOLCHAIN_ROOT=${MIDI_LLM_TOOLCHAIN_ROOT:-"$ROOT/training_resources/toolchains"}
LILYPOND_VERSION=${MIDI_LLM_LILYPOND_VERSION:-2.26.0}
LILYPOND=${MIDI_LLM_LILYPOND:-"$TOOLCHAIN_ROOT/lilypond-$LILYPOND_VERSION-darwin-x86_64/bin/lilypond"}
COMPILE_ROOT=${MIDI_LLM_MUTOPIA_COMPILE_ROOT:-"$MANIFEST_ROOT/compile-audit"}
LIMIT=${MIDI_LLM_MUTOPIA_COMPILE_LIMIT:-}

cd "$ROOT"
if [[ ! -x "$LILYPOND" ]]; then
  bash scripts/bootstrap_lilypond_runtime.sh
fi

ARGS=(
  --manifest "$MANIFEST_ROOT/mutopia_piano_manifest.jsonl"
  --source-root "$SOURCE_ROOT"
  --output-dir "$COMPILE_ROOT"
  --lilypond "$LILYPOND"
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
python3 -m midi_llm.compile_mutopia "${ARGS[@]}"
