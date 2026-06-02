#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_ROOT=${MIDI_LLM_MUTOPIA_SOURCE_ROOT:-"$ROOT/training_resources/mutopia/source"}
OUTPUT_DIR=${MIDI_LLM_MUTOPIA_MANIFEST_ROOT:-"$ROOT/training_manifests/mutopia-piano-v1"}
REPOSITORY=${MIDI_LLM_MUTOPIA_REPOSITORY:-https://github.com/MutopiaProject/MutopiaProject.git}

cd "$ROOT"
mkdir -p "$(dirname "$SOURCE_ROOT")" "$OUTPUT_DIR"

if [[ -d "$SOURCE_ROOT/.git" ]]; then
  git -C "$SOURCE_ROOT" pull --ff-only
else
  git clone --depth 1 --filter=blob:none --single-branch --branch master "$REPOSITORY" "$SOURCE_ROOT"
fi

python3 -m midi_llm.collect_mutopia \
  --source-root "$SOURCE_ROOT" \
  --output-dir "$OUTPUT_DIR"
