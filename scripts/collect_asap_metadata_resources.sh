#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESOURCE_ROOT=${MIDI_LLM_ASAP_ROOT:-"$ROOT/training_resources/asap"}
SOURCE_ROOT=${MIDI_LLM_ASAP_SOURCE_ROOT:-"$RESOURCE_ROOT/source"}
MANIFEST_ROOT=${MIDI_LLM_ASAP_MANIFEST_ROOT:-"$ROOT/training_manifests/asap-metadata-v1"}

cd "$ROOT"
mkdir -p "$RESOURCE_ROOT" "$MANIFEST_ROOT"
if [[ ! -d "$SOURCE_ROOT/.git" ]]; then
  git clone --depth 1 --filter=blob:none --no-checkout \
    https://github.com/fosfrancesco/asap-dataset.git \
    "$SOURCE_ROOT"
fi
COMMIT=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
git -C "$SOURCE_ROOT" show HEAD:metadata.csv > "$RESOURCE_ROOT/metadata.csv"
git -C "$SOURCE_ROOT" show HEAD:LICENSE.md > "$RESOURCE_ROOT/LICENSE.md"
git -C "$SOURCE_ROOT" ls-tree -r --name-only HEAD > "$RESOURCE_ROOT/repository-tree.txt"
python3 -m midi_llm.prepare_asap \
  --metadata-csv "$RESOURCE_ROOT/metadata.csv" \
  --output-dir "$MANIFEST_ROOT" \
  --repository-commit "$COMMIT"
