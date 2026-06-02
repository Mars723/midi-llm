#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PDMX_MANIFEST_ROOT=${MIDI_LLM_NATIVE_MANIFEST_ROOT:-"$ROOT/training_manifests/pdmx-native-classical-v3"}
MUTOPIA_MANIFEST_ROOT=${MIDI_LLM_MUTOPIA_MANIFEST_ROOT:-"$ROOT/training_manifests/mutopia-piano-v1"}
OUTPUT_ROOT=${MIDI_LLM_CLASSICAL_INVENTORY_ROOT:-"$ROOT/training_manifests/classical-resource-inventory-v1"}

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"

python3 -m midi_llm.merge_manifests \
  --manifest "$PDMX_MANIFEST_ROOT/catalog-all/pdmx_score_first_manifest.jsonl" \
  --manifest "$MUTOPIA_MANIFEST_ROOT/mutopia_piano_manifest.jsonl" \
  --output "$OUTPUT_ROOT/collected_sources.jsonl"

python3 -m midi_llm.dataset_audit \
  --manifest "$OUTPUT_ROOT/collected_sources.jsonl" \
  --output-dir "$OUTPUT_ROOT/audit-collected-sources" \
  --min-composer-works 10
