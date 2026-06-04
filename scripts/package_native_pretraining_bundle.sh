#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATASET=${MIDI_LLM_NATIVE_DATASET:-"$ROOT/training_manifests/pdmx-native-classical-v3/native_tokens_training_view"}
BUNDLE_ROOT=${MIDI_LLM_NATIVE_BUNDLE_ROOT:-"$ROOT/training_bundles/native-classical-pretraining-v1"}
OUTPUT=${MIDI_LLM_NATIVE_BUNDLE_OUTPUT:-"$BUNDLE_ROOT/native-classical-pretraining-v1.tar.gz"}

cd "$ROOT"
mkdir -p "$BUNDLE_ROOT"
python3 -m midi_llm.package_native_pretraining package \
  --dataset-dir "$DATASET" \
  --output "$OUTPUT" \
  --mutopia-summary "$ROOT/training_manifests/mutopia-piano-v1/midi-profile-audit/summary.json" \
  --pianocore-summary "$ROOT/training_manifests/pianocore-metadata-v1/summary.json" \
  --asap-summary "$ROOT/training_manifests/asap-metadata-v1/summary.json" \
  --maestro-summary "$ROOT/training_manifests/maestro-performance-v3/summary.json" \
  --gallery-index "$ROOT/training_manifests/mutopia-piano-v1/intermediate-review-gallery/index.html"
python3 -m midi_llm.package_native_pretraining verify --bundle "$OUTPUT"
