#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESOURCE_ROOT=${MIDI_LLM_MAESTRO_RESOURCE_ROOT:-"$ROOT/training_resources/maestro"}
MANIFEST_ROOT=${MIDI_LLM_MAESTRO_MANIFEST_ROOT:-"$ROOT/training_manifests/maestro-performance-v3"}
BASE_URL=https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0
ARCHIVE=maestro-v3.0.0-midi.zip
CSV=maestro-v3.0.0.csv
JSON=maestro-v3.0.0.json
EXPECTED_SHA256=70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c

mkdir -p "$RESOURCE_ROOT" "$MANIFEST_ROOT"
cd "$RESOURCE_ROOT"

curl --fail --location --retry 8 --retry-all-errors --continue-at - --output "$ARCHIVE" "$BASE_URL/$ARCHIVE"
curl --fail --location --retry 8 --retry-all-errors --output "$CSV" "$BASE_URL/$CSV"
curl --fail --location --retry 8 --retry-all-errors --output "$JSON" "$BASE_URL/$JSON"

ACTUAL_SHA256=$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "MAESTRO checksum mismatch: expected $EXPECTED_SHA256, got $ACTUAL_SHA256" >&2
  exit 1
fi

unzip -q -o "$ARCHIVE"

cd "$ROOT"
python3 -m midi_llm.prepare_maestro \
  --metadata-csv "$RESOURCE_ROOT/$CSV" \
  --output-dir "$MANIFEST_ROOT"
