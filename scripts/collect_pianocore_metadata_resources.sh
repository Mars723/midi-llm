#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESOURCE_ROOT=${MIDI_LLM_PIANOCORE_ROOT:-"$ROOT/training_resources/pianocore-metadata"}
MANIFEST_ROOT=${MIDI_LLM_PIANOCORE_MANIFEST_ROOT:-"$ROOT/training_manifests/pianocore-metadata-v1"}

cd "$ROOT"
mkdir -p "$RESOURCE_ROOT" "$MANIFEST_ROOT"
python3 - "$RESOURCE_ROOT" <<'PY'
from pathlib import Path
import sys
from midi_llm.fetch_pdmx import _download, _md5

root = Path(sys.argv[1])
resources = {
    "composers.csv": (14718, "1399b8af8ffa633a4f476ed1decb6626"),
    "metadata.csv": (205846408, "873f55178fc7343bcadcc1ef97305b93"),
}
for name, (size, expected_md5) in resources.items():
    destination = root / name
    _download(f"https://zenodo.org/records/19186016/files/{name}?download=1", destination, size, connections=8)
    checksum = _md5(destination)
    if checksum != expected_md5:
        raise ValueError(f"Checksum mismatch for {destination}: {checksum}")
    print({"resource": name, "bytes": size, "md5": checksum})
PY
python3 -m midi_llm.prepare_pianocore \
  --metadata-csv "$RESOURCE_ROOT/metadata.csv" \
  --composers-csv "$RESOURCE_ROOT/composers.csv" \
  --output-dir "$MANIFEST_ROOT"
