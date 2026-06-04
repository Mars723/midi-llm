#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/native-classical-pretraining-v1.tar.gz" >&2
  exit 2
fi

BUNDLE=$1
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${MIDI_LLM_NATIVE_VENV:-/root/.venv-midllm-native}
EXTRACT_ROOT=${MIDI_LLM_NATIVE_EXTRACT_ROOT:-/root/midllm-local/classical-native-v1}
DATASET=${MIDI_LLM_NATIVE_DATASET:-"$EXTRACT_ROOT/training_dataset"}
RUN_ROOT=${MIDI_LLM_NATIVE_RUN_ROOT:-/root/midllm-local/training_runs/native_classical_v1}
TORCH_INDEX=${MIDI_LLM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
TORCH_PACKAGE=${MIDI_LLM_TORCH_PACKAGE:-torch==2.8.0}

cd "$ROOT"
if [[ ! -f "$BUNDLE" ]]; then
  echo "Missing native pretraining bundle: $BUNDLE" >&2
  exit 2
fi

command -v nvidia-smi >/dev/null
nvidia-smi

python3 -m midi_llm.package_native_pretraining verify --bundle "$BUNDLE"

python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install --index-url "$TORCH_INDEX" "$TORCH_PACKAGE"
python -m pip install -r requirements-score-first-train.txt

case "$EXTRACT_ROOT" in
  /|/root|/workspace|"$ROOT")
    echo "Refusing to replace unsafe extract root: $EXTRACT_ROOT" >&2
    exit 2
    ;;
esac
rm -rf "$EXTRACT_ROOT"
mkdir -p "$EXTRACT_ROOT" "$RUN_ROOT"
tar -xzf "$BUNDLE" -C "$EXTRACT_ROOT"
if [[ ! -d "$DATASET" ]]; then
  echo "Bundle extracted, but training dataset was not found: $DATASET" >&2
  exit 2
fi

MIDI_LLM_NATIVE_VENV="$VENV" \
MIDI_LLM_NATIVE_DATASET="$DATASET" \
MIDI_LLM_NATIVE_RUN_ROOT="$RUN_ROOT" \
  bash scripts/run_native_classical_stages.sh dry-run

echo "Native classical GPU bootstrap complete."
echo "No optimizer step has been run."
echo "Run smoke: MIDI_LLM_NATIVE_DATASET=$DATASET bash scripts/run_native_classical_stages.sh smoke"
echo "Run pilot: MIDI_LLM_NATIVE_DATASET=$DATASET bash scripts/run_native_classical_stages.sh conservative-pilot"
