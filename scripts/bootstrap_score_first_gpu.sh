#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/score-first-model-dataset.tar.gz" >&2
  exit 2
fi

BUNDLE=$1
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${MIDI_LLM_VENV:-"$ROOT/.venv-score-first"}
DATA_ROOT=${MIDI_LLM_DATA_ROOT:-"$ROOT/training_manifests/pdmx-intermediate"}

cd "$ROOT"
command -v nvidia-smi >/dev/null
nvidia-smi

python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  --index-url "${MIDI_LLM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}" \
  "${MIDI_LLM_TORCH_PACKAGE:-torch==2.8.0}"
python -m pip install -r requirements-score-first-train.txt

rm -rf "$DATA_ROOT/model_dataset"
python -m midi_llm.package_cloud extract \
  --bundle "$BUNDLE" \
  --output-dir "$DATA_ROOT"

python -m midi_llm.train_scoredsl \
  --dataset-dir "$DATA_ROOT/model_dataset" \
  --output-dir training_runs/score_first_intermediate_v1/preflight \
  --dry-run

echo "GPU bootstrap complete."
echo "Run: scripts/run_score_first_stages.sh smoke"
echo "Then: scripts/run_score_first_stages.sh pilot"
