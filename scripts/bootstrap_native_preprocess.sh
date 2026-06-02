#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${MIDI_LLM_NATIVE_PREPROCESS_VENV:-"$ROOT/.venv-native-preprocess"}

cd "$ROOT"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  'anticipation @ git+https://github.com/jthickstun/anticipation.git@af37397922665a0fb8d474d7988b0f3755a38d45' \
  'mido>=1.3,<2'

python - <<'PY'
from anticipation.convert import midi_to_events
import mido
print("Native CPU preprocessing environment is ready.")
print("anticipation.convert.midi_to_events:", midi_to_events)
print("mido:", mido)
PY
