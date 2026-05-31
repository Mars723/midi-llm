#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 user@host [ssh-port]" >&2
  exit 2
fi

TARGET=$1
PORT=${2:-22}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUNDLE=${MIDI_LLM_BUNDLE:-"$ROOT/training_bundles/score-first-model-dataset.tar.gz"}
REMOTE_ROOT=${MIDI_LLM_REMOTE_ROOT:-midi-llm}
REMOTE_BUNDLE=${MIDI_LLM_REMOTE_BUNDLE:-score-first-model-dataset.tar.gz}
REPO=${MIDI_LLM_REPO:-https://github.com/Mars723/midi-llm.git}
BRANCH=${MIDI_LLM_BRANCH:-codex/score-first-piano-v1}

if [[ ! -f "$BUNDLE" ]]; then
  echo "Missing local cloud bundle: $BUNDLE" >&2
  echo "Run python -m midi_llm.package_cloud package first." >&2
  exit 2
fi

ssh -p "$PORT" "$TARGET" "
  set -euo pipefail
  if [[ -d '$REMOTE_ROOT/.git' ]]; then
    cd '$REMOTE_ROOT'
    git fetch origin '$BRANCH'
    git checkout '$BRANCH'
    git pull --ff-only origin '$BRANCH'
  else
    git clone --branch '$BRANCH' '$REPO' '$REMOTE_ROOT'
  fi
"

scp -P "$PORT" "$BUNDLE" "$TARGET:$REMOTE_BUNDLE"

ssh -p "$PORT" "$TARGET" "
  set -euo pipefail
  cd '$REMOTE_ROOT'
  bash scripts/bootstrap_score_first_gpu.sh \"\$HOME/$REMOTE_BUNDLE\"
"

echo "Remote bootstrap complete: $TARGET:$REMOTE_ROOT"
echo "Run smoke: ssh -p $PORT $TARGET 'cd $REMOTE_ROOT && bash scripts/run_score_first_stages.sh smoke'"
