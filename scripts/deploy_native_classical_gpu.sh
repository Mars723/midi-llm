#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 user@host [ssh-port]" >&2
  exit 2
fi

TARGET=$1
PORT=${2:-22}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUNDLE=${MIDI_LLM_NATIVE_BUNDLE:-"$ROOT/training_bundles/native-classical-pretraining-v1/native-classical-pretraining-v1.tar.gz"}
REMOTE_ROOT=${MIDI_LLM_REMOTE_ROOT:-midi-llm}
REMOTE_BUNDLE=${MIDI_LLM_REMOTE_BUNDLE:-native-classical-pretraining-v1.tar.gz}
REPO=${MIDI_LLM_REPO:-https://github.com/Mars723/midi-llm.git}
BRANCH=${MIDI_LLM_BRANCH:-codex/score-first-piano-v1}

if [[ ! -f "$BUNDLE" ]]; then
  echo "Missing native pretraining bundle: $BUNDLE" >&2
  echo "Run bash scripts/package_native_pretraining_bundle.sh first." >&2
  exit 2
fi

SSH_ARGS=(-p "$PORT" -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
SCP_ARGS=(-P "$PORT" -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
if [[ -n "${MIDI_LLM_SSH_KEY:-}" ]]; then
  SSH_ARGS+=(-i "$MIDI_LLM_SSH_KEY")
  SCP_ARGS+=(-i "$MIDI_LLM_SSH_KEY")
fi

ssh "${SSH_ARGS[@]}" "$TARGET" "
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

scp "${SCP_ARGS[@]}" "$BUNDLE" "$TARGET:$REMOTE_BUNDLE"

ssh "${SSH_ARGS[@]}" "$TARGET" "
  set -euo pipefail
  cd '$REMOTE_ROOT'
  bash scripts/bootstrap_native_classical_gpu.sh \"\$HOME/$REMOTE_BUNDLE\"
"

echo "Native remote bootstrap complete: $TARGET:$REMOTE_ROOT"
echo "Smoke command: ssh -p $PORT $TARGET 'cd $REMOTE_ROOT && bash scripts/run_native_classical_stages.sh smoke'"
