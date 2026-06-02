#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TOOLCHAIN_ROOT=${MIDI_LLM_TOOLCHAIN_ROOT:-"$ROOT/training_resources/toolchains"}
LILYPOND_VERSION=${MIDI_LLM_LILYPOND_VERSION:-2.26.0}
ARCHIVE="$TOOLCHAIN_ROOT/lilypond-$LILYPOND_VERSION-darwin-x86_64.tar.gz"
RUNTIME="$TOOLCHAIN_ROOT/lilypond-$LILYPOND_VERSION-darwin-x86_64"
URL="https://gitlab.com/lilypond/lilypond/-/releases/v$LILYPOND_VERSION/downloads/lilypond-$LILYPOND_VERSION-darwin-x86_64.tar.gz"

case "$LILYPOND_VERSION" in
  2.26.0)
    SHA256=6dcbca34b13ad6d4ba3606a0b48edd02688284fcd52e9c00141242df1996a148
    ;;
  *)
    echo "No pinned SHA-256 for LilyPond $LILYPOND_VERSION" >&2
    exit 2
    ;;
esac

mkdir -p "$TOOLCHAIN_ROOT"
if [[ ! -f "$ARCHIVE" ]]; then
  curl -L --fail --retry 8 --retry-all-errors --connect-timeout 20 --output "$ARCHIVE" "$URL"
fi
printf '%s  %s\n' "$SHA256" "$ARCHIVE" | shasum -a 256 -c -
if [[ ! -x "$RUNTIME/bin/lilypond" ]]; then
  mkdir -p "$RUNTIME"
  tar -xzf "$ARCHIVE" -C "$RUNTIME" --strip-components=1
fi
"$RUNTIME/bin/lilypond" --version | sed -n '1,4p'
