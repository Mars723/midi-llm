#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESOURCE_ROOT=${MIDI_LLM_CLASSICAL_RESOURCE_ROOT:-"$ROOT/training_resources"}
MANIFEST_ROOT=${MIDI_LLM_CLASSICAL_MANIFEST_ROOT:-"$ROOT/training_manifests"}
BACKUP_ROOT=${MIDI_LLM_CLASSICAL_BACKUP_ROOT:-"$HOME/Documents/MidiLLM-resource-backups/classical-resources-v1"}
RCLONE_REMOTE=${MIDI_LLM_RCLONE_REMOTE:-}
BACKUP_ARCHIVES=${MIDI_LLM_CLASSICAL_BACKUP_ARCHIVES:-0}
STAMP=$(date -u +%Y-%m-%dT%H%M%SZ)
SNAPSHOT="$BACKUP_ROOT/snapshots/$STAMP"

mkdir -p "$SNAPSHOT" "$BACKUP_ROOT/latest"
MANIFEST_DIRS=()
for name in pdmx-native-classical-v3 mutopia-piano-v1 classical-resource-inventory-v1 maestro-performance-v3; do
  [[ -d "$MANIFEST_ROOT/$name" ]] && MANIFEST_DIRS+=("$name")
done
if [[ ${#MANIFEST_DIRS[@]} -eq 0 ]]; then
  echo "No classical resource manifests found under $MANIFEST_ROOT" >&2
  exit 2
fi

if [[ -d "$RESOURCE_ROOT/maestro" ]]; then
  find "$RESOURCE_ROOT/maestro" -maxdepth 1 -type f \
    \( -name '*.zip' -o -name '*.csv' -o -name '*.json' \) \
    -exec shasum -a 256 {} + > "$SNAPSHOT/maestro-resource-checksums.sha256"
fi
tar -czf "$SNAPSHOT/classical-resource-manifests.tar.gz" -C "$MANIFEST_ROOT" "${MANIFEST_DIRS[@]}"
sha256sum "$SNAPSHOT/classical-resource-manifests.tar.gz" > "$SNAPSHOT/classical-resource-manifests.tar.gz.sha256"

{
  printf 'backup_utc=%s\n' "$STAMP"
  printf 'git_commit=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
  printf 'resource_root=%s\n' "$RESOURCE_ROOT"
  printf 'manifest_root=%s\n' "$MANIFEST_ROOT"
  if [[ -d "$RESOURCE_ROOT/mutopia/source/.git" ]]; then
    printf 'mutopia_commit=%s\n' "$(git -C "$RESOURCE_ROOT/mutopia/source" rev-parse HEAD)"
  fi
} > "$SNAPSHOT/metadata.txt"

if [[ -d "$RESOURCE_ROOT/pdmx" ]]; then
  find "$RESOURCE_ROOT/pdmx" -maxdepth 1 -type f \
    \( -name '*.tar.gz' -o -name '*.csv' -o -name 'fetch_summary.json' \) \
    -exec shasum -a 256 {} + > "$SNAPSHOT/pdmx-resource-checksums.sha256"
fi

rm -rf "$BACKUP_ROOT/latest"
mkdir -p "$BACKUP_ROOT/latest"
cp "$SNAPSHOT"/* "$BACKUP_ROOT/latest/"

if [[ "$BACKUP_ARCHIVES" != 0 ]]; then
  mkdir -p "$BACKUP_ROOT/resources/pdmx"
  rsync -a "$RESOURCE_ROOT/pdmx/" "$BACKUP_ROOT/resources/pdmx/" \
    --include '*/' --include '*.tar.gz' --include '*.csv' --include 'fetch_summary.json' --exclude '*'
fi

if [[ -n "$RCLONE_REMOTE" ]]; then
  command -v rclone >/dev/null
  rclone copy "$SNAPSHOT" "$RCLONE_REMOTE/classical-resources-v1/snapshots/$STAMP"
  rclone copy "$BACKUP_ROOT/latest" "$RCLONE_REMOTE/classical-resources-v1/latest"
  if [[ "$BACKUP_ARCHIVES" != 0 ]]; then
    rclone sync "$BACKUP_ROOT/resources" "$RCLONE_REMOTE/classical-resources-v1/resources"
  fi
fi

echo "Classical resource backup complete: $SNAPSHOT"
