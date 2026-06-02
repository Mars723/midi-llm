#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESOURCE_ROOT=${MIDI_LLM_PDMX_ROOT:-"$ROOT/training_resources/pdmx"}
MANIFEST_ROOT=${MIDI_LLM_NATIVE_MANIFEST_ROOT:-"$ROOT/training_manifests/pdmx-native-classical-v3"}
CORE_COMPOSERS=${MIDI_LLM_NATIVE_CORE_COMPOSERS:-bach,mozart,beethoven,chopin}
CONNECTIONS=${MIDI_LLM_PDMX_CONNECTIONS:-6}
PHASE=${1:-all}

cd "$ROOT"
mkdir -p "$RESOURCE_ROOT" "$MANIFEST_ROOT"

prepare_manifests() {
  python3 -m midi_llm.fetch_pdmx \
    --output-dir "$RESOURCE_ROOT" \
    --extract-subsets \
    --connections "$CONNECTIONS"

  python3 -m midi_llm.prepare_pdmx \
    --metadata-csv "$RESOURCE_ROOT/PDMX.csv" \
    --output-dir "$MANIFEST_ROOT/catalog-all"

  python3 -m midi_llm.prepare_pdmx \
    --metadata-csv "$RESOURCE_ROOT/PDMX.csv" \
    --output-dir "$MANIFEST_ROOT/core-four-composers" \
    --composers "$CORE_COMPOSERS"

  python3 -m midi_llm.prepare_pdmx \
    --metadata-csv "$RESOURCE_ROOT/PDMX.csv" \
    --output-dir "$MANIFEST_ROOT/intermediate-curated" \
    --difficulty intermediate \
    --quality metadata-curated \
    --min-measures 32 \
    --max-measures 256

  python3 -m midi_llm.prepare_pdmx \
    --metadata-csv "$RESOURCE_ROOT/PDMX.csv" \
    --output-dir "$MANIFEST_ROOT/core-four-composers-intermediate" \
    --composers "$CORE_COMPOSERS" \
    --difficulty intermediate \
    --min-measures 32 \
    --max-measures 256

  python3 -m midi_llm.merge_manifests \
    --manifest "$MANIFEST_ROOT/core-four-composers/pdmx_score_first_manifest.jsonl" \
    --manifest "$MANIFEST_ROOT/intermediate-curated/pdmx_score_first_manifest.jsonl" \
    --output "$MANIFEST_ROOT/collected_manifest.jsonl"

  python3 -m midi_llm.merge_manifests \
    --manifest "$MANIFEST_ROOT/core-four-composers-intermediate/pdmx_score_first_manifest.jsonl" \
    --manifest "$MANIFEST_ROOT/intermediate-curated/pdmx_score_first_manifest.jsonl" \
    --output "$MANIFEST_ROOT/training_view_manifest.jsonl"

  python3 -m midi_llm.dataset_audit \
    --manifest "$MANIFEST_ROOT/collected_manifest.jsonl" \
    --dataset-root "$RESOURCE_ROOT" \
    --output-dir "$MANIFEST_ROOT/audit-before-extract" \
    --min-composer-works 10

  python3 -m midi_llm.dataset_audit \
    --manifest "$MANIFEST_ROOT/catalog-all/pdmx_score_first_manifest.jsonl" \
    --dataset-root "$RESOURCE_ROOT" \
    --output-dir "$MANIFEST_ROOT/audit-catalog-all" \
    --min-composer-works 10

  python3 -m midi_llm.dataset_audit \
    --manifest "$MANIFEST_ROOT/training_view_manifest.jsonl" \
    --dataset-root "$RESOURCE_ROOT" \
    --output-dir "$MANIFEST_ROOT/audit-training-view" \
    --min-composer-works 10
}

extract_catalog_midi() {
  python3 -m midi_llm.fetch_pdmx \
    --output-dir "$RESOURCE_ROOT" \
    --include-midi \
    --extract-midi \
    --midi-manifest "$MANIFEST_ROOT/catalog-all/pdmx_score_first_manifest.jsonl" \
    --connections "$CONNECTIONS"
}

extract_resources() {
  python3 -m midi_llm.fetch_pdmx \
    --output-dir "$RESOURCE_ROOT" \
    --include-midi \
    --include-mxl \
    --extract-midi \
    --extract-mxl \
    --midi-manifest "$MANIFEST_ROOT/collected_manifest.jsonl" \
    --mxl-manifest "$MANIFEST_ROOT/collected_manifest.jsonl" \
    --connections "$CONNECTIONS"

  python3 -m midi_llm.dataset_audit \
    --manifest "$MANIFEST_ROOT/collected_manifest.jsonl" \
    --dataset-root "$RESOURCE_ROOT" \
    --output-dir "$MANIFEST_ROOT/audit-after-extract" \
    --min-composer-works 10 \
    --expect-files \
    --enforce
}

materialize_tokens() {
  python3 -m midi_llm.materialize_native_training \
    --manifest "$MANIFEST_ROOT/training_view_manifest.jsonl" \
    --dataset-root "$RESOURCE_ROOT" \
    --output-dir "$MANIFEST_ROOT/native_tokens_training_view"
}

audit_notation() {
  python3 -m midi_llm.notation_corpus_audit \
    --manifest "$MANIFEST_ROOT/training_view_manifest.jsonl" \
    --dataset-root "$RESOURCE_ROOT" \
    --output-dir "$MANIFEST_ROOT/notation-audit-training-view"
}

case "$PHASE" in
  manifests)
    prepare_manifests
    ;;
  extract)
    extract_resources
    ;;
  extract-catalog-midi)
    extract_catalog_midi
    ;;
  tokens)
    materialize_tokens
    ;;
  notation-audit)
    audit_notation
    ;;
  all)
    prepare_manifests
    extract_resources
    audit_notation
    materialize_tokens
    ;;
  *)
    echo "Usage: $0 {manifests|extract|extract-catalog-midi|notation-audit|tokens|all}" >&2
    exit 2
    ;;
esac
