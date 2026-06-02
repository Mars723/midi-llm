#!/usr/bin/env bash
set -euo pipefail

BASELINE=${1:?Usage: $0 BASELINE_RUN ADAPTER_PARITY_RUN WHOLE_PIECE_RUN OUTPUT_DIR [HUMAN_STATUS] [HUMAN_NOTES]}
ADAPTER=${2:?Usage: $0 BASELINE_RUN ADAPTER_PARITY_RUN WHOLE_PIECE_RUN OUTPUT_DIR [HUMAN_STATUS] [HUMAN_NOTES]}
WHOLE=${3:?Usage: $0 BASELINE_RUN ADAPTER_PARITY_RUN WHOLE_PIECE_RUN OUTPUT_DIR [HUMAN_STATUS] [HUMAN_NOTES]}
OUTPUT=${4:?Usage: $0 BASELINE_RUN ADAPTER_PARITY_RUN WHOLE_PIECE_RUN OUTPUT_DIR [HUMAN_STATUS] [HUMAN_NOTES]}
HUMAN_STATUS=${5:-pending}
HUMAN_NOTES=${6:-Awaiting listening review.}
PYTHON=${PYTHON:-python3}

"$PYTHON" -m midi_llm.native_progress_report \
  --baseline-run "$BASELINE" \
  --adapter-run "$ADAPTER" \
  --whole-piece-run "$WHOLE" \
  --human-review-status "$HUMAN_STATUS" \
  --human-review-notes "$HUMAN_NOTES" \
  --output-dir "$OUTPUT"
