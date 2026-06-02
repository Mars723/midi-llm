"""Merge source manifests without losing source namespaces or audit metadata."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def merge_manifests(manifests: Sequence[Path | str], output: Path | str) -> Dict[str, Any]:
    """Merge JSONL manifests by source dataset and work ID."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    source_rows: Dict[str, int] = {}
    duplicates = 0
    for manifest in manifests:
        path = Path(manifest)
        rows = _read_jsonl(path)
        source_rows[str(path)] = len(rows)
        for row in rows:
            key = (str(row.get("source_dataset") or ""), str(row["work_id"]))
            if key in merged:
                duplicates += 1
                continue
            merged[key] = row
    rows = sorted(merged.values(), key=lambda row: (str(row.get("source_dataset") or ""), str(row["work_id"])))
    _write_jsonl(output, rows)
    summary = {
        "pipeline": "score-first-source-manifest-merge-v1",
        "input_manifests": source_rows,
        "input_rows": sum(source_rows.values()),
        "duplicate_rows_removed": duplicates,
        "merged_unique_works": len(rows),
        "source_dataset_counts": _count(rows, "source_dataset"),
        "composer_style_counts": _count(rows, "composer_style"),
        "composer_period_counts": _count(rows, "composer_period"),
        "genre_counts": _count(rows, "genre"),
        "difficulty_counts": _count(rows, "difficulty"),
        "quality_counts": _count(rows, "quality_tier"),
        "artifact": str(output),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _count(rows: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "missing") for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge source manifests with namespace-aware deduplication")
    parser.add_argument("--manifest", action="append", required=True, help="Input JSONL manifest; repeat as needed")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(merge_manifests(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
