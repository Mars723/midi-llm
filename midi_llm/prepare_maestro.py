"""Prepare an isolated non-commercial MAESTRO performance-overlay manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .prepare_pdmx import (
    _composer_period,
    _composer_style_label,
    _form_label,
    _genre_label,
    _style_tags,
)


MAESTRO_VERSION = "3.0.0"
MAESTRO_DATASET_URL = "https://magenta.withgoogle.com/datasets/maestro"
MAESTRO_LICENSE = "cc-by-nc-sa-4.0"
MAESTRO_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"


def prepare_maestro_manifest(
    metadata_csv: Path | str,
    output_dir: Path | str,
    *,
    archive_prefix: str = "maestro-v3.0.0",
) -> Dict[str, Any]:
    """Write performance-only rows with an explicit non-commercial boundary."""

    metadata_csv = Path(metadata_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with metadata_csv.open(encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for source in source_rows:
        composer = source.get("canonical_composer") or ""
        title = source.get("canonical_title") or ""
        midi_filename = source.get("midi_filename") or ""
        work_id = _digest(f"MAESTRO-work|{composer}|{title}")
        performance_id = _digest(f"MAESTRO-performance|{midi_filename}")
        composer_style, composer_style_evidence = _composer_style_label({"composer": composer})
        composer_period = _composer_period(composer_style)
        genre, genre_evidence = _genre_label({"title": title})
        form, form_evidence = _form_label({"title": title}, genre)
        rows.append(
            {
                "performance_id": performance_id,
                "work_id": work_id,
                "split": source.get("split") or "",
                "source_dataset": "MAESTRO",
                "source_dataset_version": MAESTRO_VERSION,
                "source_dataset_url": MAESTRO_DATASET_URL,
                "source_license": MAESTRO_LICENSE,
                "source_license_url": MAESTRO_LICENSE_URL,
                "commercial_use_allowed": False,
                "use_channel": "non-commercial-research-performance-overlay-only",
                "path": f"{archive_prefix}/{midi_filename}",
                "title": title,
                "composer": composer,
                "composer_style": composer_style,
                "composer_period": composer_period,
                "composer_style_evidence": composer_style_evidence,
                "genre": genre,
                "genre_evidence": genre_evidence,
                "form": form,
                "form_evidence": form_evidence,
                "style_tags": _style_tags(composer_style, composer_period, genre),
                "performance_year": source.get("year") or "",
                "duration_seconds": _optional_float(source.get("duration")),
                "expressive_controls": ["velocity", "sustain-pedal", "sostenuto-pedal", "una-corda-pedal"],
                "composition_backbone_eligible": False,
            }
        )
    _write_jsonl(output_dir / "maestro_performance_manifest.jsonl", rows)
    summary = {
        "pipeline": "maestro-non-commercial-performance-overlay-v1",
        "source_dataset": "MAESTRO",
        "source_dataset_version": MAESTRO_VERSION,
        "source_dataset_url": MAESTRO_DATASET_URL,
        "source_license": MAESTRO_LICENSE,
        "source_license_url": MAESTRO_LICENSE_URL,
        "commercial_use_allowed": False,
        "use_channel": "non-commercial-research-performance-overlay-only",
        "performances": len(rows),
        "unique_works": len({row["work_id"] for row in rows}),
        "duration_hours": round(sum(row["duration_seconds"] or 0 for row in rows) / 3600, 2),
        "split_counts": _count(rows, "split"),
        "composer_style_counts": _count(rows, "composer_style"),
        "composer_period_counts": _count(rows, "composer_period"),
        "genre_counts": _count(rows, "genre"),
        "artifacts": {"manifest": "maestro_performance_manifest.jsonl"},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _count(rows: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "missing") for row in rows).items()))


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare isolated MAESTRO performance-overlay metadata")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--archive-prefix", default="maestro-v3.0.0")
    args = parser.parse_args()
    print(json.dumps(prepare_maestro_manifest(args.metadata_csv, args.output_dir, archive_prefix=args.archive_prefix), indent=2))


if __name__ == "__main__":
    main()
