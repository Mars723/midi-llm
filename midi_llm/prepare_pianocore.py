"""Prepare a metadata-only PianoCoRe research inventory."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


PIANOCORE_VERSION = "1.0"
PIANOCORE_RECORD = "19186016"
PIANOCORE_URL = f"https://zenodo.org/records/{PIANOCORE_RECORD}"
PIANOCORE_LICENSE = "cc-by-nc-sa-4.0"
PIANOCORE_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
USE_CHANNEL = "non-commercial-research-performance-overlay-and-alignment-only"


def prepare_pianocore_metadata(
    metadata_csv: Path | str,
    composers_csv: Path | str,
    output_dir: Path | str,
) -> Dict[str, Any]:
    """Write an isolated research inventory without downloading the large archive."""

    metadata_csv = Path(metadata_csv)
    composers_csv = Path(composers_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            rows.append(_inventory_row(source))
    composers = []
    with composers_csv.open(newline="", encoding="utf-8") as handle:
        composers.extend(csv.DictReader(handle))
    _write_jsonl(output_dir / "pianocore_research_manifest.jsonl", rows)
    unique_works = {
        (
            row["composer"],
            row["composition"],
            row["movement"],
            row["score_dataset"],
            row["score_id"],
            row["score_xml_path"],
        )
        for row in rows
    }
    summary = {
        "pipeline": "pianocore-metadata-research-inventory-v1",
        "source_dataset": "PianoCoRe",
        "source_dataset_version": PIANOCORE_VERSION,
        "source_dataset_url": PIANOCORE_URL,
        "source_license": PIANOCORE_LICENSE,
        "source_license_url": PIANOCORE_LICENSE_URL,
        "commercial_use_allowed": False,
        "composition_backbone_eligible": False,
        "use_channel": USE_CHANNEL,
        "archive_downloaded": False,
        "metadata_rows": len(rows),
        "unique_works": len(unique_works),
        "composers": len(composers),
        "split_counts": _count(rows, "split"),
        "tier_b_counts": _count(rows, "tier_b"),
        "tier_a_counts": _count(rows, "tier_a"),
        "tier_a_star_counts": _count(rows, "tier_a_star"),
        "score_dataset_counts": _count(rows, "score_dataset"),
        "performance_dataset_counts": _count(rows, "performance_dataset"),
        "quality_label_counts": _count(rows, "quality_label"),
        "metadata_checksums": {
            metadata_csv.name: _sha256(metadata_csv),
            composers_csv.name: _sha256(composers_csv),
        },
        "artifacts": {"manifest": "pianocore_research_manifest.jsonl"},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _inventory_row(source: Dict[str, str]) -> Dict[str, Any]:
    return {
        "id": source["id"],
        "split": source["split"],
        "source_dataset": "PianoCoRe",
        "source_dataset_version": PIANOCORE_VERSION,
        "source_dataset_url": PIANOCORE_URL,
        "source_license": PIANOCORE_LICENSE,
        "source_license_url": PIANOCORE_LICENSE_URL,
        "commercial_use_allowed": False,
        "composition_backbone_eligible": False,
        "use_channel": USE_CHANNEL,
        "composer": source["composer"],
        "composition": source["composition"],
        "movement": source["movement"],
        "score_dataset": source["score_dataset"],
        "score_id": source["score_id"],
        "score_xml_path": source["score_xml_path"],
        "score_midi_path": source["score_midi_path"],
        "performance_id": source["performance_id"],
        "performance_dataset": source["performance_dataset"],
        "performance_midi_path": source["performance_midi_path"],
        "quality_label": source["quality_label"],
        "tier_b": source["tier_b"],
        "tier_a": source["tier_a"],
        "tier_a_star": source["tier_a_star"],
        "is_transcription": source["is_transcription"],
        "is_duplicate": source["is_duplicate"],
        "is_refined": source["is_refined"],
        "refined_score_midi_path": source["refined_score_midi_path"],
        "refined_performance_midi_path": source["refined_performance_midi_path"],
        "refined_alignment_path": source["refined_alignment_path"],
    }


def _count(rows: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "missing") for row in rows).items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an isolated PianoCoRe metadata research inventory")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--composers-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_pianocore_metadata(args.metadata_csv, args.composers_csv, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
