"""Prepare a metadata-only ASAP non-commercial research inventory."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable


ASAP_REPOSITORY = "https://github.com/fosfrancesco/asap-dataset"
ASAP_LICENSE = "cc-by-nc-sa-4.0"
ASAP_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
USE_CHANNEL = "non-commercial-research-score-performance-alignment-only"


def prepare_asap_metadata(
    metadata_csv: Path | str,
    output_dir: Path | str,
    *,
    repository_commit: str,
) -> Dict[str, Any]:
    """Write an isolated metadata inventory without checking out score blobs."""

    metadata_csv = Path(metadata_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        rows = [_inventory_row(row, repository_commit) for row in csv.DictReader(handle)]
    _write_jsonl(output_dir / "asap_research_manifest.jsonl", rows)
    summary = {
        "pipeline": "asap-metadata-research-inventory-v1",
        "source_dataset": "ASAP",
        "source_repository": ASAP_REPOSITORY,
        "source_repository_commit": repository_commit,
        "source_license": ASAP_LICENSE,
        "source_license_url": ASAP_LICENSE_URL,
        "commercial_use_allowed": False,
        "composition_backbone_eligible": False,
        "use_channel": USE_CHANNEL,
        "score_blobs_checked_out": False,
        "performance_rows": len(rows),
        "unique_title_composer_works": len({(row["composer"], row["title"]) for row in rows}),
        "unique_score_musicxml_paths": len({row["score_musicxml_path"] for row in rows}),
        "unique_score_midi_paths": len({row["score_midi_path"] for row in rows}),
        "composer_counts": _count(rows, "composer"),
        "metadata_sha256": _sha256(metadata_csv),
        "artifacts": {"manifest": "asap_research_manifest.jsonl"},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _inventory_row(source: Dict[str, str], repository_commit: str) -> Dict[str, Any]:
    return {
        "source_dataset": "ASAP",
        "source_repository": ASAP_REPOSITORY,
        "source_repository_commit": repository_commit,
        "source_license": ASAP_LICENSE,
        "source_license_url": ASAP_LICENSE_URL,
        "commercial_use_allowed": False,
        "composition_backbone_eligible": False,
        "use_channel": USE_CHANNEL,
        "composer": source["composer"],
        "title": source["title"],
        "folder": source["folder"],
        "score_musicxml_path": source["xml_score"],
        "score_midi_path": source["midi_score"],
        "performance_midi_path": source["midi_performance"],
        "performance_annotations_path": source["performance_annotations"],
        "score_annotations_path": source["midi_score_annotations"],
        "maestro_midi_performance_path": source["maestro_midi_performance"],
        "maestro_audio_performance_path": source["maestro_audio_performance"],
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
    parser = argparse.ArgumentParser(description="Prepare an isolated ASAP metadata research inventory")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-commit", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_asap_metadata(
                args.metadata_csv,
                args.output_dir,
                repository_commit=args.repository_commit,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
