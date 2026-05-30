"""Create deduplicated PDMX manifests for the score-first training curriculum."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List


def prepare_manifest(metadata_csv: Path | str, output_dir: Path | str) -> Dict[str, int]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(metadata_csv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    accepted = []
    seen = set()
    for row in rows:
        if not _truthy(row.get("all_valid", "1")):
            continue
        if not _truthy(row.get("no_license_conflict", "1")):
            continue
        if not _is_solo_piano(row):
            continue
        work_id = _work_id(row)
        if work_id in seen:
            continue
        seen.add(work_id)
        split = _split(work_id)
        accepted.append(
            {
                "work_id": work_id,
                "split": split,
                "path": row.get("path") or row.get("mxl_path") or row.get("musicxml_path") or "",
                "title": row.get("title") or row.get("song_name") or "",
                "composer": row.get("composer") or "",
                "measure_count": _measure_count(row),
                "tasks": [
                    "score-dsl-autoencode",
                    "section-expand-16-64",
                    "whole-piece-generate",
                    "masked-span-inpaint",
                    "ending-complete",
                    "recapitulation-revise",
                ],
            }
        )
    manifest_path = output_dir / "pdmx_score_first_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    summary = {
        "input_rows": len(rows),
        "accepted_unique_solo_piano_works": len(accepted),
        "train": sum(row["split"] == "train" for row in accepted),
        "valid": sum(row["split"] == "valid" for row in accepted),
        "test": sum(row["split"] == "test" for row in accepted),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _truthy(value: str) -> bool:
    return str(value).strip().lower() not in ("", "0", "false", "no", "none", "nan")


def _is_solo_piano(row: Dict[str, str]) -> bool:
    text = " ".join(
        str(row.get(key, ""))
        for key in ("tracks", "instrumentation", "instruments", "instrument", "is_piano", "is_solo_piano")
    ).lower()
    if row.get("is_solo_piano"):
        return _truthy(row["is_solo_piano"])
    return "piano" in text and not any(
        word in text for word in ("violin", "cello", "flute", "voice", "orchestra", "guitar", "drum")
    )


def _work_id(row: Dict[str, str]) -> str:
    composer = str(row.get("composer", "")).strip().lower()
    title = str(row.get("title") or row.get("song_name") or "").strip().lower()
    fallback_path = str(row.get("path") or row.get("mxl_path") or row.get("musicxml_path") or "").strip().lower()
    identity = f"{composer}|{title}" if composer or title else fallback_path
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _split(work_id: str) -> str:
    bucket = int(work_id[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "valid"
    return "test"


def _measure_count(row: Dict[str, str]) -> int | None:
    for key in ("measure_count", "n_measures", "measures", "num_measures"):
        value = row.get(key)
        if value:
            try:
                return int(float(value))
            except ValueError:
                pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a public-domain PDMX training manifest")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = prepare_manifest(args.metadata_csv, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
