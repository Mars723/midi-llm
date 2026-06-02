"""Profile MusicXML notation richness for score-first corpus selection."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
import xml.etree.ElementTree as ET
import zipfile


def audit_notation_corpus(
    manifest: Path | str,
    dataset_root: Path | str,
    output_dir: Path | str,
) -> Dict[str, Any]:
    """Write per-work notation profiles and aggregate score-marking coverage."""

    manifest = Path(manifest)
    root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = []
    errors = []
    for row in _read_jsonl(manifest):
        source = _resolve(root, str(row.get("path") or ""))
        try:
            profile = profile_musicxml(source)
        except (OSError, ValueError, ET.ParseError, zipfile.BadZipFile) as error:
            errors.append({"work_id": row.get("work_id"), "path": str(source), "reason": str(error)})
            continue
        profiles.append(
            {
                "work_id": row.get("work_id"),
                "split": row.get("split"),
                "source_dataset": row.get("source_dataset"),
                "path": row.get("path"),
                "composer_style": row.get("composer_style"),
                "composer_period": row.get("composer_period"),
                "genre": row.get("genre"),
                "difficulty": row.get("difficulty"),
                "quality_tier": row.get("quality_tier"),
                "notation_profile": profile,
            }
        )
    _write_jsonl(output_dir / "notation_profiles.jsonl", profiles)
    _write_jsonl(output_dir / "notation_errors.jsonl", errors)
    marked_profiles = [
        row
        for row in profiles
        if row["notation_profile"]["marking_tier"] in ("marked-two-staff", "expressive-two-staff")
    ]
    expressive_profiles = [
        row for row in profiles if row["notation_profile"]["marking_tier"] == "expressive-two-staff"
    ]
    _write_jsonl(output_dir / "marked_two_staff_profiles.jsonl", marked_profiles)
    _write_jsonl(output_dir / "expressive_two_staff_profiles.jsonl", expressive_profiles)
    summary = {
        "pipeline": "score-first-notation-corpus-audit-v1",
        "manifest": str(manifest.resolve()),
        "dataset_root": str(root.resolve()),
        "works_requested": len(profiles) + len(errors),
        "works_profiled": len(profiles),
        "works_failed": len(errors),
        "marking_tier_counts": dict(sorted(Counter(row["notation_profile"]["marking_tier"] for row in profiles).items())),
        "two_staff_work_count": sum(row["notation_profile"]["max_staves"] >= 2 for row in profiles),
        "score_editor_views": {
            "marked_two_staff_works": len(marked_profiles),
            "expressive_two_staff_works": len(expressive_profiles),
        },
        "direction_coverage": {
            field: {
                "works": sum(row["notation_profile"][field] > 0 for row in profiles),
                "total_events": sum(row["notation_profile"][field] for row in profiles),
            }
            for field in (
                "tempo_events",
                "dynamic_events",
                "pedal_events",
                "wedge_events",
                "articulation_events",
                "fingering_events",
            )
        },
        "artifacts": {
            "profiles": "notation_profiles.jsonl",
            "marked_two_staff_profiles": "marked_two_staff_profiles.jsonl",
            "expressive_two_staff_profiles": "expressive_two_staff_profiles.jsonl",
            "errors": "notation_errors.jsonl",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def profile_musicxml(path: Path | str) -> Dict[str, Any]:
    """Extract notation features from plain MusicXML or compressed MXL."""

    path = Path(path)
    if not path.is_file():
        raise OSError(f"Score source does not exist: {path}")
    root = ET.fromstring(_read_musicxml(path))
    counts = Counter(_local_name(element.tag) for element in root.iter())
    staff_values = [
        int(element.text)
        for element in root.iter()
        if _local_name(element.tag) == "staves" and str(element.text or "").strip().isdigit()
    ]
    dynamic_events = sum(
        1
        for element in root.iter()
        if _local_name(element.tag) == "dynamics"
    )
    tempo_events = sum(
        1
        for element in root.iter()
        if _local_name(element.tag) == "sound" and element.attrib.get("tempo")
    )
    profile = {
        "measures": counts["measure"],
        "notes": counts["note"],
        "parts": counts["part"],
        "max_staves": max(staff_values, default=1),
        "tempo_events": tempo_events,
        "dynamic_events": dynamic_events,
        "pedal_events": counts["pedal"],
        "wedge_events": counts["wedge"],
        "articulation_events": counts["articulations"],
        "fingering_events": counts["fingering"],
    }
    profile["marking_tier"] = _marking_tier(profile)
    return profile


def _marking_tier(profile: Dict[str, Any]) -> str:
    expressive_fields = ("dynamic_events", "pedal_events", "wedge_events")
    if profile["max_staves"] >= 2 and all(profile[field] > 0 for field in expressive_fields):
        return "expressive-two-staff"
    if profile["max_staves"] >= 2 and any(profile[field] > 0 for field in expressive_fields):
        return "marked-two-staff"
    if profile["max_staves"] >= 2:
        return "basic-two-staff"
    return "single-staff-or-unknown"


def _read_musicxml(path: Path) -> bytes:
    if path.suffix.casefold() not in (".mxl", ".zip"):
        return path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.casefold().endswith((".xml", ".musicxml"))]
        score_names = [name for name in names if name.casefold() != "meta-inf/container.xml"]
        if not score_names:
            raise ValueError(f"Compressed score has no MusicXML document: {path}")
        return archive.read(score_names[0])


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MusicXML notation richness for score-first selection")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(audit_notation_corpus(args.manifest, args.dataset_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
