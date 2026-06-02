"""Audit collected classical-piano manifests before allocating training compute."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_REQUIRED_COMPOSERS = ("bach", "mozart", "beethoven", "chopin")


def audit_manifest(
    manifest: Path | str,
    *,
    dataset_root: Path | str | None = None,
    required_composers: Sequence[str] = DEFAULT_REQUIRED_COMPOSERS,
    min_composer_works: int = 10,
    expect_files: bool = False,
) -> Dict[str, Any]:
    """Report label coverage, source provenance, and extracted-file readiness."""

    manifest = Path(manifest)
    rows = _read_jsonl(manifest)
    root = Path(dataset_root) if dataset_root is not None else None
    composer_counts = Counter(str(row.get("composer_style") or "missing") for row in rows)
    work_keys = [(str(row.get("source_dataset") or ""), str(row.get("work_id") or "")) for row in rows]
    score_file_count = _existing_file_count(rows, root, "path")
    midi_file_count = _existing_file_count(rows, root, "native_midi_path")
    missing_score_paths = sum(not row.get("path") for row in rows)
    missing_midi_paths = sum(not row.get("native_midi_path") for row in rows)
    source_license_missing = sum(not str(row.get("source_license") or "").strip() for row in rows)
    unclassified_composer_count = composer_counts["unclassified-composer"] + composer_counts["missing"]
    genre_counts = Counter(str(row.get("genre") or "missing") for row in rows)
    unclassified_genre_count = genre_counts["unclassified-piano"] + genre_counts["missing"]
    composer_requirements = {
        composer: {
            "works": composer_counts[composer],
            "minimum": min_composer_works,
            "passed": composer_counts[composer] >= min_composer_works,
        }
        for composer in required_composers
    }
    checks = {
        "manifest_is_not_empty": bool(rows),
        "work_keys_are_unique": len(set(work_keys)) == len(work_keys),
        "work_ids_are_present": all(work_id for _, work_id in work_keys),
        "source_datasets_are_present": all(source for source, _ in work_keys),
        "required_composer_coverage": all(item["passed"] for item in composer_requirements.values()),
        "score_paths_are_present": missing_score_paths == 0,
        "native_midi_paths_are_present": missing_midi_paths == 0,
    }
    if expect_files:
        checks["score_files_are_extracted"] = score_file_count == len(rows)
        checks["native_midi_files_are_extracted"] = midi_file_count == len(rows)
    report = {
        "pipeline": "classical-piano-source-audit-v1",
        "manifest": str(manifest.resolve()),
        "dataset_root": str(root.resolve()) if root is not None else None,
        "works": len(rows),
        "counts": {
            "source_dataset": _count(rows, "source_dataset"),
            "composer_style": dict(sorted(composer_counts.items())),
            "composer_period": _count(rows, "composer_period"),
            "genre": dict(sorted(genre_counts.items())),
            "form": _count(rows, "form"),
            "difficulty": _count(rows, "difficulty"),
            "quality_tier": _count(rows, "quality_tier"),
            "source_license": _count(rows, "source_license"),
        },
        "coverage": {
            "required_composers": composer_requirements,
            "unclassified_composer_count": unclassified_composer_count,
            "unclassified_composer_ratio": _ratio(unclassified_composer_count, len(rows)),
            "unclassified_genre_count": unclassified_genre_count,
            "unclassified_genre_ratio": _ratio(unclassified_genre_count, len(rows)),
            "measure_count": _number_summary(row.get("measure_count") for row in rows),
        },
        "files": {
            "expect_files": expect_files,
            "score_paths_missing": missing_score_paths,
            "native_midi_paths_missing": missing_midi_paths,
            "score_files_extracted": score_file_count,
            "native_midi_files_extracted": midi_file_count,
        },
        "provenance": {
            "missing_source_license_count": source_license_missing,
            "license_review_required_before_commercial_release": True,
        },
        "checks": checks,
        "ready": all(checks.values()),
    }
    return report


def write_audit_report(report: Dict[str, Any], output_dir: Path | str) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "audit-report.json"
    markdown_path = output_dir / "audit-report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _markdown_report(report: Dict[str, Any]) -> str:
    composers = report["coverage"]["required_composers"]
    composer_rows = "\n".join(
        f"| {composer} | {item['works']} | {item['minimum']} | {'yes' if item['passed'] else 'no'} |"
        for composer, item in composers.items()
    )
    checks = "\n".join(f"- `{name}`: **{'pass' if passed else 'fail'}**" for name, passed in report["checks"].items())
    return f"""# Classical Piano Dataset Audit

- Works: **{report['works']}**
- Ready: **{'yes' if report['ready'] else 'no'}**
- Unclassified composer ratio: **{report['coverage']['unclassified_composer_ratio']:.1%}**
- Unclassified genre ratio: **{report['coverage']['unclassified_genre_ratio']:.1%}**
- Extracted score files: **{report['files']['score_files_extracted']}**
- Extracted native MIDI files: **{report['files']['native_midi_files_extracted']}**

## Required Composer Coverage

| Composer | Works | Minimum | Passed |
| --- | ---: | ---: | --- |
{composer_rows}

## Checks

{checks}
"""


def _existing_file_count(rows: Iterable[Dict[str, Any]], root: Path | None, field: str) -> int:
    if root is None:
        return 0
    return sum(bool(row.get(field)) and _resolve(root, str(row[field])).is_file() for row in rows)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _number_summary(values: Iterable[Any]) -> Dict[str, float | int | None]:
    numbers = sorted(float(value) for value in values if value not in (None, ""))
    if not numbers:
        return {"min": None, "median": None, "max": None}
    return {"min": numbers[0], "median": numbers[len(numbers) // 2], "max": numbers[-1]}


def _count(rows: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "missing") for row in rows).items()))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a collected classical-piano manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--required-composers", default=",".join(DEFAULT_REQUIRED_COMPOSERS))
    parser.add_argument("--min-composer-works", type=int, default=10)
    parser.add_argument("--expect-files", action="store_true")
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    report = audit_manifest(
        args.manifest,
        dataset_root=args.dataset_root,
        required_composers=tuple(part.strip() for part in args.required_composers.split(",") if part.strip()),
        min_composer_works=args.min_composer_works,
        expect_files=args.expect_files,
    )
    outputs = write_audit_report(report, args.output_dir)
    print(json.dumps({**report, "artifacts": outputs}, indent=2))
    raise SystemExit(0 if report["ready"] or not args.enforce else 1)


if __name__ == "__main__":
    main()
