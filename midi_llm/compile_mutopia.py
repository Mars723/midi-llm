"""Compile Mutopia LilyPond sources into an auditable supplemental view."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, Iterable, List, Sequence


def compile_mutopia_manifest(
    manifest: Path | str,
    source_root: Path | str,
    output_dir: Path | str,
    lilypond: Path | str,
    *,
    limit: int | None = None,
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    """Compile selected LilyPond sources without promoting them into training."""

    manifest = Path(manifest)
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    lilypond = Path(lilypond)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(manifest)
    if limit is not None:
        rows = rows[:limit]
    results = [_compile_or_reuse_row(row, source_root, output_dir / "compiled", lilypond, timeout_seconds) for row in rows]
    failures = [row for row in results if not row["compile_status"].startswith("success-")]
    review_candidates = _clean_midi_review_candidates(results)
    solo_piano_review_candidates = [row for row in review_candidates if row["instrument_tier"] == "solo-piano"]
    _write_jsonl(output_dir / "compiled_manifest.jsonl", results)
    _write_jsonl(output_dir / "compile_errors.jsonl", failures)
    _write_jsonl(output_dir / "clean_midi_review_candidates.jsonl", review_candidates)
    _write_jsonl(output_dir / "clean_solo_piano_review_candidates.jsonl", solo_piano_review_candidates)
    summary = {
        "pipeline": "mutopia-lilypond-compilation-audit-v1",
        "manifest": str(manifest.resolve()),
        "source_root": str(source_root.resolve()),
        "lilypond": str(lilypond.resolve()),
        "works_requested": len(rows),
        "compile_status_counts": dict(sorted(Counter(row["compile_status"] for row in results).items())),
        "works_cleanly_compiled": sum(row["compile_status"].startswith("success-") for row in results),
        "works_with_midi": sum(bool(row["compiled_midi_paths"]) for row in results),
        "works_with_clean_midi": sum(row["compile_status"] == "success-with-midi" for row in results),
        "works_with_midi_compile_errors": sum(row["compile_status"] == "midi-with-compile-errors" for row in results),
        "works_without_midi": sum(row["compile_status"] == "success-without-midi" for row in results),
        "works_failed": len(failures),
        "clean_midi_review_candidates": len(review_candidates),
        "clean_solo_piano_review_candidates": len(solo_piano_review_candidates),
        "clean_midi_review_candidate_composer_counts": _count(review_candidates, "composer_style"),
        "clean_midi_review_candidate_genre_counts": _count(review_candidates, "genre"),
        "artifacts": {
            "compiled_manifest": "compiled_manifest.jsonl",
            "errors": "compile_errors.jsonl",
            "compiled_files": "compiled/<work_id>/",
            "clean_midi_review_candidates": "clean_midi_review_candidates.jsonl",
            "clean_solo_piano_review_candidates": "clean_solo_piano_review_candidates.jsonl",
        },
        "promotion_policy": "supplemental-unreviewed until compile, license, instrumentation, and difficulty review pass",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _compile_row(
    row: Dict[str, Any],
    source_root: Path,
    compiled_root: Path,
    lilypond: Path,
    timeout_seconds: int,
) -> Dict[str, Any]:
    work_id = str(row["work_id"])
    source = source_root / str(row["path"])
    work_dir = compiled_root / work_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = work_dir / "score"
    base = {
        **row,
        "compiled_dir": work_dir.relative_to(compiled_root.parent).as_posix(),
        "compiled_midi_paths": [],
        "compiled_pdf_paths": [],
        "compile_returncode": None,
        "compile_error": "",
        "compile_lilypond": str(lilypond.resolve()),
    }
    if not source.is_file():
        return _record_result(
            work_dir,
            {**base, "compile_status": "source-missing", "compile_error": f"missing source: {source}"},
        )
    command = [
        str(lilypond),
        "-dno-point-and-click",
        "-dno-print-pages",
        "-o",
        str(output_prefix),
        str(source),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=source.parent,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return _record_result(work_dir, {**base, "compile_status": "timeout", "compile_error": _bounded(error.stderr or "")})
    except OSError as error:
        return _record_result(work_dir, {**base, "compile_status": "runner-error", "compile_error": str(error)})
    midi_paths = _relative_outputs(work_dir, ("*.mid", "*.midi"))
    pdf_paths = _relative_outputs(work_dir, ("*.pdf",))
    result = {
        **base,
        "compiled_midi_paths": midi_paths,
        "compiled_pdf_paths": pdf_paths,
        "compile_returncode": completed.returncode,
        "compile_error": _bounded(completed.stderr),
    }
    if completed.returncode != 0:
        status = "midi-with-compile-errors" if midi_paths else "compile-failed"
        return _record_result(work_dir, {**result, "compile_status": status})
    status = "success-with-midi" if midi_paths else "success-without-midi"
    return _record_result(work_dir, {**result, "compile_status": status})


def _compile_or_reuse_row(
    row: Dict[str, Any],
    source_root: Path,
    compiled_root: Path,
    lilypond: Path,
    timeout_seconds: int,
) -> Dict[str, Any]:
    checkpoint = compiled_root / str(row["work_id"]) / "compile_result.json"
    if checkpoint.is_file():
        cached = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (
            cached.get("source_sha256") == row.get("source_sha256")
            and cached.get("compile_lilypond") == str(lilypond.resolve())
        ):
            return cached
    return _compile_row(row, source_root, compiled_root, lilypond, timeout_seconds)


def _record_result(work_dir: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    (work_dir / "compile_result.json").write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return result


def _clean_midi_review_candidates(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        row
        for row in rows
        if row["compile_status"] == "success-with-midi"
        and row.get("composer_style") != "unclassified-composer"
        and row.get("license_evidence") != "requires-manual-license-review"
    ]


def _relative_outputs(directory: Path, patterns: Sequence[str]) -> List[str]:
    paths = []
    for pattern in patterns:
        paths.extend(directory.glob(pattern))
    return sorted(path.relative_to(directory.parent.parent).as_posix() for path in paths)


def _bounded(text: str, limit: int = 4000) -> str:
    return text[-limit:]


def _count(rows: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "missing") for row in rows).items()))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Mutopia LilyPond sources into a review-only supplement")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lilypond", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    print(
        json.dumps(
            compile_mutopia_manifest(
                args.manifest,
                args.source_root,
                args.output_dir,
                args.lilypond,
                limit=args.limit,
                timeout_seconds=args.timeout_seconds,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
