"""Render a PDF review gallery for Mutopia intermediate-proxy candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, Iterable, List


def render_mutopia_review_gallery(
    manifest: Path | str,
    source_root: Path | str,
    compile_root: Path | str,
    output_dir: Path | str,
    lilypond: Path | str,
    *,
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    """Render source-faithful PDFs and an HTML index without promoting rows."""

    manifest = Path(manifest)
    source_root = Path(source_root)
    compile_root = Path(compile_root)
    output_dir = Path(output_dir)
    lilypond = Path(lilypond)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        _render_or_reuse_row(row, source_root, compile_root, output_dir / "scores", lilypond, timeout_seconds)
        for row in _read_jsonl(manifest)
    ]
    failures = [row for row in results if row["render_status"] != "success-with-pdf"]
    _write_jsonl(output_dir / "review_gallery_manifest.jsonl", results)
    _write_jsonl(output_dir / "render_errors.jsonl", failures)
    (output_dir / "index.html").write_text(_gallery_html(results, output_dir), encoding="utf-8")
    summary = {
        "pipeline": "mutopia-intermediate-proxy-pdf-review-gallery-v1",
        "manifest": str(manifest.resolve()),
        "source_root": str(source_root.resolve()),
        "lilypond": str(lilypond.resolve()),
        "works_requested": len(results),
        "render_status_counts": dict(sorted(Counter(row["render_status"] for row in results).items())),
        "works_with_pdf": sum(row["render_status"] == "success-with-pdf" for row in results),
        "works_failed": len(failures),
        "artifacts": {
            "gallery": "index.html",
            "manifest": "review_gallery_manifest.jsonl",
            "errors": "render_errors.jsonl",
            "scores": "scores/<work_id>/score.pdf",
        },
        "promotion_policy": "human review required before any candidate enters composition training",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _render_or_reuse_row(
    row: Dict[str, Any],
    source_root: Path,
    compile_root: Path,
    scores_root: Path,
    lilypond: Path,
    timeout_seconds: int,
) -> Dict[str, Any]:
    work_dir = scores_root / str(row["work_id"])
    checkpoint = work_dir / "render_result.json"
    if checkpoint.is_file():
        cached = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (
            cached.get("source_sha256") == row.get("source_sha256")
            and cached.get("render_lilypond") == str(lilypond.resolve())
            and (scores_root.parent / cached.get("review_pdf_path", "")).is_file()
        ):
            return cached
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    source = source_root / str(row["path"])
    output_prefix = work_dir / "score"
    base = {
        **row,
        "render_lilypond": str(lilypond.resolve()),
        "review_pdf_path": "",
        "review_midi_path": os.path.relpath(compile_root / row["native_midi_path"], scores_root.parent),
        "render_returncode": None,
        "render_error": "",
    }
    if not source.is_file():
        return _record(work_dir, {**base, "render_status": "source-missing", "render_error": f"missing source: {source}"})
    try:
        completed = subprocess.run(
            [str(lilypond), "-dno-point-and-click", "-o", str(output_prefix), str(source)],
            cwd=source.parent,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return _record(work_dir, {**base, "render_status": "timeout", "render_error": _bounded(error.stderr or "")})
    except OSError as error:
        return _record(work_dir, {**base, "render_status": "runner-error", "render_error": str(error)})
    pdf = output_prefix.with_suffix(".pdf")
    result = {
        **base,
        "review_pdf_path": pdf.relative_to(scores_root.parent).as_posix() if pdf.is_file() else "",
        "render_returncode": completed.returncode,
        "render_error": _bounded(completed.stderr),
    }
    status = "success-with-pdf" if completed.returncode == 0 and pdf.is_file() else "render-failed"
    return _record(work_dir, {**result, "render_status": status})


def _gallery_html(rows: Iterable[Dict[str, Any]], output_dir: Path) -> str:
    table_rows = []
    for row in rows:
        pdf = _link(row.get("review_pdf_path", ""), "PDF")
        midi = _link(row.get("review_midi_path", ""), "MIDI")
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('composer_style', '')))}</td>"
            f"<td>{escape(str(row.get('title', '')))}</td>"
            f"<td>{escape(str(row.get('genre', '')))}</td>"
            f"<td>{escape(str(row.get('difficulty_proxy', '')))}</td>"
            f"<td>{pdf}</td><td>{midi}</td>"
            f"<td>{escape(str(row.get('render_status', '')))}</td>"
            "</tr>"
        )
    return (
        "<!doctype html><meta charset=\"utf-8\"><title>Mutopia Intermediate Review</title>"
        "<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ddd;padding:6px;text-align:left}th{background:#f4f4f4}</style>"
        "<h1>Mutopia intermediate-proxy review gallery</h1>"
        "<p>Review-only. Difficulty proxies are not formal labels and no row is promoted automatically.</p>"
        "<table><thead><tr><th>Composer</th><th>Title</th><th>Genre</th><th>Difficulty proxy</th>"
        "<th>Score</th><th>MIDI</th><th>Render</th></tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table>"
    )


def _link(path: str, label: str) -> str:
    return f'<a href="{escape(path, quote=True)}">{label}</a>' if path else ""


def _record(work_dir: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    (work_dir / "render_result.json").write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return result


def _bounded(text: str, limit: int = 4000) -> str:
    return text[-limit:]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a PDF review gallery for Mutopia intermediate-proxy candidates")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--compile-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lilypond", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    print(
        json.dumps(
            render_mutopia_review_gallery(
                args.manifest,
                args.source_root,
                args.compile_root,
                args.output_dir,
                args.lilypond,
                timeout_seconds=args.timeout_seconds,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
