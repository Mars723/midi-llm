"""Generate and evaluate complete pieces for the score-first v1 release gate."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence
import xml.etree.ElementTree as ET

from .compose import compose
from .score_ir import read_score


@dataclass(frozen=True)
class GateProfile:
    genre: str
    form: str
    key: str
    meter: str
    tempo: int
    prompt: str


PROFILES: Sequence[GateProfile] = (
    GateProfile("nocturne", "ABA", "C minor", "4/4", 72, "A lyrical nocturne with a tense middle section and a calm return."),
    GateProfile("prelude", "binary", "C major", "4/4", 88, "A concise prelude with a lucid opening and a decisive close."),
    GateProfile("waltz", "ternary", "A minor", "3/4", 108, "A restrained classical waltz with a contrasting central episode."),
    GateProfile("etude", "ABA", "E minor", "4/4", 104, "A flowing piano etude with clear motivic return and an idiomatic cadence."),
    GateProfile("minuet", "binary", "G major", "3/4", 96, "A balanced minuet with clean phrase endings and modest ornamentation."),
    GateProfile("impromptu", "free-sectional", "Eb major", "4/4", 92, "A free but coherent impromptu that settles into a quiet final cadence."),
    GateProfile("theme-and-variations", "theme-and-variations", "D major", "4/4", 84, "A classical theme followed by distinct piano variations and a complete ending."),
    GateProfile("prelude", "rondo", "F minor", "4/4", 80, "A dark rondo prelude whose refrain returns recognizably before the close."),
)

SONATA_PROFILE = GateProfile(
    "prelude",
    "sonata-allegro",
    "D minor",
    "4/4",
    116,
    "An experimental sonata-allegro piano movement with exposition, development, recapitulation, and coda.",
)


def run_release_gate(
    output_dir: Path | str,
    piece_count: int = 50,
    candidates: int = 4,
    seed: int = 2300,
    skip_musescore: bool = False,
    include_sonata: bool = False,
) -> Dict[str, Any]:
    """Run full-piece generation and write an auditable release-gate report."""

    if piece_count < 1:
        raise ValueError("piece_count must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for index in range(piece_count):
        profile = PROFILES[index % len(PROFILES)]
        rows.append(
            _run_profile(
                profile,
                output_dir / f"piece_{index + 1:03d}",
                candidates=candidates,
                seed=seed + index * 17,
                skip_musescore=skip_musescore,
                experimental=False,
            )
        )
    experimental_rows = []
    if include_sonata:
        experimental_rows.append(
            _run_profile(
                SONATA_PROFILE,
                output_dir / "experimental_sonata_allegro",
                candidates=candidates,
                seed=seed + 100_000,
                skip_musescore=skip_musescore,
                experimental=True,
            )
        )
    _write_csv(output_dir / "automatic_results.csv", rows)
    _write_review_sheet(output_dir / "human_review.csv", rows)
    summary = _summarize(rows, piece_count, skip_musescore, experimental_rows)
    write_review_gallery(output_dir, rows, experimental_rows, summary)
    (output_dir / "release_gate.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def write_review_gallery(
    output_dir: Path | str,
    rows: Sequence[Dict[str, Any]],
    experimental_rows: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Path:
    """Write a compact index linking every complete-piece review artifact."""

    output_dir = Path(output_dir)
    cards = "\n".join(_gallery_card(row) for row in rows)
    experimental = "\n".join(_gallery_card(row) for row in experimental_rows)
    rates = summary["rates"]
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Score-First V1 Release Review</title>
  <style>
    :root {{ font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #2a2219; background: #f3eee4; }}
    body {{ margin: 0; }}
    header {{ padding: 30px max(22px, 5vw); background: #2c261f; color: #fffaf0; }}
    header h1 {{ margin: 0 0 8px; font-family: Georgia, serif; }}
    header p {{ margin: 5px 0; color: #ddd0bd; }}
    main {{ padding: 22px max(18px, 4vw) 48px; }}
    .summary, .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }}
    .pill, article {{ border: 1px solid #ded2c0; border-radius: 12px; background: #fffdf8; padding: 14px; }}
    article h3 {{ margin: 0 0 6px; }}
    article p {{ margin: 4px 0 10px; color: #75634b; font-size: 14px; }}
    a {{ display: inline-block; margin: 3px 8px 3px 0; color: #67471e; }}
    h2 {{ margin-top: 28px; }}
  </style>
</head>
<body>
  <header>
    <h1>Score-First V1 Release Review</h1>
    <p>{len(rows)} complete piano pieces for human review. Experimental sonata-allegro remains non-blocking.</p>
    <p>Open each Gallery for the section timeline and full paginated engraving.</p>
  </header>
  <main>
    <section class="summary">
      <div class="pill"><strong>Valid pieces</strong><br>{rates["valid_piece_rate"]:.0%}</div>
      <div class="pill"><strong>MuseScore render</strong><br>{rates["musescore_render_rate"]:.0%}</div>
      <div class="pill"><strong>Tempo overlay leaks</strong><br>{rates["tempo_overlay_leakage_count"]}</div>
      <div class="pill"><strong>Human review</strong><br><a href="human_review.csv">human_review.csv</a></div>
    </section>
    <h2>V1 Review Set</h2>
    <section class="grid">{cards}</section>
    <h2>Experimental Sonata-Allegro</h2>
    <section class="grid">{experimental or "<p>Not generated for this run.</p>"}</section>
  </main>
</body>
</html>
"""
    path = output_dir / "review_gallery.html"
    path.write_text(html, encoding="utf-8")
    return path


def _gallery_card(row: Dict[str, Any]) -> str:
    piece = escape(row["piece"])
    return f"""
      <article>
        <h3>{piece}</h3>
        <p>{escape(row["genre"])} | {escape(row["form"])} | {escape(row["key"])} | score {row["structural_score"]}</p>
        <a href="{piece}/gallery.html">Gallery</a>
        <a href="{piece}/score.pdf">PDF</a>
        <a href="{piece}/score.musicxml">MusicXML</a>
        <a href="{piece}/score.mid">score.mid</a>
        <a href="{piece}/performance.mid">performance.mid</a>
      </article>
    """


def _run_profile(
    profile: GateProfile,
    output_dir: Path,
    candidates: int,
    seed: int,
    skip_musescore: bool,
    experimental: bool,
) -> Dict[str, Any]:
    args = argparse.Namespace(
        prompt=profile.prompt,
        controls=None,
        output_dir=str(output_dir),
        genre=profile.genre,
        form=profile.form,
        duration_minutes=3.0,
        measure_range="48-192",
        key=profile.key,
        meter=profile.meter,
        tempo=profile.tempo,
        difficulty="intermediate",
        markings="dynamics,pedal,articulations",
        title=None,
        whole_piece=True,
        strategy="hierarchical",
        candidates=candidates,
        seed=seed,
        skip_musescore=skip_musescore,
        musescore_bin=None,
    )
    compose(args)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = manifest["metrics"]
    score = read_score(output_dir / "score.ir.json")
    xml_parse_ok = _xml_parse_ok(output_dir / "score.musicxml")
    render = manifest["render"]
    musescore_render_ok = bool(
        render.get("available")
        and (output_dir / "score.pdf").exists()
        and render.get("pages")
    )
    controls_match = (
        score.plan.genre == profile.genre
        and score.plan.form == profile.form
        and score.plan.key == profile.key
        and f"{score.plan.meter.beats}/{score.plan.meter.beat_type}" == profile.meter
    )
    return {
        "piece": output_dir.name,
        "experimental": experimental,
        "genre": profile.genre,
        "form": profile.form,
        "key": profile.key,
        "meter": profile.meter,
        "run_dir": str(output_dir.resolve()),
        "valid": metrics["valid"],
        "duration_error_ratio": metrics["duration_error_ratio"],
        "duration_within_10_percent": metrics["duration_error_ratio"] <= 0.10,
        "sections_cover_piece": metrics["sections_cover_piece"],
        "has_final_tonic": metrics["has_final_tonic"],
        "tempo_overlay_isolated": metrics["tempo_overlay_isolated"],
        "xml_parse_ok": xml_parse_ok,
        "musescore_render_ok": musescore_render_ok,
        "controls_match": controls_match,
        "structural_score": metrics["structural_score"],
    }


def _summarize(
    rows: Sequence[Dict[str, Any]],
    piece_count: int,
    skip_musescore: bool,
    experimental_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    def rate(field: str) -> float:
        return round(sum(bool(row[field]) for row in rows) / len(rows), 4)

    rates = {
        "valid_piece_rate": rate("valid"),
        "duration_within_10_percent_rate": rate("duration_within_10_percent"),
        "section_coverage_rate": rate("sections_cover_piece"),
        "planned_ending_rate": rate("has_final_tonic"),
        "musicxml_parse_rate": rate("xml_parse_ok"),
        "musescore_render_rate": rate("musescore_render_ok"),
        "control_accuracy_rate": rate("controls_match"),
        "tempo_overlay_leakage_count": sum(not row["tempo_overlay_isolated"] for row in rows),
    }
    thresholds = {
        "required_piece_count": 50,
        "valid_piece_rate_min": 1.0,
        "duration_within_10_percent_rate_min": 1.0,
        "section_coverage_rate_min": 1.0,
        "planned_ending_rate_min": 1.0,
        "musicxml_parse_rate_min": 0.98,
        "musescore_render_rate_min": 0.95,
        "control_accuracy_rate_min": 0.90,
        "tempo_overlay_leakage_count_max": 0,
    }
    checks = {
        "piece_count": piece_count >= thresholds["required_piece_count"],
        "valid_piece_rate": rates["valid_piece_rate"] >= thresholds["valid_piece_rate_min"],
        "duration_within_10_percent": rates["duration_within_10_percent_rate"]
        >= thresholds["duration_within_10_percent_rate_min"],
        "section_coverage": rates["section_coverage_rate"] >= thresholds["section_coverage_rate_min"],
        "planned_endings": rates["planned_ending_rate"] >= thresholds["planned_ending_rate_min"],
        "musicxml_parse_rate": rates["musicxml_parse_rate"] >= thresholds["musicxml_parse_rate_min"],
        "musescore_render_rate": rates["musescore_render_rate"] >= thresholds["musescore_render_rate_min"],
        "control_accuracy_rate": rates["control_accuracy_rate"] >= thresholds["control_accuracy_rate_min"],
        "tempo_overlay_isolation": rates["tempo_overlay_leakage_count"]
        <= thresholds["tempo_overlay_leakage_count_max"],
    }
    return {
        "pipeline": "score-first-v1-release-gate",
        "mode": "dry-run-without-musescore" if skip_musescore else "release",
        "generated_piece_count": len(rows),
        "rates": rates,
        "thresholds": thresholds,
        "checks": checks,
        "automated_gate_passed": all(checks.values()),
        "human_review": {
            "required_piece_count": 50,
            "worksheet": "human_review.csv",
            "status": "pending",
            "criteria": [
                "theme appears in planned positions",
                "variations match the blueprint",
                "section boundaries are not abrupt",
                "ending has an effective cadence",
                "engraving is readable and professional",
            ],
        },
        "release_ready": False,
        "release_ready_reason": "Complete human review worksheet before release.",
        "experimental_sonata_allegro": list(experimental_rows),
        "artifacts": {
            "automatic_results": "automatic_results.csv",
            "human_review": "human_review.csv",
            "review_gallery": "review_gallery.html",
        },
    }


def _xml_parse_ok(path: Path) -> bool:
    try:
        ET.parse(path)
    except ET.ParseError:
        return False
    return True


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_review_sheet(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = (
        "piece",
        "genre",
        "form",
        "run_dir",
        "theme_positions",
        "variation_quality",
        "section_boundaries",
        "ending_cadence",
        "engraving_quality",
        "approved",
        "comments",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete-piece score-first release gate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pieces", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2300)
    parser.add_argument("--skip-musescore", action="store_true", help="Dry-run only; cannot pass the release gate")
    parser.add_argument("--include-sonata", action="store_true", help="Report one non-blocking sonata-allegro sample")
    args = parser.parse_args()
    summary = run_release_gate(
        output_dir=args.output_dir,
        piece_count=args.pieces,
        candidates=args.candidates,
        seed=args.seed,
        skip_musescore=args.skip_musescore,
        include_sonata=args.include_sonata,
    )
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["automated_gate_passed"] else 1)


if __name__ == "__main__":
    main()
