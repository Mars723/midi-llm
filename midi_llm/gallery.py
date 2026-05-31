"""Static local gallery for inspecting generated scores and performance layers."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any, Dict

from .score_ir import PianoScoreIR


def write_gallery(score: PianoScoreIR, manifest: Dict[str, Any], output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    pages = [Path(path).name for path in manifest.get("render", {}).get("pages", [])]
    sections = "\n".join(
        f"""
        <div class="section">
          <strong>{escape(section.label)}</strong>
          <span>m. {section.start_measure}-{section.end_measure}</span>
          <small>{escape(section.role)} | {escape(section.key)} | motifs: {escape(", ".join(section.motif_refs))}</small>
        </div>
        """
        for section in score.plan.sections
    )
    motif_cards = "\n".join(
        f"""
        <li><strong>{escape(motif.id)}</strong>: {escape(motif.description)}
        <small>{len(motif.events)} events</small></li>
        """
        for motif in score.motif_bank.motifs
    )
    page_cards = "\n".join(
        f'<figure><img src="{escape(page)}" alt="Score page {index}"><figcaption>Page {index}</figcaption></figure>'
        for index, page in enumerate(pages, start=1)
    )
    metrics = manifest.get("metrics", {})
    links = _artifact_links(output_dir)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(score.plan.title)} | MIDI-LLM Gallery</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin: 0; background: #f4efe5; color: #252018; }}
    header {{ padding: 32px max(24px, 6vw); background: #26221d; color: #fffaf0; }}
    h1 {{ margin: 0 0 8px; font-family: Georgia, serif; font-size: clamp(30px, 5vw, 54px); }}
    header p {{ max-width: 900px; color: #d8cdbb; }}
    main {{ display: grid; gap: 24px; padding: 24px max(20px, 5vw) 48px; }}
    .panel {{ background: #fffdf8; border: 1px solid #ded4c4; border-radius: 14px; padding: 18px; box-shadow: 0 8px 22px #5d4b2f18; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .pill {{ background: #ece3d4; border-radius: 999px; padding: 7px 11px; font-size: 14px; }}
    .timeline {{ display: flex; gap: 8px; overflow-x: auto; padding-bottom: 8px; }}
    .section {{ min-width: 165px; border-left: 5px solid #82653b; background: #f7f0e5; padding: 10px; }}
    .section span, .section small {{ display: block; margin-top: 5px; }}
    .section small, li small {{ color: #6c5b43; }}
    nav a {{ display: inline-block; margin: 4px 10px 4px 0; color: #5d411d; }}
    figure {{ margin: 16px auto; max-width: 1100px; }}
    img {{ display: block; width: 100%; background: white; box-shadow: 0 5px 16px #43321822; }}
    figcaption {{ padding-top: 6px; color: #6c5b43; }}
    code {{ color: #5d411d; }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(score.plan.title)}</h1>
    <p>{escape(score.plan.prompt)}</p>
  </header>
  <main>
    <section class="panel">
      <h2>Whole-Piece Summary</h2>
      <div class="summary">
        <span class="pill">{escape(score.plan.genre)}</span>
        <span class="pill">{escape(score.plan.form)}</span>
        <span class="pill">{score.plan.measure_count} measures</span>
        <span class="pill">{score.plan.duration_minutes:.1f} requested min</span>
        <span class="pill">{escape(score.plan.key)}</span>
        <span class="pill">{score.plan.meter.beats}/{score.plan.meter.beat_type}</span>
        <span class="pill">structural score {metrics.get("structural_score", "n/a")}</span>
        <span class="pill">max repeated-measure run {metrics.get("longest_identical_measure_run", "n/a")}</span>
        <span class="pill">periodic loop span {metrics.get("periodic_measure_loop_span", "n/a")}</span>
      </div>
    </section>
    <section class="panel">
      <h2>Section Timeline</h2>
      <div class="timeline">{sections}</div>
    </section>
    <section class="panel">
      <h2>Motif Bank</h2>
      <ul>{motif_cards}</ul>
    </section>
    <section class="panel">
      <h2>Artifacts</h2>
      <nav>{links}</nav>
      <p><code>score.mid</code> stays quantized for notation. <code>performance.mid</code> contains the expressive overlay.</p>
    </section>
    <section class="panel">
      <h2>Rendered Score</h2>
      {page_cards or "<p>MuseScore render was skipped or unavailable.</p>"}
    </section>
  </main>
</body>
</html>
"""
    gallery = output_dir / "gallery.html"
    gallery.write_text(html, encoding="utf-8")
    return gallery


def _artifact_links(output_dir: Path) -> str:
    names = (
        "score.pdf",
        "score.musicxml",
        "score.dsl",
        "score.mscz",
        "score.mid",
        "performance.mid",
        "score.ir.json",
        "performance.ir.json",
        "manifest.json",
    )
    return "\n".join(
        f'<a href="{name}">{name}</a>' for name in names if (output_dir / name).exists()
    )
