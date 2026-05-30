"""CLI entrypoint for complete score-first piano-piece generation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict

from .compiler import render_musescore, write_musicxml, write_performance_midi, write_score_midi
from .evaluate import evaluate_score
from .gallery import write_gallery
from .planner import controls_from_mapping, create_piece_plan, read_controls
from .rules import create_motif_bank, generate_score_candidate, render_performance
from .scoredsl import encode_score
from .score_ir import write_json


def compose(args: argparse.Namespace) -> Path:
    mapping = read_controls(args.controls)
    for key in (
        "genre",
        "form",
        "duration_minutes",
        "measure_range",
        "key",
        "meter",
        "tempo",
        "difficulty",
        "markings",
        "title",
    ):
        value = getattr(args, key, None)
        if value is not None:
            mapping[key] = value
    controls = controls_from_mapping(mapping)
    plan = create_piece_plan(args.prompt, controls)
    motif_bank = create_motif_bank(plan.key, plan.genre)
    output_dir = Path(args.output_dir or _default_output_dir())
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for index in range(args.candidates):
        seed = args.seed + index
        score = generate_score_candidate(plan, motif_bank, seed=seed)
        performance = render_performance(score, seed=seed + 10_000)
        metrics = evaluate_score(score, performance)
        write_json(candidate_dir / f"candidate_{index + 1}.score.ir.json", score)
        write_json(candidate_dir / f"candidate_{index + 1}.performance.ir.json", performance)
        (candidate_dir / f"candidate_{index + 1}.metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        candidates.append((metrics["structural_score"], index, score, performance, metrics))

    _, best_index, score, performance, metrics = max(candidates, key=lambda item: (item[0], -item[1]))
    write_json(output_dir / "score.ir.json", score)
    write_json(output_dir / "performance.ir.json", performance)
    (output_dir / "score.dsl").write_text(encode_score(score), encoding="utf-8")
    write_musicxml(score, output_dir / "score.musicxml")
    write_score_midi(score, output_dir / "score.mid")
    write_performance_midi(score, performance, output_dir / "performance.mid")
    render = {"available": False, "reason": "MuseScore rendering skipped", "pages": []}
    if not args.skip_musescore:
        render = render_musescore(output_dir / "score.musicxml", output_dir, args.musescore_bin)
    metrics = evaluate_score(score, performance, output_dir / "score.musicxml")
    manifest: Dict[str, Any] = {
        "pipeline": "score-first-whole-piece-v1",
        "strategy": args.strategy,
        "selected_candidate": best_index + 1,
        "candidate_count": args.candidates,
        "prompt": args.prompt,
        "controls": asdict(controls),
        "metrics": metrics,
        "render": render,
        "artifacts": {
            "score_ir": "score.ir.json",
            "performance_ir": "performance.ir.json",
            "musicxml": "score.musicxml",
            "score_dsl": "score.dsl",
            "score_midi": "score.mid",
            "performance_midi": "performance.mid",
            "gallery": "gallery.html",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "prompt.txt").write_text(args.prompt + "\n", encoding="utf-8")
    write_gallery(score, manifest, output_dir)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compose a complete score-first classical piano piece")
    parser.add_argument("--prompt", required=True, help="English natural-language composition prompt")
    parser.add_argument("--controls", help="YAML-like or JSON controls file")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--genre")
    parser.add_argument("--form")
    parser.add_argument("--duration-minutes", type=float)
    parser.add_argument("--measure-range", help="Minimum and maximum measures, such as 48-192")
    parser.add_argument("--key")
    parser.add_argument("--meter")
    parser.add_argument("--tempo", type=int)
    parser.add_argument("--difficulty")
    parser.add_argument("--markings", help="Comma-separated markings, such as dynamics,pedal,fingering")
    parser.add_argument("--title")
    parser.add_argument("--whole-piece", action="store_true", default=True, help="Generate a complete planned piece")
    parser.add_argument("--strategy", choices=("hierarchical", "autoregressive"), default="hierarchical")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--skip-musescore", action="store_true")
    parser.add_argument("--musescore-bin")
    return parser


def _default_output_dir() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return str(Path("generated_score_first") / timestamp)


def main() -> None:
    args = build_parser().parse_args()
    if args.candidates < 1:
        raise SystemExit("--candidates must be at least 1")
    output_dir = compose(args)
    print(f"Complete piece written to {output_dir.resolve()}")
    print(f"Gallery: {(output_dir / 'gallery.html').resolve()}")


if __name__ == "__main__":
    main()
