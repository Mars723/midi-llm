"""Report honest native-adapter progress against upstream and whole-piece gates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
from typing import Any, Dict, List

from .import_midi import parse_midi
from .native_backbone import native_midi_stats


@dataclass
class ReviewInput:
    status: str
    notes: str


def build_native_progress_report(
    baseline_run: Path | str,
    adapter_run: Path | str,
    whole_piece_run: Path | str,
    *,
    human_review: ReviewInput,
) -> Dict[str, Any]:
    """Compare fixed-seed parity and whole-piece evidence without overstating quality."""

    baseline_run = Path(baseline_run)
    adapter_run = Path(adapter_run)
    whole_piece_run = Path(whole_piece_run)
    baseline = summarize_native_run(baseline_run)
    adapter = summarize_native_run(adapter_run)
    whole_manifest = _read_json(whole_piece_run / "manifest.json")
    draft_manifest = _read_json(whole_piece_run / "score_draft" / "manifest.json")
    structural_score = draft_manifest.get("metrics", {}).get("structural_score")
    human_passed = human_review.status == "accepted"
    promotion = (
        "eligible-for-human-promotion-review"
        if human_passed and adapter["density_drift_rate"] <= baseline["density_drift_rate"]
        else "reject-promotion"
    )
    return {
        "pipeline": "native-classical-training-progress-report-v1",
        "baseline_run": str(baseline_run),
        "adapter_run": str(adapter_run),
        "whole_piece_run": str(whole_piece_run),
        "fixed_seed_native_parity": {
            "baseline": baseline,
            "adapter": adapter,
            "interpretation": (
                "Parity replay is a regression gate. A lower drift rate and stable syntax are positive signals, "
                "but they do not prove that the music sounds better."
            ),
        },
        "whole_piece_structure": {
            "duration_seconds": whole_manifest["stats"]["duration_seconds"],
            "estimated_measures": whole_manifest["stats"]["estimated_measures"],
            "all_planned_sections_realized": whole_manifest["completion"]["all_planned_sections_realized"],
            "future_termination_condition_realized": whole_manifest["completion"]["future_termination_condition_realized"],
            "boundary_analysis": whole_manifest.get("boundary_analysis", []),
            "structural_score": structural_score,
            "notation_marking_source": whole_manifest.get("limitations", {}).get("generated_score_markings"),
            "performance_overlay": whole_manifest.get("performance_overlay", {}),
        },
        "human_musicality_review": {
            "status": human_review.status,
            "notes": human_review.notes,
            "required_for_promotion": True,
        },
        "estimated_improvement": {
            "native_syntax_and_density": _density_estimate(baseline, adapter),
            "complete_piece_delivery": (
                "demonstrated structurally"
                if whole_manifest["completion"]["all_planned_sections_realized"]
                else "not demonstrated"
            ),
            "musicality_over_upstream": "demonstrated" if human_passed else "not demonstrated",
            "promotion_decision": promotion,
        },
    }


def summarize_native_run(run_dir: Path | str) -> Dict[str, Any]:
    """Summarize every native candidate, including older manifests without cached stats."""

    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "manifest.json")
    candidates = []
    for candidate in manifest["candidates"]:
        midi = run_dir / candidate["native_midi"]
        stats = candidate.get("stats") or native_midi_stats(parse_midi(midi))
        normalization = candidate.get("normalization", {})
        max_tokens = manifest.get("config", {}).get("max_tokens")
        accepted = normalization.get("accepted_event_tokens")
        hit_budget = stats.get("hit_max_token_budget")
        if hit_budget is None:
            hit_budget = bool(max_tokens and accepted and accepted >= max_tokens and normalization.get("stop_model_token_id") is None)
        candidates.append(
            {
                "candidate": candidate["candidate"],
                "seed": candidate.get("seed"),
                "native_midi": candidate["native_midi"],
                **stats,
                "hit_max_token_budget": hit_budget,
                "natural_stop_detected": stats.get("natural_stop_detected", not hit_budget),
            }
        )
    return {
        "candidate_count": len(candidates),
        "syntax_valid_rate": round(len(candidates) / max(1, len(candidates) + len(manifest.get("failures", []))), 4),
        "density_drift_rate": round(sum(bool(row["potential_density_drift"]) for row in candidates) / max(1, len(candidates)), 4),
        "natural_stop_rate": round(sum(bool(row["natural_stop_detected"]) for row in candidates) / max(1, len(candidates)), 4),
        "median_duration_seconds": _median(candidates, "duration_seconds"),
        "median_distinct_pitches": _median(candidates, "distinct_pitches"),
        "median_notes_per_second": _median(candidates, "notes_per_second"),
        "candidates": candidates,
    }


def write_progress_report(report: Dict[str, Any], output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "progress-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = _markdown_report(report)
    (output_dir / "progress-report.md").write_text(markdown, encoding="utf-8")
    return output_dir


def _density_estimate(baseline: Dict[str, Any], adapter: Dict[str, Any]) -> str:
    if adapter["syntax_valid_rate"] < baseline["syntax_valid_rate"]:
        return "regressed"
    if adapter["density_drift_rate"] < baseline["density_drift_rate"]:
        return "improved"
    if adapter["density_drift_rate"] == baseline["density_drift_rate"]:
        return "no measured change"
    return "regressed"


def _markdown_report(report: Dict[str, Any]) -> str:
    parity = report["fixed_seed_native_parity"]
    baseline = parity["baseline"]
    adapter = parity["adapter"]
    structure = report["whole_piece_structure"]
    estimate = report["estimated_improvement"]
    review = report["human_musicality_review"]
    return f"""# Native Classical Training Progress

## Honest Estimate

- Native syntax and density: **{estimate["native_syntax_and_density"]}**
- Complete-piece delivery: **{estimate["complete_piece_delivery"]}**
- Musicality over upstream: **{estimate["musicality_over_upstream"]}**
- Promotion: **{estimate["promotion_decision"]}**

## Fixed-Seed Native Parity

| Metric | Upstream | Adapter |
| --- | ---: | ---: |
| Syntax-valid rate | {baseline["syntax_valid_rate"]:.1%} | {adapter["syntax_valid_rate"]:.1%} |
| Density-drift rate | {baseline["density_drift_rate"]:.1%} | {adapter["density_drift_rate"]:.1%} |
| Natural-stop rate | {baseline["natural_stop_rate"]:.1%} | {adapter["natural_stop_rate"]:.1%} |
| Median duration | {baseline["median_duration_seconds"]:.1f}s | {adapter["median_duration_seconds"]:.1f}s |
| Median distinct pitches | {baseline["median_distinct_pitches"]:.1f} | {adapter["median_distinct_pitches"]:.1f} |

## Whole-Piece Structure

- Duration: {structure["duration_seconds"]:.1f}s
- Measures: {structure["estimated_measures"]}
- Planned sections realized: {structure["all_planned_sections_realized"]}
- Ending target realized: {structure["future_termination_condition_realized"]}
- Structural score: {structure["structural_score"]}

## Human Review

- Status: **{review["status"]}**
- Notes: {review["notes"]}

Automatic structure metrics are not a substitute for listening review.
"""


def _median(rows: List[Dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return round(statistics.median(values), 3) if values else 0.0


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Report native adapter progress honestly against upstream")
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--adapter-run", required=True)
    parser.add_argument("--whole-piece-run", required=True)
    parser.add_argument("--human-review-status", choices=("accepted", "rejected", "pending"), default="pending")
    parser.add_argument("--human-review-notes", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = build_native_progress_report(
        args.baseline_run,
        args.adapter_run,
        args.whole_piece_run,
        human_review=ReviewInput(status=args.human_review_status, notes=args.human_review_notes),
    )
    output = write_progress_report(report, args.output_dir)
    print((output / "progress-report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
