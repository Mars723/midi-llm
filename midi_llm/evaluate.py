"""Evaluation helpers and CLI for score-first generation runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Optional
import xml.etree.ElementTree as ET

from .notation_analysis import requires_two_staff_texture, staff_measure_coverage
from .score_ir import PianoPerformanceIR, PianoScoreIR, read_performance, read_score, validate_score


def evaluate_score(
    score: PianoScoreIR,
    performance: PianoPerformanceIR,
    musicxml_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    errors = validate_score(score)
    final_notes = [
        note.pitch % 12
        for note in score.notes
        if note.measure == score.plan.measure_count and note.staff == 1
    ]
    tonic = _tonic_pitch_class(score.plan.key)
    has_final_tonic = tonic in final_notes
    expected_refs = {ref for section in score.plan.sections for ref in section.motif_refs}
    available_refs = {motif.id for motif in score.motif_bank.motifs}
    missing_refs = sorted(expected_refs - available_refs)
    score_tempo_count = len(score.plan.tempo_marks)
    xml_tempo_count = None
    if musicxml_path and Path(musicxml_path).exists():
        root = ET.parse(musicxml_path).getroot()
        xml_tempo_count = len(root.findall(".//sound[@tempo]"))
    tempo_overlay_isolated = xml_tempo_count in (None, score_tempo_count)
    duration_minutes = (
        score.plan.measure_count * score.plan.meter.quarter_beats / score.plan.tempo_bpm
    )
    duration_error = abs(duration_minutes - score.plan.duration_minutes) / score.plan.duration_minutes
    repetition = _measure_repetition_metrics(score)
    repetition_penalty = min(
        30,
        max(0, repetition["longest_identical_measure_run"] - 8)
        + min(15, max(0, round((0.35 - repetition["unique_measure_signature_ratio"]) * 50)))
        + min(15, max(0, round((repetition["periodic_measure_loop_ratio"] - 0.50) * 30))),
    )
    periodic_loop_acceptable = (
        repetition["periodic_measure_loop_span"] < 16
        or repetition["periodic_measure_loop_ratio"] <= 0.65
    )
    measure_diversity_acceptable = (
        repetition["unique_measure_signature_ratio"] >= 0.30
        and repetition["longest_identical_measure_run"] <= 8
    )
    staff_coverage = staff_measure_coverage(score.notes, score.plan.measure_count)
    two_staff_texture_acceptable = (
        not requires_two_staff_texture(score.plan.texture)
        or min(staff_coverage.values()) >= 0.70
    )
    structural_score = 100.0
    structural_score -= len(errors) * 20
    structural_score -= len(missing_refs) * 10
    structural_score -= 15 if not has_final_tonic else 0
    structural_score -= 15 if not tempo_overlay_isolated else 0
    structural_score -= min(15, duration_error * 100)
    structural_score -= repetition_penalty
    structural_score -= 20 if not two_staff_texture_acceptable else 0
    return {
        "valid": (
            not errors
            and has_final_tonic
            and not missing_refs
            and tempo_overlay_isolated
            and periodic_loop_acceptable
            and measure_diversity_acceptable
            and two_staff_texture_acceptable
        ),
        "validation_errors": errors,
        "measure_count": score.plan.measure_count,
        "estimated_duration_minutes": round(duration_minutes, 3),
        "requested_duration_minutes": score.plan.duration_minutes,
        "duration_error_ratio": round(duration_error, 4),
        "section_count": len(score.plan.sections),
        "sections_cover_piece": not any("Sections do not cover" in error for error in errors),
        "missing_motif_refs": missing_refs,
        "has_final_tonic": has_final_tonic,
        "score_structural_tempo_marks": score_tempo_count,
        "performance_tempo_curve_points": len(performance.tempo_curve),
        "musicxml_tempo_marks": xml_tempo_count,
        "tempo_overlay_isolated": tempo_overlay_isolated,
        "periodic_loop_acceptable": periodic_loop_acceptable,
        "measure_diversity_acceptable": measure_diversity_acceptable,
        "upper_staff_measure_coverage": staff_coverage[1],
        "lower_staff_measure_coverage": staff_coverage[2],
        "two_staff_texture_acceptable": two_staff_texture_acceptable,
        **repetition,
        "repetition_score_penalty": repetition_penalty,
        "structural_score": round(max(0.0, structural_score), 2),
    }


def _measure_repetition_metrics(score: PianoScoreIR) -> Dict[str, Any]:
    notes_by_measure = defaultdict(list)
    for note in score.notes:
        notes_by_measure[note.measure].append(
            (
                note.beat,
                note.duration,
                note.pitch,
                note.staff,
                note.voice,
                note.articulation,
                note.fingering,
                note.tie_start,
                note.tie_stop,
            )
        )
    longest_run = 0
    current_run = 0
    prior_signature = None
    for measure in sorted(notes_by_measure):
        signature = tuple(notes_by_measure[measure])
        current_run = current_run + 1 if signature == prior_signature else 1
        prior_signature = signature
        longest_run = max(longest_run, current_run)
    signatures = [tuple(notes_by_measure[measure]) for measure in sorted(notes_by_measure)]
    periodic_measure_loop_period, periodic_measure_loop_span = _longest_periodic_measure_run(signatures)
    return {
        "unique_measure_signatures": len(set(signatures)),
        "unique_measure_signature_ratio": round(len(set(signatures)) / len(signatures), 4) if signatures else 0.0,
        "longest_identical_measure_run": longest_run,
        "identical_measure_run_ratio": round(longest_run / len(notes_by_measure), 4) if notes_by_measure else 0.0,
        "periodic_measure_loop_period": periodic_measure_loop_period,
        "periodic_measure_loop_span": periodic_measure_loop_span,
        "periodic_measure_loop_ratio": round(periodic_measure_loop_span / len(signatures), 4) if signatures else 0.0,
    }


def _longest_periodic_measure_run(signatures, max_period: int = 8):
    best_period = None
    best_span = 0
    for period in range(1, min(max_period, len(signatures) // 2) + 1):
        current_span = period
        for index in range(period, len(signatures)):
            if signatures[index] == signatures[index - period]:
                current_span += 1
                if current_span >= period * 2 and current_span > best_span:
                    best_period = period
                    best_span = current_span
            else:
                current_span = period
    return best_period, best_span


def evaluate_run(run_dir: Path | str) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    score = read_score(run_dir / "score.ir.json")
    performance = read_performance(run_dir / "performance.ir.json")
    return evaluate_score(score, performance, run_dir / "score.musicxml")


def _tonic_pitch_class(key: str) -> int:
    return {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "Ab": 8,
        "A": 9,
        "Bb": 10,
        "B": 11,
    }.get(key.split()[0], 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a score-first generation run")
    parser.add_argument("--run", required=True, help="Generation output directory")
    parser.add_argument("--output", help="Optional JSON metrics output")
    args = parser.parse_args()
    metrics = evaluate_run(args.run)
    payload = json.dumps(metrics, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    raise SystemExit(0 if metrics["valid"] else 1)


if __name__ == "__main__":
    main()
