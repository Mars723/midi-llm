"""Derive compact notation controls from score-first piano IR."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .score_ir import NoteEvent, PianoScoreIR, PiecePlanIR


def analyze_notation(score: PianoScoreIR) -> Dict[str, Any]:
    """Summarize texture, markings, and measure-level variation for training controls."""

    signatures = _measure_signatures(score.notes, score.plan.measure_count)
    densities = [_measure_note_count(signature) for signature in signatures]
    unique_ratio = len(set(signatures)) / max(1, len(signatures))
    longest_run = _longest_identical_run(signatures)
    pitch_span = _pitch_span(score.notes)
    voice_count = len({(note.staff, note.voice) for note in score.notes})
    section_profiles = [
        _section_profile(score.notes, section.label, section.role, section.start_measure, section.end_measure)
        for section in score.plan.sections
    ]
    density_variation = _normalized_density_variation(densities)
    section_contrast = _section_contrast(section_profiles)
    marking_score = min(1.0, len(score.plan.markings) / 4.0)
    variation_score = round(
        min(
            1.0,
            0.45 * unique_ratio
            + 0.20 * density_variation
            + 0.15 * min(1.0, pitch_span / 36.0)
            + 0.10 * section_contrast
            + 0.10 * marking_score,
        ),
        4,
    )
    return {
        "texture": infer_texture(score.notes),
        "note_count": len(score.notes),
        "notes_per_measure": round(mean(densities), 3) if densities else 0.0,
        "voice_count": voice_count,
        "pitch_span": pitch_span,
        "marking_families": sorted(score.plan.markings),
        "unique_measure_signatures": len(set(signatures)),
        "unique_measure_signature_ratio": round(unique_ratio, 4),
        "longest_identical_measure_run": longest_run,
        "identical_measure_run_ratio": round(longest_run / max(1, len(signatures)), 4),
        "density_variation": round(density_variation, 4),
        "section_contrast": round(section_contrast, 4),
        "variation_score": variation_score,
        "variation_tier": _variation_tier(variation_score),
        "section_profiles": section_profiles,
    }


def notation_constraints_from_profile(profile: Dict[str, Any], difficulty: str) -> Dict[str, Any]:
    """Expose source-derived controls without leaking individual notes."""

    return {
        "difficulty": difficulty,
        "texture": profile["texture"],
        "preferred_notes_per_measure": profile["notes_per_measure"],
        "preferred_voice_count": profile["voice_count"],
        "preferred_pitch_span": profile["pitch_span"],
        "required_marking_families": profile["marking_families"],
        "minimum_unique_measure_signature_ratio": profile["unique_measure_signature_ratio"],
        "maximum_identical_measure_run": profile["longest_identical_measure_run"],
        "require_section_contrast": True,
        "section_goals": profile["section_profiles"],
    }


def intermediate_notation_constraints(plan: PiecePlanIR) -> Dict[str, Any]:
    """Return explicit inference controls for a reviewable intermediate piano score."""

    return {
        "difficulty": plan.difficulty,
        "texture": plan.texture,
        "preferred_notes_per_measure_range": [8, 20],
        "preferred_voice_count_range": [2, 4],
        "preferred_pitch_span_range": [24, 60],
        "required_marking_families": sorted(plan.markings),
        "minimum_unique_measure_signature_ratio": 0.45,
        "maximum_identical_measure_run": 8,
        "require_section_contrast": True,
        "section_goals": [
            {
                "label": section.label,
                "role": section.role,
                "variation_goal": {
                    "statement": "establish-theme",
                    "contrast": "introduce-contrasting-material",
                    "develop": "develop-and-transform",
                    "return": "return-with-variation",
                    "variation": "transform-theme",
                    "intro": "prepare-theme",
                    "coda": "resolve-and-close",
                }.get(section.role, "maintain-coherence"),
            }
            for section in plan.sections
        ],
    }


def infer_texture(notes: Sequence[NoteEvent]) -> str:
    """Classify a compact piano texture label from staff and onset statistics."""

    if not notes:
        return "empty"
    staves = {note.staff for note in notes}
    voices = {(note.staff, note.voice) for note in notes}
    onset_groups: Dict[Tuple[int, float, int], int] = defaultdict(int)
    for note in notes:
        onset_groups[(note.measure, note.beat, note.staff)] += 1
    chord_ratio = sum(count > 1 for count in onset_groups.values()) / max(1, len(onset_groups))
    if len(voices) >= 4:
        return "multi-voice-piano"
    if staves == {1, 2} and chord_ratio >= 0.30:
        return "melody-with-chordal-accompaniment"
    if staves == {1, 2}:
        return "melody-with-accompaniment"
    if chord_ratio >= 0.30:
        return "single-staff-chordal"
    return "single-staff-melodic"


def _measure_signatures(notes: Sequence[NoteEvent], measure_count: int) -> List[Tuple[Tuple[Any, ...], ...]]:
    by_measure: Dict[int, List[Tuple[Any, ...]]] = defaultdict(list)
    for note in notes:
        by_measure[note.measure].append(
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
    return [tuple(by_measure[measure]) for measure in range(1, measure_count + 1)]


def _measure_note_count(signature: Sequence[Tuple[Any, ...]]) -> int:
    return len(signature)


def _longest_identical_run(signatures: Iterable[Tuple[Tuple[Any, ...], ...]]) -> int:
    longest = 0
    current = 0
    prior = None
    for signature in signatures:
        current = current + 1 if signature == prior else 1
        prior = signature
        longest = max(longest, current)
    return longest


def _pitch_span(notes: Sequence[NoteEvent]) -> int:
    pitches = [note.pitch for note in notes]
    return max(pitches) - min(pitches) if pitches else 0


def _normalized_density_variation(densities: Sequence[int]) -> float:
    if not densities:
        return 0.0
    average = mean(densities)
    if not average:
        return 0.0
    return min(1.0, pstdev(densities) / average)


def _section_profile(
    notes: Sequence[NoteEvent],
    label: str,
    role: str,
    start: int,
    end: int,
) -> Dict[str, Any]:
    section_notes = [note for note in notes if start <= note.measure <= end]
    signatures = _measure_signatures(section_notes, end)[start - 1 : end]
    densities = [_measure_note_count(signature) for signature in signatures]
    pitches = [note.pitch for note in section_notes]
    return {
        "label": label,
        "role": role,
        "range": [start, end],
        "notes_per_measure": round(mean(densities), 3) if densities else 0.0,
        "unique_measure_signature_ratio": round(len(set(signatures)) / max(1, len(signatures)), 4),
        "pitch_center": round(mean(pitches), 2) if pitches else None,
        "texture": infer_texture(section_notes),
    }


def _section_contrast(sections: Sequence[Dict[str, Any]]) -> float:
    if len(sections) < 2:
        return 0.0
    values = [
        (
            section["notes_per_measure"],
            section["pitch_center"] if section["pitch_center"] is not None else 0.0,
        )
        for section in sections
    ]
    density_span = max(value[0] for value in values) - min(value[0] for value in values)
    pitch_span = max(value[1] for value in values) - min(value[1] for value in values)
    return min(1.0, density_span / 8.0 + pitch_span / 24.0)


def _variation_tier(score: float) -> str:
    if score >= 0.60:
        return "high"
    if score >= 0.40:
        return "moderate"
    return "low"
