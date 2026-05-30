"""Deterministic v1 generators used before trained score models are available."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from .score_ir import (
    ExpressiveTempoEvent,
    LayoutHint,
    Motif,
    MotifBank,
    MotifEvent,
    NoteEvent,
    PedalEvent,
    PerformanceNote,
    PianoPerformanceIR,
    PianoScoreIR,
    ScoreDirection,
    section_for_measure,
)


KEY_TO_MIDI = {
    "C": 60,
    "C#": 61,
    "Db": 61,
    "D": 62,
    "Eb": 63,
    "E": 64,
    "F": 65,
    "F#": 66,
    "Gb": 66,
    "G": 67,
    "Ab": 68,
    "A": 69,
    "Bb": 70,
    "B": 71,
}


def create_motif_bank(key: str, genre: str) -> MotifBank:
    tonic = KEY_TO_MIDI.get(key.split()[0], 60)
    theme_a = (0, 2, 4, 7, 5, 4, 2, 0)
    theme_b = (4, 5, 7, 9, 7, 5, 4, 2)
    if genre == "etude":
        theme_a = (0, 2, 4, 7, 9, 7, 4, 2)
    elif genre == "waltz":
        theme_a = (0, 4, 7, 9, 7, 4)
    motifs = [
        Motif(
            id="theme-a",
            kind="melody",
            events=_motif_events(theme_a, tonic + 12),
            description="Primary lyrical theme",
        ),
        Motif(
            id="theme-b",
            kind="melody",
            events=_motif_events(theme_b, tonic + 12),
            description="Contrasting theme",
        ),
        Motif(
            id="accompaniment",
            kind="accompaniment",
            events=[
                MotifEvent(offset=0.0, duration=0.5, pitch=tonic - 12),
                MotifEvent(offset=0.5, duration=0.5, pitch=tonic - 5),
                MotifEvent(offset=1.0, duration=0.5, pitch=tonic),
                MotifEvent(offset=1.5, duration=0.5, pitch=tonic - 5),
            ],
            description="Broken-chord left-hand accompaniment",
        ),
    ]
    return MotifBank(motifs=motifs)


def _motif_events(intervals: Tuple[int, ...], base: int) -> List[MotifEvent]:
    duration = 0.5 if len(intervals) >= 8 else 1.0
    return [
        MotifEvent(offset=index * duration, duration=duration, pitch=base + interval)
        for index, interval in enumerate(intervals)
    ]


def generate_score_candidate(plan, motif_bank: MotifBank, seed: int) -> PianoScoreIR:
    rng = random.Random(seed)
    notes: List[NoteEvent] = []
    directions: List[ScoreDirection] = []
    layout_hints: List[LayoutHint] = []
    note_index = 0
    measure_beats = plan.meter.quarter_beats
    root = KEY_TO_MIDI.get(plan.key.split()[0], 60)

    for tempo_mark in plan.tempo_marks:
        directions.append(
            ScoreDirection(
                measure=tempo_mark.measure,
                beat=0.0,
                kind="tempo",
                value=f"{tempo_mark.bpm}|{tempo_mark.text}",
            )
        )
    dynamics = ("p", "mp", "mf", "f")
    for section_index, section in enumerate(plan.sections):
        directions.append(
            ScoreDirection(
                measure=section.start_measure,
                beat=0.0,
                kind="dynamic",
                value=dynamics[min(section_index, len(dynamics) - 1)],
                staff=1,
            )
        )
        if section_index and section.role in ("develop", "contrast"):
            directions.append(
                ScoreDirection(
                    measure=section.start_measure,
                    beat=0.0,
                    kind="wedge-start",
                    value="crescendo",
                    end_measure=min(section.end_measure, section.start_measure + 3),
                )
            )
            directions.append(
                ScoreDirection(
                    measure=min(section.end_measure, section.start_measure + 3),
                    beat=0.0,
                    kind="wedge-stop",
                    value="stop",
                )
            )
        if "pedal" in plan.markings:
            directions.append(
                ScoreDirection(
                    measure=section.start_measure,
                    beat=0.0,
                    kind="pedal-start",
                    value="start",
                    staff=2,
                )
            )
            directions.append(
                ScoreDirection(
                    measure=min(section.end_measure, section.start_measure + 1),
                    beat=max(0.0, plan.meter.quarter_beats - 0.25),
                    kind="pedal-stop",
                    value="stop",
                    staff=2,
                )
            )
        if section.start_measure > 1:
            layout_hints.append(LayoutHint(measure=section.start_measure, kind="system-break"))

    for measure in range(1, plan.measure_count + 1):
        section = section_for_measure(plan.sections, measure)
        local_measure = measure - section.start_measure
        transpose = _section_transpose(section.role, local_measure)
        melody_id = "theme-b" if section.role in ("contrast", "develop") else "theme-a"
        melody = motif_bank.by_id(melody_id)
        variation = _variation_shift(section.role, local_measure, rng)

        for event in _fit_motif_to_measure(melody.events, measure_beats):
            note_index += 1
            notes.append(
                NoteEvent(
                    id=f"n{note_index}",
                    measure=measure,
                    beat=event.offset,
                    duration=event.duration,
                    pitch=_clamp(event.pitch + transpose + variation, 60, 88),
                    staff=1,
                    articulation="staccato" if plan.genre == "etude" and local_measure % 4 == 1 else None,
                    fingering=str(1 + (note_index % 5))
                    if "fingering" in plan.markings and local_measure == 0
                    else None,
                )
            )
        accompaniment = _left_hand_pattern(plan.genre, root + transpose, measure_beats)
        for beat, duration, pitch in accompaniment:
            note_index += 1
            notes.append(
                NoteEvent(
                    id=f"n{note_index}",
                    measure=measure,
                    beat=beat,
                    duration=duration,
                    pitch=_clamp(pitch, 28, 60),
                    staff=2,
                )
            )
    _apply_final_cadence(notes, plan.measure_count, measure_beats, root)
    return PianoScoreIR(
        plan=plan,
        motif_bank=motif_bank,
        notes=notes,
        directions=directions,
        layout_hints=layout_hints,
        metadata={"generator": "whole-piece-rules-v1", "seed": seed},
    )


def render_performance(score: PianoScoreIR, seed: int) -> PianoPerformanceIR:
    rng = random.Random(seed)
    dynamic_by_measure = _dynamic_map(score)
    performance_notes: List[PerformanceNote] = []
    for note in score.notes:
        dynamic = dynamic_by_measure.get(note.measure, "mp")
        center = {"pp": 42, "p": 52, "mp": 62, "mf": 74, "f": 88, "ff": 102}.get(dynamic, 68)
        phrase = 5.0 * math.sin((note.measure % 8) / 8.0 * math.pi)
        articulation_ratio = 0.72 if note.articulation == "staccato" else 0.96
        performance_notes.append(
            PerformanceNote(
                event_id=note.id,
                velocity=_clamp(round(center + phrase + rng.uniform(-5, 5)), 20, 120),
                onset_shift_beats=round(rng.uniform(-0.025, 0.025), 4),
                duration_ratio=round(articulation_ratio + rng.uniform(-0.03, 0.03), 4),
            )
        )
    pedal: List[PedalEvent] = []
    beats = score.plan.meter.quarter_beats
    if "pedal" in score.plan.markings:
        for measure in range(1, score.plan.measure_count + 1):
            pedal.append(PedalEvent(measure=measure, beat=0.10, value=72))
            pedal.append(PedalEvent(measure=measure, beat=max(0.15, beats - 0.10), value=0))
    tempo_curve: List[ExpressiveTempoEvent] = []
    for measure in range(1, score.plan.measure_count + 1):
        section = section_for_measure(score.plan.sections, measure)
        local = measure - section.start_measure
        bpm = score.plan.tempo_bpm * (1.0 + 0.025 * math.sin(local * math.pi / 4.0))
        if measure > score.plan.measure_count - 4:
            progress = (measure - (score.plan.measure_count - 4)) / 4.0
            bpm *= 1.0 - 0.16 * progress
        tempo_curve.append(ExpressiveTempoEvent(measure=measure, beat=0.0, bpm=round(bpm, 2)))
    return PianoPerformanceIR(notes=performance_notes, pedal=pedal, tempo_curve=tempo_curve)


def _fit_motif_to_measure(events: List[MotifEvent], measure_beats: float) -> List[MotifEvent]:
    fitted: List[MotifEvent] = []
    cursor = 0.0
    while cursor < measure_beats:
        for event in events:
            offset = cursor + event.offset
            if offset >= measure_beats:
                break
            fitted.append(
                MotifEvent(
                    offset=offset,
                    duration=min(event.duration, measure_beats - offset),
                    pitch=event.pitch,
                )
            )
        motif_length = max(event.offset + event.duration for event in events)
        cursor += motif_length
    return fitted


def _left_hand_pattern(genre: str, root: int, measure_beats: float) -> List[Tuple[float, float, int]]:
    if genre == "waltz" and measure_beats >= 3:
        pattern = [(0.0, 1.0, root - 12), (1.0, 1.0, root - 5), (2.0, 1.0, root)]
    else:
        pattern = []
        pitches = (root - 12, root - 5, root, root - 5)
        beat = 0.0
        while beat < measure_beats:
            pattern.append((beat, min(0.5, measure_beats - beat), pitches[int(beat * 2) % len(pitches)]))
            beat += 0.5
    return pattern


def _section_transpose(role: str, local_measure: int) -> int:
    if role == "contrast":
        return 5
    if role == "develop":
        return (0, 2, 5, 7)[(local_measure // 2) % 4]
    if role == "variation":
        return 12 if local_measure % 4 == 2 else 0
    return 0


def _variation_shift(role: str, local_measure: int, rng: random.Random) -> int:
    if role not in ("develop", "variation"):
        return 0
    return rng.choice((-2, 0, 2)) if local_measure % 2 else 0


def _apply_final_cadence(notes: List[NoteEvent], last_measure: int, beats: float, root: int) -> None:
    notes[:] = [note for note in notes if not (note.measure == last_measure and note.staff == 1)]
    start_index = len(notes)
    for offset, pitch in enumerate((root + 12, root + 16, root + 19)):
        notes.append(
            NoteEvent(
                id=f"cadence-{offset + 1}",
                measure=last_measure,
                beat=0.0,
                duration=beats,
                pitch=pitch,
                staff=1,
            )
        )


def _dynamic_map(score: PianoScoreIR) -> Dict[int, str]:
    result: Dict[int, str] = {}
    current = "mp"
    for measure in range(1, score.plan.measure_count + 1):
        for direction in score.directions:
            if direction.measure == measure and direction.kind == "dynamic":
                current = direction.value
        result[measure] = current
    return result


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))
