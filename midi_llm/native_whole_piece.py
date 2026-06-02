"""Assemble form-planned whole pieces from upstream-native MIDI materials."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .compiler import MIDI_TICKS_PER_BEAT, _tempo_message, _write_midi, write_musicxml
from .evaluate import evaluate_score
from .gallery import write_gallery
from .import_midi import MidiNote, ParsedMidi, draft_from_parsed, parse_midi
from .native_backbone import native_midi_stats
from .score_ir import (
    Meter,
    Motif,
    MotifBank,
    MotifEvent,
    PiecePlanIR,
    ScoreDirection,
    SectionPlan,
    TempoMark,
    write_json,
)


KEY_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def select_native_material(
    run_dir: Path | str,
    target_seconds: float,
    *,
    following_midi: Path | str | None = None,
) -> Path:
    """Select a non-drifting candidate with useful duration and optional return compatibility."""

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    candidates = [
        candidate
        for candidate in manifest["candidates"]
        if not candidate["stats"].get("potential_density_drift", False)
    ]
    if not candidates:
        raise ValueError(f"No non-drifting native MIDI candidates in {run_dir}")
    selected = min(
        candidates,
        key=lambda candidate: (
            abs(candidate["stats"]["duration_seconds"] - target_seconds) / max(1.0, target_seconds)
            + (
                boundary_pitch_distance(
                    parse_midi(run_dir / candidate["native_midi"]).notes,
                    parse_midi(following_midi).notes,
                )
                / 48.0
                if following_midi
                else 0.0
            ),
            -candidate["stats"]["distinct_pitches"],
            candidate["candidate"],
        ),
    )
    return run_dir / selected["native_midi"]


def assemble_hierarchical_aba_piece(
    a_midi: Path | str,
    b_midi: Path | str,
    output_dir: Path | str,
    *,
    coda_midi: Path | str | None = None,
    prompt: str,
    blueprint: str,
    title: str = "Native Nocturne Study",
    tempo_bpm: int = 72,
    a_seconds: float = 38.0,
    b_seconds: float = 42.0,
    coda_seconds: float = 8.0,
    gap_beats: float = 0.25,
) -> Path:
    """Create an ABA plus coda piece without extending one absolute-time token window."""

    if tempo_bpm <= 0:
        raise ValueError("tempo_bpm must be positive")
    if min(a_seconds, b_seconds, coda_seconds) <= 0:
        raise ValueError("section durations must be positive")
    if gap_beats < 0:
        raise ValueError("gap_beats must be non-negative")

    a_midi = Path(a_midi)
    b_midi = Path(b_midi)
    coda_midi = Path(coda_midi) if coda_midi else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    a_parsed = parse_midi(a_midi)
    b_parsed = parse_midi(b_midi)
    ticks_per_beat = MIDI_TICKS_PER_BEAT
    gap_ticks = round(gap_beats * ticks_per_beat)

    a_notes_source, a_end_source = trim_material(a_parsed, a_seconds)
    a_notes = scale_notes(a_notes_source, a_parsed.ticks_per_beat, ticks_per_beat)
    a_end = scale_tick(a_end_source, a_parsed.ticks_per_beat, ticks_per_beat)
    a_notes, a_end = normalize_material_start(a_notes, a_end)
    b_notes_source, b_end_source = trim_material(b_parsed, b_seconds)
    b_notes = scale_notes(b_notes_source, b_parsed.ticks_per_beat, ticks_per_beat)
    b_end = scale_tick(b_end_source, b_parsed.ticks_per_beat, ticks_per_beat)
    b_notes, b_end = normalize_material_start(b_notes, b_end)
    if coda_midi:
        coda_parsed = parse_midi(coda_midi)
        coda_notes_source, coda_end_source = trim_material(coda_parsed, coda_seconds, minimum_ratio=0.5)
        coda_notes = scale_notes(coda_notes_source, coda_parsed.ticks_per_beat, ticks_per_beat)
        coda_end = scale_tick(coda_end_source, coda_parsed.ticks_per_beat, ticks_per_beat)
        coda_notes, coda_end = normalize_material_start(coda_notes, coda_end)
        coda_realization = "prefix-conditioned-model-closure-plus-sparse-cadence"
    else:
        coda_notes = closing_material(a_notes, a_end, coda_seconds, ticks_per_beat, tempo_bpm)
        coda_end = max((note.tick + note.duration for note in coda_notes), default=ticks_per_beat)
        coda_realization = "reused-closing-phrase-plus-sparse-cadence"
    key_root, key_mode = infer_key(a_notes)

    a_offset = 0
    b_offset = a_offset + a_end + gap_ticks
    reprise_offset = b_offset + b_end + gap_ticks
    coda_offset = reprise_offset + a_end + gap_ticks
    cadence_offset = coda_offset + coda_end + gap_ticks
    cadence_notes, cadence_end = sparse_voice_led_cadence(
        coda_notes,
        key_root,
        key_mode,
        cadence_offset,
        ticks_per_beat,
    )
    final_tick = cadence_end

    sections = [
        ("A", "statement", a_offset, a_offset + a_end, "theme-a", "model-generated"),
        ("B", "contrast", b_offset, b_offset + b_end, "theme-b", "model-generated"),
        ("A'", "return", reprise_offset, reprise_offset + a_end, "theme-a", "exact-reuse"),
        ("Coda", "coda", coda_offset, final_tick, "theme-a-closing", coda_realization),
    ]
    performance_notes = (
        offset_notes(a_notes, a_offset, velocity_scale=0.96)
        + offset_notes(b_notes, b_offset, velocity_scale=1.12)
        + offset_notes(a_notes, reprise_offset, velocity_scale=0.88)
        + offset_notes(coda_notes, coda_offset, velocity_scale=0.78)
        + cadence_notes
    )
    score_notes = [
        MidiNote(tick=note.tick, duration=note.duration, pitch=note.pitch, velocity=72)
        for note in performance_notes
    ]
    performance_pedal = performance_pedal_events(final_tick, ticks_per_beat)
    score_events = midi_events(score_notes, tempo_bpm)
    performance_events = midi_events(performance_notes, tempo_bpm, pedal=performance_pedal)
    score_midi = _write_midi(output_dir / "score.mid", score_events)
    performance_midi = _write_midi(output_dir / "performance.mid", performance_events)
    shutil.copyfile(performance_midi, output_dir / "whole_piece.mid")

    parsed_piece = parse_midi(performance_midi)
    stats = native_midi_stats(parsed_piece)
    plan = build_piece_plan(
        title=title,
        prompt=prompt,
        tempo_bpm=tempo_bpm,
        key_root=key_root,
        key_mode=key_mode,
        ticks_per_beat=ticks_per_beat,
        final_tick=final_tick,
        sections=sections,
    )
    motif_bank = build_motif_bank(a_notes, ticks_per_beat)
    write_json(output_dir / "piece_plan.ir.json", plan)
    write_json(output_dir / "motif_bank.ir.json", motif_bank)
    manifest = {
        "pipeline": "form-planned-upstream-native-whole-piece-v1",
        "whole_piece": True,
        "native_midi_is_authoritative_musical_content": True,
        "score_conversion_is_draft_only": True,
        "prompt": prompt,
        "blueprint": blueprint,
        "controls": {
            "genre": "nocturne",
            "form": "ABA-with-coda",
            "difficulty": "intermediate",
            "tempo_bpm": tempo_bpm,
            "inferred_key": f"{KEY_NAMES[key_root]} {key_mode}",
        },
        "materials": {
            "a_midi": str(a_midi.resolve()),
            "b_midi": str(b_midi.resolve()),
            "coda_midi": str(coda_midi.resolve()) if coda_midi else None,
        },
        "sections": [
            {
                "label": label,
                "role": role,
                "start_tick": start,
                "end_tick": end,
                "motif_ref": motif,
                "realization": realization,
            }
            for label, role, start, end, motif, realization in sections
        ],
        "completion": {
            "all_planned_sections_realized": True,
            "a_prime_reuses_theme_a": True,
            "future_termination_condition_realized": True,
            "ending": "sparse voice-led cadence derived from inferred A-material key",
        },
        "limitations": {
            "boundary_inpainting": False,
            "boundary_strategy": "prefix-conditioned continuation plus boundary compatibility ranking",
            "generated_score_markings": "inferred dynamics, wedges, and pedal guidance",
            "human_review_required": True,
            "not_a_release_candidate": True,
        },
        "boundary_analysis": analyze_boundaries(performance_notes, sections, ticks_per_beat),
        "performance_overlay": {
            "velocity_range": [
                min(note.velocity for note in performance_notes),
                max(note.velocity for note in performance_notes),
            ],
            "pedal_events": len(performance_pedal),
        },
        "stats": stats,
        "artifacts": {
            "piece_plan": "piece_plan.ir.json",
            "motif_bank": "motif_bank.ir.json",
            "score_midi": score_midi.name,
            "performance_midi": performance_midi.name,
            "whole_piece_midi": "whole_piece.mid",
        },
    }
    write_plain_json(output_dir / "manifest.json", manifest)
    write_notation_draft(score_midi, performance_midi, output_dir / "score_draft", plan, motif_bank)
    return output_dir


def trim_material(parsed: ParsedMidi, target_seconds: float, minimum_ratio: float = 0.72) -> Tuple[List[MidiNote], int]:
    """Trim generated material at its latest useful phrase gap near a target duration."""

    target_tick = tick_at_seconds(parsed, target_seconds)
    minimum_tick = round(target_tick * minimum_ratio)
    notes = [note for note in parsed.notes if note.tick < target_tick]
    if not notes:
        raise ValueError("Native material has no notes before the requested target duration")
    gap_threshold = max(1, round(parsed.ticks_per_beat * 0.35))
    onsets = sorted({note.tick for note in notes})
    boundaries: List[int] = []
    for next_onset in onsets[1:]:
        prior_end = max((note.tick + note.duration for note in notes if note.tick < next_onset), default=0)
        if prior_end >= minimum_tick and next_onset - prior_end >= gap_threshold:
            boundaries.append(prior_end)
    end_tick = max(boundaries, default=0)
    if not end_tick:
        end_tick = max(
            (note.tick + note.duration for note in notes if note.tick + note.duration <= target_tick),
            default=target_tick,
        )
    selected = [
        MidiNote(
            tick=note.tick,
            duration=max(1, min(note.duration, end_tick - note.tick)),
            pitch=note.pitch,
            velocity=note.velocity,
        )
        for note in notes
        if note.tick < end_tick
    ]
    audible_end = max((note.tick + note.duration for note in selected), default=end_tick)
    return selected, audible_end


def tick_at_seconds(parsed: ParsedMidi, seconds: float) -> int:
    """Map elapsed seconds to ticks while respecting stable source tempo events."""

    remaining = seconds
    tempos = sorted(parsed.tempos, key=lambda point: point.tick)
    for index, tempo in enumerate(tempos):
        next_tick = tempos[index + 1].tick if index + 1 < len(tempos) else None
        seconds_per_tick = 60.0 / tempo.bpm / parsed.ticks_per_beat
        if next_tick is None:
            return tempo.tick + round(remaining / seconds_per_tick)
        span_seconds = (next_tick - tempo.tick) * seconds_per_tick
        if remaining <= span_seconds:
            return tempo.tick + round(remaining / seconds_per_tick)
        remaining -= span_seconds
    return 0


def infer_key(notes: Sequence[MidiNote]) -> Tuple[int, str]:
    """Infer a conservative tonic and mode for a deterministic final cadence."""

    histogram = Counter()
    for note in notes:
        histogram[note.pitch % 12] += max(1, note.duration)
    if not histogram:
        return 0, "minor"
    scores: List[Tuple[float, int, str]] = []
    for root in range(12):
        scores.append((sum(histogram[pitch] * MAJOR_PROFILE[(pitch - root) % 12] for pitch in range(12)), root, "major"))
        scores.append((sum(histogram[pitch] * MINOR_PROFILE[(pitch - root) % 12] for pitch in range(12)), root, "minor"))
    _, root, mode = max(scores)
    return root, mode


def closing_material(
    notes: Sequence[MidiNote],
    end_tick: int,
    seconds: float,
    ticks_per_beat: int,
    tempo_bpm: int,
) -> List[MidiNote]:
    span = max(ticks_per_beat, round(seconds * tempo_bpm / 60 * ticks_per_beat))
    start = max(0, end_tick - span)
    selected = [
        MidiNote(
            tick=max(0, note.tick - start),
            duration=note.duration,
            pitch=note.pitch,
            velocity=note.velocity,
        )
        for note in notes
        if note.tick >= start
    ]
    return selected or [
        MidiNote(tick=0, duration=note.duration, pitch=note.pitch, velocity=note.velocity)
        for note in notes[-8:]
    ]


def sparse_voice_led_cadence(
    preceding_notes: Sequence[MidiNote],
    root: int,
    mode: str,
    offset: int,
    ticks_per_beat: int,
) -> Tuple[List[MidiNote], int]:
    """Append a quiet V-I close with sparse voices instead of a terminal block chord."""

    third = 3 if mode == "minor" else 4
    dominant = (root + 7) % 12
    dominant_duration = 2 * ticks_per_beat
    tonic_duration = 4 * ticks_per_beat
    tonic_offset = offset + dominant_duration
    prior_melody = max((note.pitch for note in preceding_notes[-24:]), default=72)
    prior_bass = min((note.pitch for note in preceding_notes[-24:]), default=45)
    notes = [
        MidiNote(offset, dominant_duration, pitch_near(dominant, prior_bass), 44),
        MidiNote(offset, dominant_duration, pitch_near(dominant, prior_melody), 50),
        MidiNote(tonic_offset, tonic_duration, pitch_near(root, prior_bass - 5), 40),
        MidiNote(tonic_offset, tonic_duration, pitch_near(root, max(60, prior_melody)), 48),
        MidiNote(tonic_offset, tonic_duration, pitch_near((root + third) % 12, prior_melody - 4), 44),
        MidiNote(
            tonic_offset + tonic_duration - ticks_per_beat,
            ticks_per_beat,
            pitch_near(root, max(60, prior_melody)),
            42,
        ),
    ]
    return notes, tonic_offset + tonic_duration


def pitch_near(pitch_class: int, target: int) -> int:
    return min((pitch for pitch in range(21, 109) if pitch % 12 == pitch_class), key=lambda pitch: abs(pitch - target))


def scale_notes(notes: Iterable[MidiNote], source_ticks_per_beat: int, target_ticks_per_beat: int) -> List[MidiNote]:
    return [
        MidiNote(
            tick=scale_tick(note.tick, source_ticks_per_beat, target_ticks_per_beat),
            duration=max(1, scale_tick(note.duration, source_ticks_per_beat, target_ticks_per_beat)),
            pitch=note.pitch,
            velocity=note.velocity,
        )
        for note in notes
    ]


def scale_tick(tick: int, source_ticks_per_beat: int, target_ticks_per_beat: int) -> int:
    return round(tick * target_ticks_per_beat / source_ticks_per_beat)


def normalize_material_start(notes: Sequence[MidiNote], end_tick: int) -> Tuple[List[MidiNote], int]:
    """Remove generated leading silence before placing section-local material."""

    if not notes:
        return [], end_tick
    start = min(note.tick for note in notes)
    return [
        MidiNote(
            tick=note.tick - start,
            duration=note.duration,
            pitch=note.pitch,
            velocity=note.velocity,
        )
        for note in notes
    ], max(1, end_tick - start)


def offset_notes(notes: Iterable[MidiNote], offset: int, velocity_scale: float = 1.0) -> List[MidiNote]:
    return [
        MidiNote(
            tick=note.tick + offset,
            duration=note.duration,
            pitch=note.pitch,
            velocity=max(1, min(127, round(note.velocity * velocity_scale))),
        )
        for note in notes
    ]


def midi_events(
    notes: Iterable[MidiNote],
    tempo_bpm: int,
    *,
    pedal: Iterable[Tuple[int, int]] = (),
) -> List[Tuple[int, int, bytes]]:
    events: List[Tuple[int, int, bytes]] = [(0, -10, b"\xc0\x00"), (0, -9, _tempo_message(tempo_bpm))]
    for note in notes:
        events.append((note.tick, 1, bytes((0x90, note.pitch, note.velocity))))
        events.append((note.tick + note.duration, 0, bytes((0x80, note.pitch, 0))))
    for tick, value in pedal:
        events.append((tick, -1, bytes((0xB0, 64, value))))
    return events


def performance_pedal_events(final_tick: int, ticks_per_beat: int, beats_per_measure: int = 4) -> List[Tuple[int, int]]:
    """Render a restrained measure-level sustain overlay without printing micro-pedal noise."""

    measure_ticks = ticks_per_beat * beats_per_measure
    events = []
    for start in range(0, final_tick, measure_ticks):
        events.append((start + round(ticks_per_beat * 0.10), 64))
        events.append((min(final_tick - 1, start + measure_ticks - round(ticks_per_beat * 0.12)), 0))
    return events


def boundary_pitch_distance(before: Sequence[MidiNote], after: Sequence[MidiNote]) -> float:
    """Estimate boundary compatibility using nearby melody and bass movement."""

    if not before or not after:
        return 99.0
    before_tail = sorted(before, key=lambda note: note.tick)[-24:]
    after_head = sorted(after, key=lambda note: note.tick)[:24]
    melody = abs(max(note.pitch for note in before_tail) - max(note.pitch for note in after_head))
    bass = abs(min(note.pitch for note in before_tail) - min(note.pitch for note in after_head))
    return round((melody + bass) / 2.0, 3)


def analyze_boundaries(
    notes: Sequence[MidiNote],
    sections: Sequence[Tuple[str, str, int, int, str, str]],
    ticks_per_beat: int,
) -> List[Dict[str, Any]]:
    """Report measurable boundary shocks for each assembled section transition."""

    report = []
    for before, after in zip(sections, sections[1:]):
        boundary = after[2]
        prior = [note for note in notes if note.tick < boundary][-24:]
        following = [note for note in notes if note.tick >= boundary][:24]
        prior_end = max((note.tick + note.duration for note in prior), default=boundary)
        next_onset = min((note.tick for note in following), default=boundary)
        report.append(
            {
                "from": before[0],
                "to": after[0],
                "silence_gap_beats": round(max(0, next_onset - prior_end) / ticks_per_beat, 3),
                "pitch_distance": boundary_pitch_distance(prior, following),
                "next_onset_note_count": sum(note.tick == next_onset for note in following),
            }
        )
    return report


def write_notation_draft(
    score_midi: Path,
    performance_midi: Path,
    output_dir: Path,
    plan: PiecePlanIR,
    motif_bank: MotifBank,
) -> Path:
    """Write a draft score with inferred printable markings and an expressive overlay."""

    output_dir.mkdir(parents=True, exist_ok=True)
    score, _ = draft_from_parsed(parse_midi(score_midi), plan.title)
    _, performance = draft_from_parsed(parse_midi(performance_midi), plan.title)
    score.plan = plan
    score.motif_bank = motif_bank
    score.directions = notation_directions(plan)
    score.metadata = {
        "draft": True,
        "source": "form-planned-upstream-native-whole-piece",
        "marking_source": "inferred-print-layer-v1",
    }
    write_json(output_dir / "score.ir.json", score)
    write_json(output_dir / "performance.ir.json", performance)
    write_musicxml(score, output_dir / "score.musicxml")
    shutil.copyfile(score_midi, output_dir / "score.mid")
    shutil.copyfile(performance_midi, output_dir / "performance.mid")
    metrics = evaluate_score(score, performance, output_dir / "score.musicxml")
    manifest = {
        "pipeline": "native-whole-piece-notation-draft-v1",
        "draft": True,
        "metrics": metrics,
        "render": {"available": False, "reason": "MuseScore rendering skipped", "pages": []},
        "artifacts": {
            "score_ir": "score.ir.json",
            "performance_ir": "performance.ir.json",
            "musicxml": "score.musicxml",
            "score_midi": "score.mid",
            "performance_midi": "performance.mid",
            "gallery": "gallery.html",
        },
    }
    write_plain_json(output_dir / "manifest.json", manifest)
    write_gallery(score, manifest, output_dir)
    return output_dir


def notation_directions(plan: PiecePlanIR) -> List[ScoreDirection]:
    """Infer restrained printable dynamics, wedges, and pedal guidance by section."""

    directions = [ScoreDirection(measure=1, beat=0.0, kind="tempo", value=f"{plan.tempo_bpm}|Andante")]
    dynamics = {"statement": "p", "contrast": "mf", "return": "mp", "coda": "pp"}
    for section in plan.sections:
        directions.append(
            ScoreDirection(
                measure=section.start_measure,
                beat=0.0,
                kind="dynamic",
                value=dynamics.get(section.role, "mp"),
            )
        )
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
                measure=section.start_measure,
                beat=max(0.0, plan.meter.quarter_beats - 0.25),
                kind="pedal-stop",
                value="stop",
                staff=2,
            )
        )
        if section.role == "contrast":
            directions.append(
                ScoreDirection(
                    measure=section.start_measure,
                    beat=0.0,
                    kind="wedge-start",
                    value="crescendo",
                    end_measure=min(section.end_measure, section.start_measure + 2),
                )
            )
            directions.append(
                ScoreDirection(
                    measure=min(section.end_measure, section.start_measure + 2),
                    beat=0.0,
                    kind="wedge-stop",
                    value="stop",
                )
            )
        if section.role == "coda":
            directions.append(ScoreDirection(measure=section.start_measure, beat=0.0, kind="tempo-text", value="rit."))
    return directions


def build_piece_plan(
    *,
    title: str,
    prompt: str,
    tempo_bpm: int,
    key_root: int,
    key_mode: str,
    ticks_per_beat: int,
    final_tick: int,
    sections: Sequence[Tuple[str, str, int, int, str, str]],
) -> PiecePlanIR:
    meter = Meter()
    measure_ticks = round(ticks_per_beat * meter.quarter_beats)
    measure_count = max(1, math.ceil(final_tick / measure_ticks))
    boundaries = [0]
    for _, _, _, end, _, _ in sections[:-1]:
        boundaries.append(min(measure_count - 1, max(boundaries[-1] + 1, math.ceil(end / measure_ticks))))
    boundaries.append(measure_count)
    key = f"{KEY_NAMES[key_root]} {key_mode}"
    plans = []
    for index, (label, role, _, _, motif, _) in enumerate(sections):
        plans.append(
            SectionPlan(
                label=label,
                role=role,
                start_measure=boundaries[index] + 1,
                end_measure=boundaries[index + 1],
                key=key,
                tension=(0.35, 0.72, 0.42, 0.18)[index],
                motif_refs=[motif],
                cadence="authentic" if role == "coda" else "half",
            )
        )
    return PiecePlanIR(
        title=title,
        prompt=prompt,
        genre="nocturne",
        form="free-sectional",
        duration_minutes=round(final_tick / ticks_per_beat / tempo_bpm, 3),
        measure_count=measure_count,
        key=key,
        meter=meter,
        tempo_bpm=tempo_bpm,
        difficulty="intermediate",
        texture="melody-with-accompaniment",
        sections=plans,
        key_route=[key for _ in plans],
        tempo_marks=[TempoMark(measure=1, bpm=tempo_bpm, text="Andante")],
        tension_curve=[section.tension for section in plans],
        markings=["native-material", "explicit-theme-reuse", "draft-score-conversion-only"],
    )


def build_motif_bank(notes: Sequence[MidiNote], ticks_per_beat: int) -> MotifBank:
    melody = sorted((note for note in notes if note.pitch >= 60), key=lambda note: (note.tick, note.pitch))[:12]
    if not melody:
        melody = sorted(notes, key=lambda note: (note.tick, note.pitch))[:12]
    start = melody[0].tick if melody else 0
    events = [
        MotifEvent(
            offset=round((note.tick - start) / ticks_per_beat, 3),
            duration=round(note.duration / ticks_per_beat, 3),
            pitch=note.pitch,
        )
        for note in melody
    ]
    return MotifBank(
        motifs=[
            Motif(
                id="theme-a",
                kind="generated-native-opening",
                events=events,
                description="Opening native material reused explicitly in A prime.",
            ),
            Motif(
                id="theme-b",
                kind="prefix-conditioned-contrast",
                events=[],
                description="Contrasting native continuation material selected for return compatibility.",
            ),
            Motif(
                id="theme-a-closing",
                kind="generated-native-coda",
                events=events[-4:],
                description="Closing material followed by a sparse voice-led cadence.",
            ),
        ]
    )


def write_plain_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a form-planned whole piece from native MIDI materials")
    parser.add_argument("--a-run", help="Native backbone run directory containing A-material candidates")
    parser.add_argument("--b-run", help="Native backbone run directory containing B-material candidates")
    parser.add_argument("--a-midi", help="Explicit A-material MIDI path")
    parser.add_argument("--b-midi", help="Explicit B-material MIDI path")
    parser.add_argument("--coda-run", help="Native backbone run directory containing prefix-conditioned coda candidates")
    parser.add_argument("--coda-midi", help="Explicit coda-material MIDI path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--blueprint", required=True)
    parser.add_argument("--title", default="Native Nocturne Study")
    parser.add_argument("--tempo", type=int, default=72)
    parser.add_argument("--a-seconds", type=float, default=38.0)
    parser.add_argument("--b-seconds", type=float, default=42.0)
    parser.add_argument("--coda-seconds", type=float, default=8.0)
    args = parser.parse_args()
    a_midi = Path(args.a_midi) if args.a_midi else select_native_material(args.a_run, args.a_seconds)
    b_midi = Path(args.b_midi) if args.b_midi else select_native_material(args.b_run, args.b_seconds, following_midi=a_midi)
    coda_midi = (
        Path(args.coda_midi)
        if args.coda_midi
        else select_native_material(args.coda_run, args.coda_seconds)
        if args.coda_run
        else None
    )
    output = assemble_hierarchical_aba_piece(
        a_midi,
        b_midi,
        args.output_dir,
        coda_midi=coda_midi,
        prompt=args.prompt,
        blueprint=args.blueprint,
        title=args.title,
        tempo_bpm=args.tempo,
        a_seconds=args.a_seconds,
        b_seconds=args.b_seconds,
        coda_seconds=args.coda_seconds,
    )
    print(f"Native whole-piece sample written to {output.resolve()}")


if __name__ == "__main__":
    main()
