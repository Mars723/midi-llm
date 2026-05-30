"""Import an external performance MIDI as a draft score plus expressive overlay."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Dict, List, Tuple

from .compiler import render_musescore, write_musicxml, write_performance_midi, write_score_midi
from .evaluate import evaluate_score
from .gallery import write_gallery
from .score_ir import (
    ExpressiveTempoEvent,
    Meter,
    NoteEvent,
    PedalEvent,
    PerformanceNote,
    PianoPerformanceIR,
    PianoScoreIR,
    PiecePlanIR,
    ScoreDirection,
    SectionPlan,
    TempoMark,
    MotifBank,
    write_json,
)


@dataclass
class MidiNote:
    tick: int
    duration: int
    pitch: int
    velocity: int


@dataclass
class TempoPoint:
    tick: int
    bpm: float


@dataclass
class PedalPoint:
    tick: int
    value: int


@dataclass
class ParsedMidi:
    ticks_per_beat: int
    notes: List[MidiNote]
    tempos: List[TempoPoint]
    pedal: List[PedalPoint]
    meter: Meter


def parse_midi(path: Path | str) -> ParsedMidi:
    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        raise ValueError("Not a Standard MIDI file")
    header_length, _, track_count, division = struct.unpack(">IHHH", data[4:14])
    if division & 0x8000:
        raise ValueError("SMPTE MIDI timing is not supported")
    offset = 8 + header_length
    notes: List[MidiNote] = []
    tempos: List[TempoPoint] = []
    pedal: List[PedalPoint] = []
    time_signatures: List[Tuple[int, Meter]] = []
    for _ in range(track_count):
        if data[offset : offset + 4] != b"MTrk":
            raise ValueError("Malformed MIDI track")
        track_length = struct.unpack(">I", data[offset + 4 : offset + 8])[0]
        _parse_track(
            data[offset + 8 : offset + 8 + track_length],
            notes,
            tempos,
            pedal,
            time_signatures,
        )
        offset += 8 + track_length
    if not tempos:
        tempos.append(TempoPoint(tick=0, bpm=120.0))
    if tempos[0].tick != 0:
        tempos.insert(0, TempoPoint(tick=0, bpm=120.0))
    meter = sorted(time_signatures, key=lambda item: item[0])[0][1] if time_signatures else Meter()
    return ParsedMidi(
        ticks_per_beat=division,
        notes=sorted(notes, key=lambda note: (note.tick, note.pitch)),
        tempos=sorted(tempos, key=lambda tempo: tempo.tick),
        pedal=sorted(pedal, key=lambda point: point.tick),
        meter=meter,
    )


def import_draft(path: Path | str, output_dir: Path | str, render: bool = True) -> Path:
    parsed = parse_midi(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score, performance = draft_from_parsed(parsed, Path(path).stem)
    write_json(output_dir / "score.ir.json", score)
    write_json(output_dir / "performance.ir.json", performance)
    write_musicxml(score, output_dir / "score.musicxml")
    write_score_midi(score, output_dir / "score.mid")
    write_performance_midi(score, performance, output_dir / "performance.mid")
    render_result = {"available": False, "reason": "MuseScore rendering skipped", "pages": []}
    if render:
        render_result = render_musescore(output_dir / "score.musicxml", output_dir)
    metrics = evaluate_score(score, performance, output_dir / "score.musicxml")
    manifest = {
        "pipeline": "performance-midi-draft-import-v1",
        "draft": True,
        "source_midi": str(Path(path).resolve()),
        "metrics": metrics,
        "render": render_result,
        "artifacts": {
            "score_ir": "score.ir.json",
            "performance_ir": "performance.ir.json",
            "musicxml": "score.musicxml",
            "score_midi": "score.mid",
            "performance_midi": "performance.mid",
            "gallery": "gallery.html",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_gallery(score, manifest, output_dir)
    return output_dir


def draft_from_parsed(parsed: ParsedMidi, title: str) -> Tuple[PianoScoreIR, PianoPerformanceIR]:
    beats_per_measure = parsed.meter.quarter_beats
    max_tick = max((note.tick + note.duration for note in parsed.notes), default=parsed.ticks_per_beat)
    measure_count = max(1, math.ceil(max_tick / parsed.ticks_per_beat / beats_per_measure))
    structural_tempos = select_structural_tempos(
        parsed.tempos,
        ticks_per_beat=parsed.ticks_per_beat,
        beats_per_measure=beats_per_measure,
        final_tick=max_tick,
    )
    tempo_marks = [
        TempoMark(
            measure=_tick_to_measure(point.tick, parsed.ticks_per_beat, beats_per_measure)[0],
            bpm=round(point.bpm),
            text="Imported structural tempo" if index else "Imported tempo",
        )
        for index, point in enumerate(structural_tempos)
    ]
    initial_bpm = tempo_marks[0].bpm
    plan = PiecePlanIR(
        title=f"{title} (Draft Import)",
        prompt="Draft score inferred from an external performance MIDI.",
        genre="prelude",
        form="free-sectional",
        duration_minutes=round(max_tick / parsed.ticks_per_beat / initial_bpm, 3),
        measure_count=measure_count,
        key="C major",
        meter=parsed.meter,
        tempo_bpm=initial_bpm,
        difficulty="unknown",
        texture="draft-import",
        sections=[
            SectionPlan(
                label="Draft",
                role="statement",
                start_measure=1,
                end_measure=measure_count,
                key="C major",
                tension=0.5,
                cadence="unknown",
            )
        ],
        key_route=["C major"],
        tempo_marks=tempo_marks,
        tension_curve=[0.5],
        markings=["draft-import"],
    )
    notes: List[NoteEvent] = []
    performance_notes: List[PerformanceNote] = []
    for index, imported in enumerate(parsed.notes, start=1):
        absolute_beat = imported.tick / parsed.ticks_per_beat
        quantized_beat = round(absolute_beat * 4) / 4
        measure = int(quantized_beat // beats_per_measure) + 1
        beat = round(quantized_beat % beats_per_measure, 3)
        raw_duration = imported.duration / parsed.ticks_per_beat
        duration = max(0.25, round(raw_duration * 4) / 4)
        event_id = f"imported-{index}"
        notes.append(
            NoteEvent(
                id=event_id,
                measure=min(measure, measure_count),
                beat=beat,
                duration=duration,
                pitch=imported.pitch,
                staff=1 if imported.pitch >= 60 else 2,
            )
        )
        performance_notes.append(
            PerformanceNote(
                event_id=event_id,
                velocity=imported.velocity,
                onset_shift_beats=round(absolute_beat - quantized_beat, 4),
                duration_ratio=round(raw_duration / duration, 4),
            )
        )
    directions = [
        ScoreDirection(measure=mark.measure, beat=0.0, kind="tempo", value=f"{mark.bpm}|{mark.text}")
        for mark in tempo_marks
    ]
    gradual = detect_gradual_tempo_text(parsed.tempos, parsed.ticks_per_beat, beats_per_measure)
    directions.extend(gradual)
    score = PianoScoreIR(
        plan=plan,
        motif_bank=MotifBank(motifs=[]),
        notes=notes,
        directions=directions,
        metadata={"draft": True, "source": "external-performance-midi"},
    )
    pedal = [
        PedalEvent(
            measure=_tick_to_measure(point.tick, parsed.ticks_per_beat, beats_per_measure)[0],
            beat=_tick_to_measure(point.tick, parsed.ticks_per_beat, beats_per_measure)[1],
            value=point.value,
        )
        for point in parsed.pedal
    ]
    tempo_curve = [
        ExpressiveTempoEvent(
            measure=_tick_to_measure(point.tick, parsed.ticks_per_beat, beats_per_measure)[0],
            beat=_tick_to_measure(point.tick, parsed.ticks_per_beat, beats_per_measure)[1],
            bpm=round(point.bpm, 3),
        )
        for point in parsed.tempos
    ]
    return score, PianoPerformanceIR(notes=performance_notes, pedal=pedal, tempo_curve=tempo_curve)


def select_structural_tempos(
    tempos: List[TempoPoint],
    ticks_per_beat: int,
    beats_per_measure: float,
    final_tick: int,
    min_measures: int = 2,
    min_change_ratio: float = 0.10,
) -> List[TempoPoint]:
    result = [tempos[0]]
    stable_ticks = ticks_per_beat * beats_per_measure * min_measures
    for index, point in enumerate(tempos[1:], start=1):
        next_tick = tempos[index + 1].tick if index + 1 < len(tempos) else final_tick
        stable_for = next_tick - point.tick
        changed = abs(point.bpm - result[-1].bpm) / max(1.0, result[-1].bpm)
        if stable_for >= stable_ticks and changed >= min_change_ratio:
            result.append(point)
    return result


def detect_gradual_tempo_text(
    tempos: List[TempoPoint],
    ticks_per_beat: int,
    beats_per_measure: float,
) -> List[ScoreDirection]:
    if len(tempos) < 3:
        return []
    first, last = tempos[0], tempos[-1]
    span_measures = (last.tick - first.tick) / ticks_per_beat / beats_per_measure
    bpms = [point.bpm for point in tempos]
    increasing = all(before <= after for before, after in zip(bpms, bpms[1:]))
    decreasing = all(before >= after for before, after in zip(bpms, bpms[1:]))
    if span_measures < 2 or not (increasing or decreasing):
        return []
    measure, beat = _tick_to_measure(first.tick, ticks_per_beat, beats_per_measure)
    return [
        ScoreDirection(
            measure=measure,
            beat=beat,
            kind="tempo-text",
            value="accel." if increasing else "rit.",
        )
    ]


def _parse_track(
    track: bytes,
    notes: List[MidiNote],
    tempos: List[TempoPoint],
    pedal: List[PedalPoint],
    signatures: List[Tuple[int, Meter]],
) -> None:
    index = 0
    tick = 0
    running_status = None
    open_notes: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    while index < len(track):
        delta, index = _read_variable_length(track, index)
        tick += delta
        status = track[index]
        if status & 0x80:
            index += 1
            running_status = status if status < 0xF0 else None
        elif running_status is None:
            raise ValueError("MIDI running status without a prior status byte")
        else:
            status = running_status
        if status == 0xFF:
            meta_type = track[index]
            index += 1
            length, index = _read_variable_length(track, index)
            payload = track[index : index + length]
            index += length
            if meta_type == 0x51 and len(payload) == 3:
                tempos.append(TempoPoint(tick=tick, bpm=60_000_000 / int.from_bytes(payload, "big")))
            elif meta_type == 0x58 and len(payload) >= 2:
                signatures.append((tick, Meter(beats=payload[0], beat_type=2 ** payload[1])))
            continue
        if status in (0xF0, 0xF7):
            length, index = _read_variable_length(track, index)
            index += length
            running_status = None
            continue
        event = status & 0xF0
        channel = status & 0x0F
        data_length = 1 if event in (0xC0, 0xD0) else 2
        payload = track[index : index + data_length]
        index += data_length
        if event == 0x90 and payload[1] > 0:
            open_notes.setdefault((channel, payload[0]), []).append((tick, payload[1]))
        elif event in (0x80, 0x90):
            starts = open_notes.get((channel, payload[0]), [])
            if starts:
                start_tick, velocity = starts.pop(0)
                notes.append(MidiNote(tick=start_tick, duration=max(1, tick - start_tick), pitch=payload[0], velocity=velocity))
        elif event == 0xB0 and payload[0] == 64:
            pedal.append(PedalPoint(tick=tick, value=payload[1]))


def _read_variable_length(data: bytes, index: int) -> Tuple[int, int]:
    value = 0
    while True:
        byte = data[index]
        index += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, index


def _tick_to_measure(tick: int, ticks_per_beat: int, beats_per_measure: float) -> Tuple[int, float]:
    absolute_beat = tick / ticks_per_beat
    return int(absolute_beat // beats_per_measure) + 1, round(absolute_beat % beats_per_measure, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import an external performance MIDI as a draft score")
    parser.add_argument("--input", required=True, help="Input performance MIDI")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-musescore", action="store_true")
    args = parser.parse_args()
    output = import_draft(args.input, args.output_dir, render=not args.skip_musescore)
    print(f"Draft import written to {output.resolve()}")


if __name__ == "__main__":
    main()
