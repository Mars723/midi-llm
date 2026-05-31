"""Compile score-first IR into MusicXML, MIDI, and MuseScore previews."""

from __future__ import annotations

from collections import defaultdict
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

from .score_ir import (
    NoteEvent,
    PianoPerformanceIR,
    PianoScoreIR,
    ScoreDirection,
)


DIVISIONS = 4
MIDI_TICKS_PER_BEAT = 480
DEFAULT_MUSESCORE_PATHS = (
    "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    "/Applications/MuseScore.app/Contents/MacOS/mscore",
)

KEY_FIFTHS = {
    "C major": 0,
    "A minor": 0,
    "G major": 1,
    "E minor": 1,
    "D major": 2,
    "B minor": 2,
    "A major": 3,
    "F# minor": 3,
    "E major": 4,
    "C# minor": 4,
    "B major": 5,
    "G# minor": 5,
    "Gb major": -6,
    "Eb minor": -6,
    "Db major": -5,
    "Bb minor": -5,
    "Ab major": -4,
    "F minor": -4,
    "Eb major": -3,
    "C minor": -3,
    "Bb major": -2,
    "G minor": -2,
    "F major": -1,
    "D minor": -1,
}

PITCH_SHARP = (
    ("C", None),
    ("C", 1),
    ("D", None),
    ("D", 1),
    ("E", None),
    ("F", None),
    ("F", 1),
    ("G", None),
    ("G", 1),
    ("A", None),
    ("A", 1),
    ("B", None),
)

PITCH_FLAT = (
    ("C", None),
    ("D", -1),
    ("D", None),
    ("E", -1),
    ("E", None),
    ("F", None),
    ("G", -1),
    ("G", None),
    ("A", -1),
    ("A", None),
    ("B", -1),
    ("B", None),
)


def write_musicxml(score: PianoScoreIR, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = score.plan.title
    identification = ET.SubElement(root, "identification")
    ET.SubElement(identification, "creator", type="composer").text = "MIDI-LLM score-first v1"
    encoding = ET.SubElement(identification, "encoding")
    ET.SubElement(encoding, "software").text = "MIDI-LLM score-first compiler"
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Piano"
    part = ET.SubElement(root, "part", id="P1")

    notes_by_measure: Dict[int, List[NoteEvent]] = defaultdict(list)
    directions_by_measure: Dict[int, List[ScoreDirection]] = defaultdict(list)
    layout_by_measure = {hint.measure: hint.kind for hint in score.layout_hints}
    for note in score.notes:
        notes_by_measure[note.measure].append(note)
    for direction in score.directions:
        if direction.kind == "tempo":
            continue
        directions_by_measure[direction.measure].append(direction)
    for mark in score.plan.tempo_marks:
        directions_by_measure[mark.measure].append(
            ScoreDirection(
                measure=mark.measure,
                beat=0.0,
                kind="tempo",
                value=f"{mark.bpm}|{mark.text}",
            )
        )

    for measure_number in range(1, score.plan.measure_count + 1):
        measure = ET.SubElement(part, "measure", number=str(measure_number))
        if layout_by_measure.get(measure_number) == "system-break":
            ET.SubElement(measure, "print", {"new-system": "yes"})
        if measure_number == 1:
            _append_attributes(measure, score)
        for direction in sorted(directions_by_measure[measure_number], key=lambda item: item.beat):
            _append_direction(measure, direction)
        measure_ticks = _ticks(score.plan.meter.quarter_beats)
        for staff in (1, 2):
            if staff == 2:
                _append_backup(measure, measure_ticks)
            _append_staff_notes(
                measure=measure,
                notes=[note for note in notes_by_measure[measure_number] if note.staff == staff],
                staff=staff,
                measure_ticks=measure_ticks,
                prefer_flats=KEY_FIFTHS.get(score.plan.key, 0) < 0,
            )

    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode", xml_declaration=True)
    path.write_text(xml + "\n", encoding="utf-8")
    return path


def write_score_midi(score: PianoScoreIR, path: Path | str) -> Path:
    events: List[Tuple[int, int, bytes]] = [(0, -10, b"\xc0\x00")]
    for mark in score.plan.tempo_marks:
        events.append((_absolute_tick(score, mark.measure, 0.0), -9, _tempo_message(mark.bpm)))
    events.extend(_note_midi_events(score))
    for direction in score.directions:
        if direction.kind in ("pedal-start", "pedal-stop"):
            value = 72 if direction.kind == "pedal-start" else 0
            events.append((_absolute_tick(score, direction.measure, direction.beat), -1, bytes((0xB0, 64, value))))
    return _write_midi(path, events)


def write_performance_midi(score: PianoScoreIR, performance: PianoPerformanceIR, path: Path | str) -> Path:
    events: List[Tuple[int, int, bytes]] = [(0, -10, b"\xc0\x00")]
    for tempo in performance.tempo_curve:
        tick = _absolute_tick(score, tempo.measure, tempo.beat)
        events.append((tick, -9, _tempo_message(tempo.bpm)))
    performance_by_id = {note.event_id: note for note in performance.notes}
    events.extend(_note_midi_events(score, performance_by_id))
    for pedal in performance.pedal:
        tick = _absolute_tick(score, pedal.measure, pedal.beat)
        events.append((tick, -1, bytes((0xB0, 64, pedal.value))))
    return _write_midi(path, events)


def find_musescore(binary: Optional[str] = None) -> Optional[str]:
    if binary:
        return binary if Path(binary).exists() else shutil.which(binary)
    configured = os.environ.get("MUSESCORE_BIN")
    if configured:
        return configured if Path(configured).exists() else shutil.which(configured)
    for candidate in DEFAULT_MUSESCORE_PATHS:
        if Path(candidate).exists():
            return candidate
    return shutil.which("mscore") or shutil.which("musescore")


def render_musescore(
    musicxml_path: Path | str,
    output_dir: Path | str,
    binary: Optional[str] = None,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    musicxml_path = Path(musicxml_path)
    executable = find_musescore(binary)
    if not executable:
        return {"available": False, "reason": "MuseScore executable not found", "pages": []}

    mscz = output_dir / "score.mscz"
    pdf = output_dir / "score.pdf"
    png = output_dir / "score.png"
    _run_musescore(executable, musicxml_path, mscz)
    _run_musescore(executable, mscz, pdf)
    _run_musescore(executable, mscz, png)
    pages = sorted(output_dir.glob("score-*.png"))
    return {
        "available": True,
        "binary": executable,
        "mscz": str(mscz),
        "pdf": str(pdf),
        "pages": [str(page) for page in pages],
    }


def _append_attributes(measure: ET.Element, score: PianoScoreIR) -> None:
    attributes = ET.SubElement(measure, "attributes")
    ET.SubElement(attributes, "divisions").text = str(DIVISIONS)
    key = ET.SubElement(attributes, "key")
    ET.SubElement(key, "fifths").text = str(KEY_FIFTHS.get(score.plan.key, 0))
    time = ET.SubElement(attributes, "time")
    ET.SubElement(time, "beats").text = str(score.plan.meter.beats)
    ET.SubElement(time, "beat-type").text = str(score.plan.meter.beat_type)
    ET.SubElement(attributes, "staves").text = "2"
    clef = ET.SubElement(attributes, "clef", number="1")
    ET.SubElement(clef, "sign").text = "G"
    ET.SubElement(clef, "line").text = "2"
    clef = ET.SubElement(attributes, "clef", number="2")
    ET.SubElement(clef, "sign").text = "F"
    ET.SubElement(clef, "line").text = "4"


def _append_direction(measure: ET.Element, direction: ScoreDirection) -> None:
    element = ET.SubElement(measure, "direction", placement="below" if direction.staff == 2 else "above")
    direction_type = ET.SubElement(element, "direction-type")
    if direction.kind == "tempo":
        bpm, _, text = direction.value.partition("|")
        if text:
            ET.SubElement(direction_type, "words").text = text
        metronome = ET.SubElement(direction_type, "metronome")
        ET.SubElement(metronome, "beat-unit").text = "quarter"
        ET.SubElement(metronome, "per-minute").text = bpm
        ET.SubElement(element, "sound", tempo=bpm)
    elif direction.kind == "tempo-text":
        ET.SubElement(direction_type, "words").text = direction.value
    elif direction.kind == "dynamic":
        dynamics = ET.SubElement(direction_type, "dynamics")
        ET.SubElement(dynamics, direction.value)
    elif direction.kind in ("wedge-start", "wedge-stop"):
        wedge_type = direction.value if direction.kind == "wedge-start" else "stop"
        ET.SubElement(direction_type, "wedge", type=wedge_type)
    elif direction.kind in ("pedal-start", "pedal-stop"):
        pedal_type = "start" if direction.kind == "pedal-start" else "stop"
        ET.SubElement(direction_type, "pedal", type=pedal_type, line="yes")
    else:
        ET.SubElement(direction_type, "words").text = direction.value
    if direction.beat:
        ET.SubElement(element, "offset").text = str(_ticks(direction.beat))
    ET.SubElement(element, "staff").text = str(direction.staff)


def _append_staff_notes(
    measure: ET.Element,
    notes: Sequence[NoteEvent],
    staff: int,
    measure_ticks: int,
    prefer_flats: bool,
) -> None:
    voices = sorted({note.voice for note in notes})
    if not voices:
        _append_rest(measure, measure_ticks, staff)
        return
    for index, voice in enumerate(voices):
        if index:
            _append_backup(measure, measure_ticks)
        _append_voice_notes(
            measure,
            [note for note in notes if note.voice == voice],
            staff,
            voice,
            measure_ticks,
            prefer_flats,
        )


def _append_voice_notes(
    measure: ET.Element,
    notes: Sequence[NoteEvent],
    staff: int,
    voice: int,
    measure_ticks: int,
    prefer_flats: bool,
) -> None:
    groups: Dict[int, List[NoteEvent]] = defaultdict(list)
    for note in notes:
        groups[_ticks(note.beat)].append(note)
    cursor = 0
    for onset in sorted(groups):
        if onset > cursor:
            _append_rest(measure, onset - cursor, staff, voice)
            cursor = onset
        group = groups[onset]
        duration_ticks = max(_ticks(note.duration) for note in group)
        for index, note in enumerate(group):
            _append_note(measure, note, staff, chord=index > 0, prefer_flats=prefer_flats)
        cursor += duration_ticks
    if cursor < measure_ticks:
        _append_rest(measure, measure_ticks - cursor, staff, voice)


def _append_note(measure: ET.Element, note: NoteEvent, staff: int, chord: bool, prefer_flats: bool) -> None:
    element = ET.SubElement(measure, "note")
    if chord:
        ET.SubElement(element, "chord")
    pitch = ET.SubElement(element, "pitch")
    step, alter = (PITCH_FLAT if prefer_flats else PITCH_SHARP)[note.pitch % 12]
    ET.SubElement(pitch, "step").text = step
    if alter is not None:
        ET.SubElement(pitch, "alter").text = str(alter)
    ET.SubElement(pitch, "octave").text = str(note.pitch // 12 - 1)
    ET.SubElement(element, "duration").text = str(_ticks(note.duration))
    ET.SubElement(element, "voice").text = str(note.voice)
    note_type, dotted = _note_type(_ticks(note.duration))
    ET.SubElement(element, "type").text = note_type
    if dotted:
        ET.SubElement(element, "dot")
    ET.SubElement(element, "staff").text = str(staff)
    if note.tie_start:
        ET.SubElement(element, "tie", type="start")
    if note.tie_stop:
        ET.SubElement(element, "tie", type="stop")
    if note.articulation or note.fingering or note.tie_start or note.tie_stop:
        notations = ET.SubElement(element, "notations")
        if note.tie_start:
            ET.SubElement(notations, "tied", type="start")
        if note.tie_stop:
            ET.SubElement(notations, "tied", type="stop")
        if note.articulation:
            articulations = ET.SubElement(notations, "articulations")
            ET.SubElement(articulations, note.articulation)
        if note.fingering:
            technical = ET.SubElement(notations, "technical")
            ET.SubElement(technical, "fingering").text = note.fingering


def _append_rest(measure: ET.Element, duration_ticks: int, staff: int, voice: int = 1) -> None:
    if duration_ticks <= 0:
        return
    element = ET.SubElement(measure, "note")
    ET.SubElement(element, "rest")
    ET.SubElement(element, "duration").text = str(duration_ticks)
    ET.SubElement(element, "voice").text = str(voice)
    note_type, dotted = _note_type(duration_ticks)
    ET.SubElement(element, "type").text = note_type
    if dotted:
        ET.SubElement(element, "dot")
    ET.SubElement(element, "staff").text = str(staff)


def _append_backup(measure: ET.Element, duration_ticks: int) -> None:
    backup = ET.SubElement(measure, "backup")
    ET.SubElement(backup, "duration").text = str(duration_ticks)


def _note_type(duration_ticks: int) -> Tuple[str, bool]:
    mapping = {
        1: ("16th", False),
        2: ("eighth", False),
        3: ("eighth", True),
        4: ("quarter", False),
        6: ("quarter", True),
        8: ("half", False),
        12: ("half", True),
        16: ("whole", False),
    }
    return mapping.get(duration_ticks, ("quarter", False))


def _note_midi_events(score: PianoScoreIR, performance_by_id=None) -> List[Tuple[int, int, bytes]]:
    performance_by_id = performance_by_id or {}
    events: List[Tuple[int, int, bytes]] = []
    for note in score.notes:
        rendered = performance_by_id.get(note.id)
        shift = rendered.onset_shift_beats if rendered else 0.0
        ratio = rendered.duration_ratio if rendered else 1.0
        velocity = rendered.velocity if rendered else 72
        start = _absolute_tick(score, note.measure, note.beat + shift)
        duration = max(1, round(note.duration * ratio * MIDI_TICKS_PER_BEAT))
        end = start + duration
        events.append((start, 1, bytes((0x90, note.pitch, velocity))))
        events.append((end, 0, bytes((0x80, note.pitch, 0))))
    return events


def _absolute_tick(score: PianoScoreIR, measure: int, beat: float) -> int:
    position = (measure - 1) * score.plan.meter.quarter_beats + beat
    return max(0, round(position * MIDI_TICKS_PER_BEAT))


def _tempo_message(bpm: float) -> bytes:
    micros = max(1, round(60_000_000 / bpm))
    return b"\xff\x51\x03" + micros.to_bytes(3, byteorder="big")


def _write_midi(path: Path | str, events: Iterable[Tuple[int, int, bytes]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    track = bytearray()
    previous_tick = 0
    for tick, _, message in sorted(events, key=lambda event: (event[0], event[1])):
        track.extend(_variable_length(max(0, tick - previous_tick)))
        track.extend(message)
        previous_tick = tick
    track.extend(b"\x00\xff\x2f\x00")
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, MIDI_TICKS_PER_BEAT)
    payload = header + b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    path.write_bytes(payload)
    return path


def _variable_length(value: int) -> bytes:
    buffer = value & 0x7F
    result = bytearray((buffer,))
    while value >> 7:
        value >>= 7
        buffer = (value & 0x7F) | 0x80
        result.insert(0, buffer)
    return bytes(result)


def _ticks(beats: float) -> int:
    return round(beats * DIVISIONS)


def _run_musescore(binary: str, source: Path, target: Path) -> None:
    _remove_musescore_export(target)
    for attempt in range(4):
        try:
            subprocess.run(
                [binary, "-F", "-o", str(target), str(source)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
            return
        except subprocess.CalledProcessError:
            # MuseScore 4.7.2 on macOS can abort before or after a CLI export.
            if _musescore_export_exists(target):
                return
            if attempt == 3:
                raise
            time.sleep(1)


def _musescore_export_exists(target: Path) -> bool:
    if target.exists() and target.stat().st_size:
        return True
    if target.suffix.lower() == ".png":
        return any(page.stat().st_size for page in target.parent.glob(f"{target.stem}-*.png"))
    return False


def _remove_musescore_export(target: Path) -> None:
    target.unlink(missing_ok=True)
    if target.suffix.lower() == ".png":
        for page in target.parent.glob(f"{target.stem}-*.png"):
            page.unlink()
