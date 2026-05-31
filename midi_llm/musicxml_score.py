"""Import notation-first MusicXML scores into PianoScoreIR."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET
import zipfile

from .score_ir import (
    LayoutHint,
    Meter,
    Motif,
    MotifBank,
    MotifEvent,
    NoteEvent,
    PianoScoreIR,
    PiecePlanIR,
    ScoreDirection,
    SectionPlan,
    TempoMark,
    write_json,
)
from .scoredsl import encode_score


MAJOR_KEYS = ("Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#")
MINOR_KEYS = ("Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#")
PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def import_musicxml_score(
    path: Path | str,
    *,
    title: Optional[str] = None,
    composer: Optional[str] = None,
    prompt: Optional[str] = None,
    genre: Optional[str] = None,
    form: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> PianoScoreIR:
    """Parse the supported notation subset from MusicXML or compressed MXL."""

    path = Path(path)
    root = _read_xml_root(path)
    part = next(iter(_children(root, "part")), None)
    if part is None:
        raise ValueError("MusicXML score has no part")
    measures = list(_children(part, "measure"))
    if not measures:
        raise ValueError("MusicXML score has no measures")

    parsed = _parse_measures(measures)
    meter = parsed["meter"]
    tempo_marks = parsed["tempo_marks"] or [TempoMark(measure=1, bpm=84, text="Imported score tempo")]
    initial_tempo = tempo_marks[0].bpm
    key = parsed["key"]
    measure_count = len(measures)
    motif_bank = _derive_motif_bank(parsed["notes"], meter)
    sections = _sections(measure_count, motif_bank, key)
    score_title = title or _first_text(root, "work-title") or path.stem
    score_composer = composer or _creator(root) or ""
    markings = _marking_families(parsed["notes"], parsed["directions"])
    duration_minutes = round(measure_count * meter.quarter_beats / initial_tempo, 3)
    plan = PiecePlanIR(
        title=score_title,
        prompt=prompt or _training_prompt(score_title, score_composer, key, meter),
        genre=genre or "unclassified-piano",
        form=form or "free-sectional",
        duration_minutes=max(0.001, duration_minutes),
        measure_count=measure_count,
        key=key,
        meter=meter,
        tempo_bpm=initial_tempo,
        difficulty=difficulty or "unknown",
        texture="imported-score",
        sections=sections,
        key_route=[key for _ in sections],
        tempo_marks=tempo_marks,
        tension_curve=[section.tension for section in sections],
        markings=markings,
    )
    return PianoScoreIR(
        plan=plan,
        motif_bank=motif_bank,
        notes=parsed["notes"],
        directions=parsed["directions"],
        layout_hints=parsed["layout_hints"],
        metadata={
            "source": "notation-first-musicxml",
            "source_path": str(path),
            "composer": score_composer,
            "source_classification": {
                "genre": genre or "unclassified-piano",
                "form": form or "free-sectional",
                "difficulty": difficulty or "unknown",
            },
            "key_changes": parsed["key_changes"],
            "meter_changes": parsed["meter_changes"],
            "import_warnings": parsed["warnings"],
        },
    )


def write_imported_score(path: Path | str, output_dir: Path | str) -> Path:
    """Write the normalized IR and ScoreDSL used by the training pipeline."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score = import_musicxml_score(path)
    write_json(output_dir / "score.ir.json", score)
    (output_dir / "score.dsl").write_text(encode_score(score), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "pipeline": "notation-first-musicxml-import-v1",
                "source_musicxml": str(Path(path).resolve()),
                "score": asdict(score.plan),
                "artifacts": {"score_ir": "score.ir.json", "score_dsl": "score.dsl"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir


def _parse_measures(measures: Sequence[ET.Element]) -> Dict[str, Any]:
    divisions = 1
    meter = Meter()
    key = "C major"
    primary_meter: Optional[Meter] = None
    primary_key: Optional[str] = None
    notes: List[NoteEvent] = []
    directions: List[ScoreDirection] = []
    tempo_marks: List[TempoMark] = []
    layout_hints: List[LayoutHint] = []
    key_changes: List[Dict[str, Any]] = []
    meter_changes: List[Dict[str, Any]] = []
    warnings: List[str] = []
    note_index = 0

    for measure_index, measure in enumerate(measures, start=1):
        cursor = 0
        last_onset = 0
        if measure.attrib.get("new-system") == "yes":
            layout_hints.append(LayoutHint(measure=measure_index, kind="system-break"))
        for child in measure:
            name = _local_name(child.tag)
            if name == "print" and child.attrib.get("new-system") == "yes":
                layout_hints.append(LayoutHint(measure=measure_index, kind="system-break"))
            elif name == "attributes":
                new_divisions = _child_int(child, "divisions")
                if new_divisions:
                    divisions = new_divisions
                key_element = _child(child, "key")
                if key_element is not None:
                    new_key = _key_name(_child_int(key_element, "fifths"), _child_text(key_element, "mode") or "major")
                    if new_key != key:
                        key_changes.append({"measure": measure_index, "key": new_key})
                    key = new_key
                    if primary_key is None:
                        primary_key = key
                time = _child(child, "time")
                if time is not None:
                    beats = _child_int(time, "beats")
                    beat_type = _child_int(time, "beat-type")
                    if beats and beat_type:
                        new_meter = Meter(beats=beats, beat_type=beat_type)
                        if new_meter != meter:
                            meter_changes.append({"measure": measure_index, "meter": f"{beats}/{beat_type}"})
                        meter = new_meter
                        if primary_meter is None:
                            primary_meter = meter
            elif name == "backup":
                cursor = max(0, cursor - (_child_int(child, "duration") or 0))
            elif name == "forward":
                cursor += _child_int(child, "duration") or 0
            elif name == "direction":
                parsed_directions, parsed_tempos = _parse_direction(
                    child,
                    measure=measure_index,
                    cursor=cursor,
                    divisions=divisions,
                )
                directions.extend(parsed_directions)
                tempo_marks.extend(parsed_tempos)
            elif name == "sound" and child.attrib.get("tempo"):
                bpm = round(float(child.attrib["tempo"]))
                mark = TempoMark(measure=measure_index, bpm=bpm, text="")
                tempo_marks.append(mark)
                directions.append(ScoreDirection(measure=measure_index, beat=cursor / divisions, kind="tempo", value=f"{bpm}|"))
            elif name == "note":
                duration = _child_int(child, "duration") or 0
                is_chord = _child(child, "chord") is not None
                onset = last_onset if is_chord else cursor
                if not is_chord:
                    last_onset = onset
                if _child(child, "rest") is None and _child(child, "grace") is None:
                    pitch = _pitch(child)
                    if pitch is not None and duration > 0:
                        note_index += 1
                        notes.append(
                            NoteEvent(
                                id=f"xml-{note_index}",
                                measure=measure_index,
                                beat=round(onset / divisions, 4),
                                duration=round(duration / divisions, 4),
                                pitch=pitch,
                                staff=_child_int(child, "staff") or 1,
                                voice=_child_int(child, "voice") or 1,
                                articulation=_articulation(child),
                                fingering=_descendant_text(child, "fingering"),
                                tie_start=_has_tie(child, "start"),
                                tie_stop=_has_tie(child, "stop"),
                            )
                        )
                if not is_chord:
                    cursor += duration

    if not notes:
        warnings.append("Score contains no supported non-grace pitched notes")
    return {
        "meter": primary_meter or meter,
        "key": primary_key or key,
        "notes": notes,
        "directions": _deduplicate_directions(directions),
        "tempo_marks": _deduplicate_tempos(tempo_marks),
        "layout_hints": _deduplicate_layout(layout_hints),
        "key_changes": key_changes,
        "meter_changes": meter_changes,
        "warnings": warnings,
    }


def _parse_direction(
    element: ET.Element,
    *,
    measure: int,
    cursor: int,
    divisions: int,
) -> Tuple[List[ScoreDirection], List[TempoMark]]:
    offset = _child_int(element, "offset") or 0
    beat = round((cursor + offset) / divisions, 4)
    staff = _child_int(element, "staff") or 1
    sound = _child(element, "sound")
    raw_bpm = sound.attrib.get("tempo") if sound is not None else None
    raw_bpm = raw_bpm or _descendant_text(element, "per-minute")
    bpm = round(float(raw_bpm)) if raw_bpm else None
    words = " ".join(
        (child.text or "").strip()
        for direction_type in _children(element, "direction-type")
        for child in direction_type
        if _local_name(child.tag) == "words" and (child.text or "").strip()
    )
    directions: List[ScoreDirection] = []
    tempo_marks: List[TempoMark] = []
    if bpm is not None:
        directions.append(ScoreDirection(measure=measure, beat=beat, kind="tempo", value=f"{bpm}|{words}", staff=staff))
        tempo_marks.append(TempoMark(measure=measure, bpm=bpm, text=words))
    elif words:
        kind = "tempo-text" if any(marker in words.lower() for marker in ("rit", "accel", "a tempo", "rall")) else "words"
        directions.append(ScoreDirection(measure=measure, beat=beat, kind=kind, value=words, staff=staff))

    for direction_type in _children(element, "direction-type"):
        for child in direction_type:
            name = _local_name(child.tag)
            if name == "dynamics":
                dynamic = next(iter(child), None)
                if dynamic is not None:
                    directions.append(
                        ScoreDirection(measure=measure, beat=beat, kind="dynamic", value=_local_name(dynamic.tag), staff=staff)
                    )
            elif name == "wedge":
                wedge_type = child.attrib.get("type", "")
                kind = "wedge-stop" if wedge_type == "stop" else "wedge-start"
                directions.append(ScoreDirection(measure=measure, beat=beat, kind=kind, value=wedge_type or "stop", staff=staff))
            elif name == "pedal":
                pedal_type = child.attrib.get("type", "")
                if pedal_type in ("start", "resume"):
                    directions.append(ScoreDirection(measure=measure, beat=beat, kind="pedal-start", value="start", staff=staff))
                elif pedal_type in ("stop", "discontinue"):
                    directions.append(ScoreDirection(measure=measure, beat=beat, kind="pedal-stop", value="stop", staff=staff))
    return directions, tempo_marks


def _derive_motif_bank(notes: Sequence[NoteEvent], meter: Meter) -> MotifBank:
    melody = [note for note in notes if note.staff == 1]
    bass = [note for note in notes if note.staff == 2]
    motifs: List[Motif] = []
    if melody:
        motifs.append(_motif("theme-a", "melody", melody[:8], meter, "Opening score-derived theme"))
        if len(melody) > 8:
            middle = max(8, len(melody) // 2)
            motifs.append(_motif("theme-b", "melody", melody[middle : middle + 8], meter, "Contrasting score-derived theme"))
    if bass:
        motifs.append(_motif("accompaniment", "accompaniment", bass[:4], meter, "Opening score-derived accompaniment"))
    return MotifBank(motifs=motifs)


def _motif(motif_id: str, kind: str, notes: Sequence[NoteEvent], meter: Meter, description: str) -> Motif:
    first = _absolute_beat(notes[0], meter)
    return Motif(
        id=motif_id,
        kind=kind,
        events=[
            MotifEvent(
                offset=round(_absolute_beat(note, meter) - first, 4),
                duration=note.duration,
                pitch=note.pitch,
            )
            for note in notes
        ],
        description=description,
    )


def _sections(measure_count: int, motif_bank: MotifBank, key: str) -> List[SectionPlan]:
    motif_ids = {motif.id for motif in motif_bank.motifs}
    if measure_count < 24:
        specs = (("Whole", "statement", 1, measure_count, "authentic"),)
    else:
        a_end = max(8, round(measure_count * 0.35))
        b_end = min(max(a_end + 8, round(measure_count * 0.65)), measure_count - 8)
        specs = (
            ("A", "statement", 1, a_end, "half"),
            ("B", "contrast", a_end + 1, b_end, "half"),
            ("A'", "return", b_end + 1, measure_count, "authentic"),
        )
    sections = []
    for index, (label, role, start, end, cadence) in enumerate(specs):
        refs = ["theme-a"] if "theme-a" in motif_ids else []
        if role == "contrast" and "theme-b" in motif_ids:
            refs = ["theme-b", *refs]
        sections.append(
            SectionPlan(
                label=label,
                role=role,
                start_measure=start,
                end_measure=end,
                key=key,
                tension=round(0.3 + 0.5 * index / max(1, len(specs) - 1), 2),
                motif_refs=refs,
                cadence=cadence,
            )
        )
    return sections


def _marking_families(notes: Sequence[NoteEvent], directions: Sequence[ScoreDirection]) -> List[str]:
    markings = set()
    kinds = {direction.kind for direction in directions}
    if "dynamic" in kinds:
        markings.add("dynamics")
    if kinds & {"pedal-start", "pedal-stop"}:
        markings.add("pedal")
    if kinds & {"wedge-start", "wedge-stop"}:
        markings.add("wedges")
    if any(note.articulation for note in notes):
        markings.add("articulations")
    if any(note.fingering for note in notes):
        markings.add("fingering")
    if any(note.tie_start or note.tie_stop for note in notes):
        markings.add("ties")
    return sorted(markings)


def _deduplicate_directions(directions: Sequence[ScoreDirection]) -> List[ScoreDirection]:
    seen = set()
    result = []
    for direction in directions:
        identity = (
            direction.measure,
            direction.beat,
            direction.kind,
            direction.value,
            direction.staff,
        )
        if identity not in seen:
            result.append(direction)
            seen.add(identity)
    return result


def _deduplicate_tempos(marks: Sequence[TempoMark]) -> List[TempoMark]:
    seen = set()
    result = []
    for mark in marks:
        identity = (mark.measure, mark.bpm)
        if identity not in seen:
            result.append(mark)
            seen.add(identity)
    return result


def _deduplicate_layout(hints: Sequence[LayoutHint]) -> List[LayoutHint]:
    seen = set()
    result = []
    for hint in hints:
        identity = (hint.measure, hint.kind)
        if identity not in seen:
            result.append(hint)
            seen.add(identity)
    return result


def _training_prompt(title: str, composer: str, key: str, meter: Meter) -> str:
    byline = f" in the style metadata of {composer}" if composer else ""
    return f"Compose a complete classical piano score titled {title}{byline}, in {key} and {meter.beats}/{meter.beat_type}."


def _pitch(note: ET.Element) -> Optional[int]:
    pitch = _child(note, "pitch")
    if pitch is None:
        return None
    step = _child_text(pitch, "step")
    octave = _child_int(pitch, "octave")
    alter = _child_int(pitch, "alter") or 0
    if step not in PITCH_CLASS or octave is None:
        return None
    return max(0, min(127, (octave + 1) * 12 + PITCH_CLASS[step] + alter))


def _articulation(note: ET.Element) -> Optional[str]:
    for element in note.iter():
        if _local_name(element.tag) == "articulations":
            child = next(iter(element), None)
            return _local_name(child.tag) if child is not None else None
    return None


def _has_tie(note: ET.Element, tie_type: str) -> bool:
    return any(_local_name(element.tag) in ("tie", "tied") and element.attrib.get("type") == tie_type for element in note.iter())


def _absolute_beat(note: NoteEvent, meter: Meter) -> float:
    return (note.measure - 1) * meter.quarter_beats + note.beat


def _read_xml_root(path: Path) -> ET.Element:
    if path.suffix.lower() != ".mxl":
        return ET.parse(path).getroot()
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".xml", ".musicxml")) and not name.startswith("META-INF/")
        ]
        if not candidates:
            raise ValueError("MXL archive has no MusicXML score")
        return ET.fromstring(archive.read(candidates[0]))


def _key_name(fifths: Optional[int], mode: str) -> str:
    if fifths is None:
        return "C major"
    index = max(-7, min(7, fifths)) + 7
    normalized_mode = "minor" if mode.lower() == "minor" else "major"
    tonic = MINOR_KEYS[index] if normalized_mode == "minor" else MAJOR_KEYS[index]
    return f"{tonic} {normalized_mode}"


def _creator(root: ET.Element) -> Optional[str]:
    for child in root.iter():
        if _local_name(child.tag) == "creator" and child.attrib.get("type") in (None, "composer"):
            return child.text
    return None


def _first_text(root: ET.Element, name: str) -> Optional[str]:
    for child in root.iter():
        if _local_name(child.tag) == name:
            return child.text
    return None


def _descendant_text(element: ET.Element, name: str) -> Optional[str]:
    for child in element.iter():
        if _local_name(child.tag) == name:
            return child.text
    return None


def _child(element: ET.Element, name: str) -> Optional[ET.Element]:
    return next((child for child in element if _local_name(child.tag) == name), None)


def _children(element: ET.Element, name: str) -> Iterable[ET.Element]:
    return (child for child in element if _local_name(child.tag) == name)


def _child_text(element: ET.Element, name: str) -> Optional[str]:
    child = _child(element, name)
    return child.text if child is not None else None


def _child_int(element: ET.Element, name: str) -> Optional[int]:
    value = _child_text(element, name)
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a notation-first MusicXML score into ScoreDSL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = write_imported_score(args.input, args.output_dir)
    print(f"Imported notation-first score written to {output.resolve()}")


if __name__ == "__main__":
    main()
