"""Whole-piece planning for score-first classical piano generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .score_ir import (
    Meter,
    PiecePlanIR,
    SUPPORTED_FORMS,
    SUPPORTED_GENRES,
    SectionPlan,
    TempoMark,
)


FORM_LAYOUTS: Dict[str, Sequence[Tuple[str, str, float, str]]] = {
    "binary": (("A", "statement", 0.50, "half"), ("B", "contrast", 0.50, "authentic")),
    "ternary": (
        ("A", "statement", 0.35, "half"),
        ("B", "contrast", 0.30, "half"),
        ("A'", "return", 0.35, "authentic"),
    ),
    "ABA": (
        ("A", "statement", 0.35, "half"),
        ("B", "contrast", 0.30, "half"),
        ("A'", "return", 0.35, "authentic"),
    ),
    "rondo": (
        ("A", "statement", 0.22, "half"),
        ("B", "contrast", 0.17, "half"),
        ("A'", "return", 0.22, "half"),
        ("C", "develop", 0.17, "half"),
        ("A''", "return", 0.22, "authentic"),
    ),
    "theme-and-variations": (
        ("Theme", "statement", 0.25, "authentic"),
        ("Var. I", "variation", 0.25, "authentic"),
        ("Var. II", "variation", 0.25, "half"),
        ("Var. III", "variation", 0.25, "authentic"),
    ),
    "free-sectional": (
        ("Intro", "intro", 0.14, "half"),
        ("A", "statement", 0.30, "half"),
        ("B", "contrast", 0.25, "half"),
        ("A'", "return", 0.23, "half"),
        ("Coda", "coda", 0.08, "authentic"),
    ),
    "sonata-allegro": (
        ("Exposition", "statement", 0.34, "half"),
        ("Development", "develop", 0.26, "half"),
        ("Recapitulation", "return", 0.32, "half"),
        ("Coda", "coda", 0.08, "authentic"),
    ),
}


@dataclass
class ComposeControls:
    genre: str = "nocturne"
    form: str = "ABA"
    duration_minutes: float = 3.0
    measure_range: Tuple[int, int] = (48, 192)
    key: str = "C major"
    meter: Meter = field(default_factory=Meter)
    tempo: int = 84
    difficulty: str = "intermediate"
    texture: str = "melody-with-accompaniment"
    markings: List[str] = field(default_factory=lambda: ["dynamics", "pedal", "articulations"])
    title: Optional[str] = None


def parse_meter(value: str | Meter) -> Meter:
    if isinstance(value, Meter):
        return value
    beats, beat_type = value.split("/", maxsplit=1)
    return Meter(beats=int(beats), beat_type=int(beat_type))


def read_controls(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        return json.loads(text)
    result: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line or ":" not in line:
            continue
        key, raw_value = line.split(":", maxsplit=1)
        result[key.strip()] = _parse_scalar(raw_value.strip())
    return result


def _parse_scalar(value: str) -> Any:
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        return [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value.strip("'\"")


def controls_from_mapping(mapping: Dict[str, Any]) -> ComposeControls:
    controls = ComposeControls()
    for key, value in mapping.items():
        if key == "meter":
            value = parse_meter(str(value))
        elif key == "measure_range":
            if isinstance(value, str):
                parts = value.replace("-", ",").split(",")
                value = (int(parts[0]), int(parts[1]))
            else:
                value = tuple(value)
        elif key == "markings" and isinstance(value, str):
            value = [part.strip() for part in value.split(",") if part.strip()]
        if hasattr(controls, key):
            setattr(controls, key, value)
    return controls


def create_piece_plan(prompt: str, controls: ComposeControls) -> PiecePlanIR:
    if controls.genre not in SUPPORTED_GENRES:
        raise ValueError(f"Unsupported genre: {controls.genre}")
    if controls.form not in SUPPORTED_FORMS:
        raise ValueError(f"Unsupported form: {controls.form}")
    if controls.duration_minutes <= 0:
        raise ValueError("duration_minutes must be positive")

    lower, upper = controls.measure_range
    raw_count = controls.duration_minutes * controls.tempo / controls.meter.quarter_beats
    measure_count = max(lower, min(upper, int(round(raw_count))))
    layout = FORM_LAYOUTS[controls.form]
    lengths = _allocate_measures(measure_count, [weight for _, _, weight, _ in layout])

    tonic = controls.key
    dominant = _related_key(controls.key, "dominant")
    relative = _related_key(controls.key, "relative")
    routes = {
        "statement": tonic,
        "intro": tonic,
        "contrast": relative,
        "develop": dominant,
        "variation": tonic,
        "return": tonic,
        "coda": tonic,
    }
    sections: List[SectionPlan] = []
    start = 1
    for index, ((label, role, _, cadence), length) in enumerate(zip(layout, lengths)):
        refs = ["theme-a"]
        if role in ("contrast", "develop"):
            refs = ["theme-b", "theme-a"]
        elif role == "variation":
            refs = ["theme-a", "accompaniment"]
        sections.append(
            SectionPlan(
                label=label,
                role=role,
                start_measure=start,
                end_measure=start + length - 1,
                key=routes[role],
                tension=round(0.28 + 0.55 * (index / max(1, len(layout) - 1)), 2),
                motif_refs=refs,
                cadence=cadence,
            )
        )
        start += length

    title = controls.title or _title_from_controls(controls)
    tempo_marks = [TempoMark(measure=1, bpm=controls.tempo, text=_tempo_text(controls.tempo))]
    if sections[-1].measure_count >= 4:
        tempo_marks.append(
            TempoMark(
                measure=max(sections[-1].start_measure, measure_count - 3),
                bpm=max(40, round(controls.tempo * 0.84)),
                text="rit.",
            )
        )
    tension_curve = [section.tension for section in sections]
    return PiecePlanIR(
        title=title,
        prompt=prompt,
        genre=controls.genre,
        form=controls.form,
        duration_minutes=controls.duration_minutes,
        measure_count=measure_count,
        key=controls.key,
        meter=controls.meter,
        tempo_bpm=controls.tempo,
        difficulty=controls.difficulty,
        texture=controls.texture,
        sections=sections,
        key_route=[section.key for section in sections],
        tempo_marks=tempo_marks,
        tension_curve=tension_curve,
        markings=controls.markings,
        experimental=controls.form == "sonata-allegro",
    )


def _allocate_measures(total: int, weights: Sequence[float]) -> List[int]:
    lengths = [max(4, int(round(total * weight))) for weight in weights]
    while sum(lengths) > total:
        index = max(range(len(lengths)), key=lengths.__getitem__)
        if lengths[index] <= 4:
            break
        lengths[index] -= 1
    while sum(lengths) < total:
        index = min(range(len(lengths)), key=lengths.__getitem__)
        lengths[index] += 1
    return lengths


def _title_from_controls(controls: ComposeControls) -> str:
    name = controls.genre.replace("-", " ").title()
    return f"{name} in {controls.key}"


def _tempo_text(bpm: int) -> str:
    if bpm < 60:
        return "Adagio"
    if bpm < 84:
        return "Andante"
    if bpm < 112:
        return "Moderato"
    return "Allegro"


def _related_key(key: str, relation: str) -> str:
    tonic, *mode_parts = key.split()
    mode = " ".join(mode_parts) or "major"
    chromatic = {
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
    }
    names = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
    root = chromatic.get(tonic, 0)
    if relation == "dominant":
        return f"{names[(root + 7) % 12]} {mode}"
    if mode == "minor":
        return f"{names[(root + 3) % 12]} major"
    return f"{names[(root + 9) % 12]} minor"
