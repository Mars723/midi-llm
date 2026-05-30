"""Versioned intermediate representations for score-first piano generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCHEMA_VERSION = "1.0"

SUPPORTED_GENRES = (
    "etude",
    "nocturne",
    "waltz",
    "prelude",
    "minuet",
    "impromptu",
    "theme-and-variations",
)

SUPPORTED_FORMS = (
    "binary",
    "ternary",
    "ABA",
    "rondo",
    "theme-and-variations",
    "free-sectional",
    "sonata-allegro",
)


@dataclass
class Meter:
    beats: int = 4
    beat_type: int = 4

    @property
    def quarter_beats(self) -> float:
        return self.beats * (4.0 / self.beat_type)


@dataclass
class TempoMark:
    measure: int
    bpm: int
    text: str = ""


@dataclass
class SectionPlan:
    label: str
    role: str
    start_measure: int
    end_measure: int
    key: str
    tension: float
    motif_refs: List[str] = field(default_factory=list)
    cadence: str = "half"

    @property
    def measure_count(self) -> int:
        return self.end_measure - self.start_measure + 1


@dataclass
class PiecePlanIR:
    title: str
    prompt: str
    genre: str
    form: str
    duration_minutes: float
    measure_count: int
    key: str
    meter: Meter
    tempo_bpm: int
    difficulty: str
    texture: str
    sections: List[SectionPlan]
    key_route: List[str]
    tempo_marks: List[TempoMark]
    tension_curve: List[float]
    markings: List[str] = field(default_factory=list)
    experimental: bool = False
    schema_version: str = SCHEMA_VERSION


@dataclass
class MotifEvent:
    offset: float
    duration: float
    pitch: int


@dataclass
class Motif:
    id: str
    kind: str
    events: List[MotifEvent]
    description: str


@dataclass
class MotifBank:
    motifs: List[Motif]
    schema_version: str = SCHEMA_VERSION

    def by_id(self, motif_id: str) -> Motif:
        for motif in self.motifs:
            if motif.id == motif_id:
                return motif
        raise KeyError(motif_id)


@dataclass
class NoteEvent:
    id: str
    measure: int
    beat: float
    duration: float
    pitch: int
    staff: int
    voice: int = 1
    articulation: Optional[str] = None
    fingering: Optional[str] = None
    tie_start: bool = False
    tie_stop: bool = False


@dataclass
class ScoreDirection:
    measure: int
    beat: float
    kind: str
    value: str
    staff: int = 1
    end_measure: Optional[int] = None
    end_beat: Optional[float] = None


@dataclass
class LayoutHint:
    measure: int
    kind: str


@dataclass
class PianoScoreIR:
    plan: PiecePlanIR
    motif_bank: MotifBank
    notes: List[NoteEvent]
    directions: List[ScoreDirection]
    layout_hints: List[LayoutHint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass
class PerformanceNote:
    event_id: str
    velocity: int
    onset_shift_beats: float = 0.0
    duration_ratio: float = 1.0


@dataclass
class PedalEvent:
    measure: int
    beat: float
    value: int


@dataclass
class ExpressiveTempoEvent:
    measure: int
    beat: float
    bpm: float


@dataclass
class PianoPerformanceIR:
    notes: List[PerformanceNote]
    pedal: List[PedalEvent]
    tempo_curve: List[ExpressiveTempoEvent]
    schema_version: str = SCHEMA_VERSION


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(value), handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def read_score(path: Path | str) -> PianoScoreIR:
    return score_from_dict(_read_json(path))


def read_performance(path: Path | str) -> PianoPerformanceIR:
    return performance_from_dict(_read_json(path))


def _read_json(path: Path | str) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def plan_from_dict(data: Dict[str, Any]) -> PiecePlanIR:
    return PiecePlanIR(
        **{
            **data,
            "meter": Meter(**data["meter"]),
            "sections": [SectionPlan(**section) for section in data["sections"]],
            "tempo_marks": [TempoMark(**mark) for mark in data["tempo_marks"]],
        }
    )


def motif_bank_from_dict(data: Dict[str, Any]) -> MotifBank:
    return MotifBank(
        motifs=[
            Motif(
                **{
                    **motif,
                    "events": [MotifEvent(**event) for event in motif["events"]],
                }
            )
            for motif in data["motifs"]
        ],
        schema_version=data.get("schema_version", SCHEMA_VERSION),
    )


def score_from_dict(data: Dict[str, Any]) -> PianoScoreIR:
    return PianoScoreIR(
        plan=plan_from_dict(data["plan"]),
        motif_bank=motif_bank_from_dict(data["motif_bank"]),
        notes=[NoteEvent(**note) for note in data["notes"]],
        directions=[ScoreDirection(**direction) for direction in data["directions"]],
        layout_hints=[LayoutHint(**hint) for hint in data.get("layout_hints", [])],
        metadata=data.get("metadata", {}),
        schema_version=data.get("schema_version", SCHEMA_VERSION),
    )


def performance_from_dict(data: Dict[str, Any]) -> PianoPerformanceIR:
    return PianoPerformanceIR(
        notes=[PerformanceNote(**note) for note in data["notes"]],
        pedal=[PedalEvent(**event) for event in data["pedal"]],
        tempo_curve=[ExpressiveTempoEvent(**event) for event in data["tempo_curve"]],
        schema_version=data.get("schema_version", SCHEMA_VERSION),
    )


def section_for_measure(sections: Iterable[SectionPlan], measure: int) -> SectionPlan:
    for section in sections:
        if section.start_measure <= measure <= section.end_measure:
            return section
    raise ValueError(f"No section contains measure {measure}")


def validate_score(score: PianoScoreIR) -> List[str]:
    """Return structural validation errors without modifying the score."""

    errors: List[str] = []
    plan = score.plan
    if plan.genre not in SUPPORTED_GENRES:
        errors.append(f"Unsupported genre: {plan.genre}")
    if plan.form not in SUPPORTED_FORMS:
        errors.append(f"Unsupported form: {plan.form}")
    if not plan.sections:
        errors.append("Piece plan has no sections")
    else:
        expected = 1
        for section in plan.sections:
            if section.start_measure != expected:
                errors.append(f"Section {section.label} should start at measure {expected}")
            if section.end_measure < section.start_measure:
                errors.append(f"Section {section.label} has an invalid range")
            expected = section.end_measure + 1
        if expected - 1 != plan.measure_count:
            errors.append("Sections do not cover the complete piece")
    if not score.notes:
        errors.append("Score has no notes")
    for note in score.notes:
        if not 1 <= note.measure <= plan.measure_count:
            errors.append(f"Note {note.id} falls outside the score")
        if note.staff not in (1, 2):
            errors.append(f"Note {note.id} has invalid staff {note.staff}")
        if not 0 <= note.pitch <= 127:
            errors.append(f"Note {note.id} has invalid pitch {note.pitch}")
        if note.duration <= 0:
            errors.append(f"Note {note.id} has invalid duration")
    return errors
