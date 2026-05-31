"""Compact fixed-column ScoreDSL used only for model targets and generation."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .score_ir import (
    LayoutHint,
    MotifBank,
    NoteEvent,
    PianoScoreIR,
    PiecePlanIR,
    SCHEMA_VERSION,
    ScoreDirection,
)


MODEL_SCOREDLS_VERSION = "compact-fixed-columns-v2"
MODEL_SCOREDLS_TAGS = ("SCORE", "NOTE", "DIRECTION", "LAYOUT", "END_SCORE")


def encode_model_score(score: PianoScoreIR) -> str:
    """Encode notation events without repeating the shared plan or motif bank."""

    lines = [_line("SCORE", [MODEL_SCOREDLS_VERSION])]
    lines.extend(
        _line(
            "NOTE",
            [
                note.measure,
                note.beat,
                note.duration,
                note.pitch,
                note.staff,
                note.voice,
                note.articulation,
                note.fingering,
                note.tie_start,
                note.tie_stop,
            ],
        )
        for note in score.notes
    )
    lines.extend(
        _line(
            "DIRECTION",
            [
                direction.measure,
                direction.beat,
                direction.kind,
                direction.value,
                direction.staff,
                direction.end_measure,
                direction.end_beat,
            ],
        )
        for direction in score.directions
    )
    lines.extend(_line("LAYOUT", [hint.measure, hint.kind]) for hint in score.layout_hints)
    lines.append("END_SCORE")
    return "\n".join(lines) + "\n"


def decode_model_score(
    text: str,
    plan: PiecePlanIR,
    motif_bank: MotifBank,
    *,
    metadata: Dict[str, Any] | None = None,
) -> PianoScoreIR:
    """Decode compact notation events against an authoritative shared blueprint."""

    notes: List[NoteEvent] = []
    directions: List[ScoreDirection] = []
    layout_hints: List[LayoutHint] = []
    saw_header = False
    saw_end = False
    for raw_line in text.splitlines():
        if not raw_line:
            continue
        if raw_line == "END_SCORE":
            saw_end = True
            break
        tag, separator, payload = raw_line.partition(" ")
        if not separator:
            raise ValueError(f"Malformed ModelScoreDSL line: {raw_line}")
        data = json.loads(payload)
        if tag == "SCORE":
            if data != [MODEL_SCOREDLS_VERSION]:
                raise ValueError(f"Unsupported ModelScoreDSL header: {data!r}")
            saw_header = True
        elif tag == "NOTE":
            _require_columns(tag, data, 10)
            notes.append(
                NoteEvent(
                    id=f"model-note-{len(notes) + 1}",
                    measure=data[0],
                    beat=data[1],
                    duration=data[2],
                    pitch=data[3],
                    staff=data[4],
                    voice=data[5],
                    articulation=data[6],
                    fingering=data[7],
                    tie_start=data[8],
                    tie_stop=data[9],
                )
            )
        elif tag == "DIRECTION":
            _require_columns(tag, data, 7)
            directions.append(
                ScoreDirection(
                    measure=data[0],
                    beat=data[1],
                    kind=data[2],
                    value=data[3],
                    staff=data[4],
                    end_measure=data[5],
                    end_beat=data[6],
                )
            )
        elif tag == "LAYOUT":
            _require_columns(tag, data, 2)
            layout_hints.append(LayoutHint(measure=data[0], kind=data[1]))
        else:
            raise ValueError(f"Unknown ModelScoreDSL tag: {tag}")
    if not saw_header:
        raise ValueError("ModelScoreDSL is missing SCORE header")
    if not saw_end:
        raise ValueError("ModelScoreDSL is missing END_SCORE")
    return PianoScoreIR(
        plan=plan,
        motif_bank=motif_bank,
        notes=notes,
        directions=directions,
        layout_hints=layout_hints,
        metadata={"model_representation": MODEL_SCOREDLS_VERSION, **(metadata or {})},
        schema_version=SCHEMA_VERSION,
    )


def _line(tag: str, payload: Any) -> str:
    return f"{tag} {json.dumps(payload, separators=(',', ':'), ensure_ascii=True)}"


def _require_columns(tag: str, data: Any, expected: int) -> None:
    if not isinstance(data, list) or len(data) != expected:
        raise ValueError(f"{tag} must contain exactly {expected} fixed columns")
