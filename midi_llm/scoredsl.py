"""Line-oriented ScoreDSL used as the model-facing representation."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, Dict, List

from .score_ir import PianoScoreIR, score_from_dict


def encode_score(score: PianoScoreIR) -> str:
    """Encode a score as stable JSON payloads prefixed by semantic token families."""

    lines = [
        _line("SCHEMA", {"version": score.schema_version}),
        _line("PLAN", asdict(score.plan)),
        _line("MOTIF_BANK", asdict(score.motif_bank)),
    ]
    lines.extend(_line("NOTE", asdict(note)) for note in score.notes)
    lines.extend(_line("DIRECTION", asdict(direction)) for direction in score.directions)
    lines.extend(_line("LAYOUT", asdict(hint)) for hint in score.layout_hints)
    lines.append(_line("METADATA", score.metadata))
    lines.append("END_SCORE")
    return "\n".join(lines) + "\n"


def decode_score(text: str) -> PianoScoreIR:
    plan: Dict[str, Any] | None = None
    motif_bank: Dict[str, Any] | None = None
    notes: List[Dict[str, Any]] = []
    directions: List[Dict[str, Any]] = []
    layout_hints: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    schema_version = "1.0"
    for raw_line in text.splitlines():
        if not raw_line or raw_line == "END_SCORE":
            continue
        tag, separator, payload = raw_line.partition(" ")
        if not separator:
            raise ValueError(f"Malformed ScoreDSL line: {raw_line}")
        data = json.loads(payload)
        if tag == "SCHEMA":
            schema_version = data["version"]
        elif tag == "PLAN":
            plan = data
        elif tag == "MOTIF_BANK":
            motif_bank = data
        elif tag == "NOTE":
            notes.append(data)
        elif tag == "DIRECTION":
            directions.append(data)
        elif tag == "LAYOUT":
            layout_hints.append(data)
        elif tag == "METADATA":
            metadata = data
        else:
            raise ValueError(f"Unknown ScoreDSL tag: {tag}")
    if plan is None or motif_bank is None:
        raise ValueError("ScoreDSL is missing PLAN or MOTIF_BANK")
    return score_from_dict(
        {
            "schema_version": schema_version,
            "plan": plan,
            "motif_bank": motif_bank,
            "notes": notes,
            "directions": directions,
            "layout_hints": layout_hints,
            "metadata": metadata,
        }
    )


def _line(tag: str, payload: Any) -> str:
    return f"{tag} {json.dumps(payload, separators=(',', ':'), ensure_ascii=True, sort_keys=True)}"
