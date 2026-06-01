"""Materialize classical piano training examples in upstream native MIDI tokens."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .native_backbone import upstream_generation_prompt
from .native_tokens import native_event_tokens_to_model_tokens


def materialize_native_training_dataset(
    manifest: Path | str,
    output_dir: Path | str,
    dataset_root: Path | str,
    *,
    max_event_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert complete source MIDIs without changing the upstream token vocabulary."""

    manifest = Path(manifest)
    output_dir = Path(output_dir)
    root = Path(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir = output_dir / "tokens"
    tokens_dir.mkdir(exist_ok=True)
    split_rows: Dict[str, List[Dict[str, Any]]] = {split: [] for split in ("train", "valid", "test")}
    errors: List[Dict[str, str]] = []
    oversized: List[Dict[str, Any]] = []
    for row in _read_jsonl(manifest):
        work_id = row["work_id"]
        source = _resolve_source(root, row.get("native_midi_path", ""))
        if not row.get("native_midi_path"):
            errors.append({"work_id": work_id, "path": "", "reason": "manifest has no native_midi_path"})
            continue
        try:
            event_tokens = _midi_to_event_tokens(source)
            model_tokens = native_event_tokens_to_model_tokens(event_tokens)
        except (OSError, RuntimeError, ValueError) as error:
            errors.append({"work_id": work_id, "path": str(source), "reason": str(error)})
            continue
        if max_event_tokens is not None and len(event_tokens) > max_event_tokens:
            oversized.append(
                {
                    "work_id": work_id,
                    "path": str(source),
                    "event_token_count": len(event_tokens),
                    "max_event_tokens": max_event_tokens,
                    "reason": "complete piece exceeds configured context budget; excluded without truncation",
                }
            )
            continue
        token_path = tokens_dir / f"{work_id}.model_tokens.json"
        token_path.write_text(json.dumps(model_tokens, separators=(",", ":")) + "\n", encoding="utf-8")
        split = row.get("split", "train")
        split_rows.setdefault(split, []).append(
            {
                "work_id": work_id,
                "split": split,
                "source_dataset": row.get("source_dataset", "PDMX"),
                "source_license": row.get("source_license", ""),
                "source_license_url": row.get("source_license_url", ""),
                "native_midi_path": str(source),
                "native_model_tokens_path": str(token_path.relative_to(output_dir)),
                "native_event_token_count": len(event_tokens),
                "native_model_token_count": len(model_tokens),
                "prompt": classical_native_prompt(row),
                "upstream_prompt": upstream_generation_prompt(classical_native_prompt(row)),
                "controls": {
                    "composer_style": row.get("composer_style", "unclassified-composer"),
                    "composer_period": row.get("composer_period", "unclassified-period"),
                    "genre": row.get("genre", "unclassified-piano"),
                    "form": row.get("form", "free-sectional"),
                    "difficulty": row.get("difficulty", "unknown"),
                    "style_tags": row.get("style_tags", []),
                },
                "complete_piece_retained": True,
            }
        )
    for split in ("train", "valid", "test"):
        _write_jsonl(output_dir / f"{split}.jsonl", split_rows[split])
    _write_jsonl(output_dir / "materialization_errors.jsonl", errors)
    _write_jsonl(output_dir / "oversized_complete_pieces.jsonl", oversized)
    written = [row for rows in split_rows.values() for row in rows]
    summary = {
        "pipeline": "classical-piano-upstream-native-token-dataset-v1",
        "manifest": str(manifest.resolve()),
        "dataset_root": str(root.resolve()),
        "examples_written": len(written),
        "examples_failed": len(errors),
        "examples_excluded_oversized": len(oversized),
        "split_counts": {split: len(split_rows[split]) for split in ("train", "valid", "test")},
        "composer_style_counts": dict(sorted(Counter(row["controls"]["composer_style"] for row in written).items())),
        "composer_period_counts": dict(sorted(Counter(row["controls"]["composer_period"] for row in written).items())),
        "genre_counts": dict(sorted(Counter(row["controls"]["genre"] for row in written).items())),
        "event_token_counts": _token_count_summary(written),
        "invariants": {
            "representation": "upstream-native-anticipation-midi-tokens",
            "upstream_native_vocabulary_is_preserved": True,
            "complete_pieces_are_not_truncated": True,
            "score_dsl_is_not_used_as_composition_target": True,
            "composer_style_is_an_explicit_control": True,
        },
        "artifacts": {
            "splits": "train.jsonl, valid.jsonl, test.jsonl",
            "tokens": "tokens/<work_id>.model_tokens.json",
            "errors": "materialization_errors.jsonl",
            "oversized": "oversized_complete_pieces.jsonl",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def classical_native_prompt(row: Dict[str, Any]) -> str:
    """Create a compact text control prompt while preserving structured tags separately."""

    difficulty = row.get("difficulty") or "unspecified-difficulty"
    genre = row.get("genre") or "classical-piano"
    composer = row.get("composer_style") or "unclassified-composer"
    period = row.get("composer_period") or "unclassified-period"
    parts = [f"A {difficulty} solo piano {genre}"]
    if composer != "unclassified-composer":
        parts.append(f"in the style of {composer.title()}")
    if period != "unclassified-period":
        parts.append(f"from the {period} period")
    return ", ".join(parts) + "."


def _midi_to_event_tokens(path: Path) -> List[int]:
    try:
        from anticipation.convert import midi_to_events
    except ImportError as error:
        raise RuntimeError("Install anticipation before materializing native MIDI training tokens") from error
    if not path.exists():
        raise OSError(f"Native MIDI source does not exist: {path}")
    return [int(token_id) for token_id in midi_to_events(str(path))]


def _resolve_source(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _token_count_summary(rows: List[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    counts = sorted(row["native_event_token_count"] for row in rows)
    if not counts:
        return {"min": None, "median": None, "max": None}
    return {"min": counts[0], "median": counts[len(counts) // 2], "max": counts[-1]}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize classical piano data in upstream native MIDI tokens")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-event-tokens", type=int)
    args = parser.parse_args()
    summary = materialize_native_training_dataset(
        args.manifest,
        args.output_dir,
        args.dataset_root,
        max_event_tokens=args.max_event_tokens,
    )
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if not summary["examples_failed"] else 1)


if __name__ == "__main__":
    main()
