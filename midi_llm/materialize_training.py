"""Materialize model-ready ScoreDSL examples from curriculum indexes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence
import xml.etree.ElementTree as ET

from .model_scoredsl import MODEL_SCOREDLS_VERSION, encode_model_score
from .musicxml_score import import_musicxml_score
from .notation_analysis import analyze_notation, notation_constraints_from_profile
from .scoredsl import encode_score
from .score_ir import PianoScoreIR, validate_score, write_json
from .train_scoredsl import model_prompt


DEFAULT_MAX_EXAMPLE_CHARACTERS = 175_000


def materialize_training_dataset(
    curriculum_dir: Path | str,
    output_dir: Path | str,
    dataset_root: Optional[Path | str] = None,
    max_example_characters: Optional[int] = DEFAULT_MAX_EXAMPLE_CHARACTERS,
) -> Dict[str, Any]:
    """Import source scores and emit structured inputs paired with ScoreDSL targets."""

    curriculum_dir = Path(curriculum_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(exist_ok=True)
    curriculum_summary = _read_json(curriculum_dir / "summary.json")
    root = Path(dataset_root) if dataset_root else Path(curriculum_summary["dataset_root"])
    examples = _read_jsonl(curriculum_dir / "curriculum_examples.jsonl")
    blueprints = {
        row["work_id"]: row
        for row in _read_jsonl(curriculum_dir / "piece_blueprints.jsonl")
    }
    scores: Dict[str, PianoScoreIR] = {}
    profiles: Dict[str, Dict[str, Any]] = {}
    errors: List[Dict[str, str]] = []
    for work_id, blueprint in blueprints.items():
        source = _resolve_source(root, blueprint["path"])
        try:
            score = import_musicxml_score(
                source,
                title=blueprint.get("title") or None,
                composer=blueprint.get("composer") or None,
                genre=blueprint.get("genre") or None,
                form=blueprint.get("form") or None,
                difficulty=blueprint.get("difficulty") or None,
            )
            validation_errors = validate_score(score)
            if validation_errors:
                raise ValueError("; ".join(validation_errors))
            if blueprint["measure_count"] != score.plan.measure_count:
                raise ValueError(
                    f"Blueprint measure count {blueprint['measure_count']} does not match imported score "
                    f"{score.plan.measure_count}; regenerate curriculum from the available score sources"
                )
        except (OSError, ValueError, ET.ParseError) as error:
            errors.append({"work_id": work_id, "path": str(source), "reason": str(error)})
            continue
        profile = analyze_notation(score)
        score.plan.texture = profile["texture"]
        score.metadata["notation_profile"] = profile
        scores[work_id] = score
        profiles[work_id] = profile
        work_dir = scores_dir / work_id
        work_dir.mkdir(exist_ok=True)
        write_json(work_dir / "score.ir.json", score)
        (work_dir / "score.dsl").write_text(encode_score(score), encoding="utf-8")
        (work_dir / "score.model.dsl").write_text(encode_model_score(score), encoding="utf-8")

    split_rows: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    task_counts: Counter[str] = Counter()
    quality_tier_counts: Counter[str] = Counter()
    texture_counts: Counter[str] = Counter()
    variation_tier_counts: Counter[str] = Counter()
    marking_counts: Counter[str] = Counter()
    skipped_examples = 0
    oversized_examples: List[Dict[str, Any]] = []
    for example in examples:
        score = scores.get(example["work_id"])
        if score is None:
            skipped_examples += 1
            continue
        profile = profiles[example["work_id"]]
        row = _materialize_example(score, example, profile)
        character_length = _example_character_length(row)
        if max_example_characters is not None and character_length > max_example_characters:
            oversized_examples.append(
                {
                    "example_id": row["example_id"],
                    "work_id": row["work_id"],
                    "split": row["split"],
                    "task": row["task"],
                    "quality_tier": row["quality_tier"],
                    "target_scope": row["target_scope"],
                    "target_range": row["target_range"],
                    "character_length": character_length,
                    "max_example_characters": max_example_characters,
                    "reason": "estimated context budget exceeded; target was excluded without truncation",
                }
            )
            continue
        split_rows[example["split"]].append(row)
        task_counts[row["task"]] += 1
        quality_tier_counts[row["quality_tier"]] += 1
    for profile in profiles.values():
        texture_counts[profile["texture"]] += 1
        variation_tier_counts[profile["variation_tier"]] += 1
    for score in scores.values():
        marking_counts.update(score.plan.markings)
    for split in ("train", "valid", "test"):
        _write_jsonl(output_dir / f"{split}.jsonl", split_rows[split])
    _write_jsonl(output_dir / "materialization_errors.jsonl", errors)
    _write_jsonl(output_dir / "oversized_examples.jsonl", oversized_examples)
    summary = {
        "pipeline": "score-first-model-dataset-v3",
        "model_representation": MODEL_SCOREDLS_VERSION,
        "curriculum_dir": str(curriculum_dir.resolve()),
        "dataset_root": str(root.resolve()),
        "source_scores_imported": len(scores),
        "source_scores_failed": len(errors),
        "examples_written": sum(len(rows) for rows in split_rows.values()),
        "examples_skipped": skipped_examples,
        "examples_excluded_oversized": len(oversized_examples),
        "max_example_characters": max_example_characters,
        "oversized_task_counts": dict(sorted(Counter(row["task"] for row in oversized_examples).items())),
        "split_counts": {split: len(split_rows[split]) for split in ("train", "valid", "test")},
        "task_counts": dict(sorted(task_counts.items())),
        "example_quality_tier_counts": dict(sorted(quality_tier_counts.items())),
        "score_texture_counts": dict(sorted(texture_counts.items())),
        "score_variation_tier_counts": dict(sorted(variation_tier_counts.items())),
        "score_marking_counts": dict(sorted(marking_counts.items())),
        "invariants": {
            "targets_are_compact_model_scoredsl": True,
            "rich_score_dsl_is_preserved_as_authoritative_artifact": True,
            "complete_piece_targets_are_not_truncated": True,
            "oversized_targets_are_excluded_without_truncation": True,
            "local_targets_read_complete_piece_plan": True,
            "local_targets_read_complete_motif_bank": True,
            "local_targets_read_neighbor_fragments": True,
            "performance_microtempo_is_not_materialized_as_score_tempo": True,
        },
        "artifacts": {
            "scores": "scores/<work_id>/score.ir.json, score.dsl, and score.model.dsl",
            "train": "train.jsonl",
            "valid": "valid.jsonl",
            "test": "test.jsonl",
            "errors": "materialization_errors.jsonl",
            "oversized_examples": "oversized_examples.jsonl",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _materialize_example(score: PianoScoreIR, example: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    start, end = example["target_range"]
    whole_piece = start == 1 and end == score.plan.measure_count
    left_range = example["context"].get("left_neighbor_range")
    right_range = example["context"].get("right_neighbor_range")
    target = score if whole_piece else _slice_score(score, start, end, "target")
    return {
        "example_id": example["example_id"],
        "work_id": example["work_id"],
        "split": example["split"],
        "task": example["task"],
        "quality_tier": example.get("quality_tier", "unlabeled"),
        "notation_profile": profile,
        "prompt": score.plan.prompt,
        "target_scope": "complete-piece" if whole_piece else "fragment",
        "target_range": [start, end],
        "model_input": {
            "instruction": _instruction(example["task"]),
            "piece_plan": asdict(score.plan),
            "motif_bank": asdict(score.motif_bank),
            "sparse_skeleton": [
                {
                    "label": section.label,
                    "role": section.role,
                    "range": [section.start_measure, section.end_measure],
                    "key": section.key,
                    "motif_refs": section.motif_refs,
                    "cadence": section.cadence,
                }
                for section in score.plan.sections
            ],
            "target_range": [start, end],
            "left_neighbor_scoredsl": _encoded_range(score, left_range, "left-neighbor"),
            "right_neighbor_scoredsl": _encoded_range(score, right_range, "right-neighbor"),
            "future_ending_target": example["context"]["future_ending_target"],
            "notation_constraints": notation_constraints_from_profile(profile, score.plan.difficulty),
            "bidirectional": example["context"]["bidirectional"],
        },
        "target_scoredsl": encode_model_score(target),
    }


def _slice_score(score: PianoScoreIR, start: int, end: int, role: str) -> PianoScoreIR:
    return replace(
        score,
        notes=[note for note in score.notes if start <= note.measure <= end],
        directions=[direction for direction in score.directions if start <= direction.measure <= end],
        layout_hints=[hint for hint in score.layout_hints if start <= hint.measure <= end],
        metadata={**score.metadata, "fragment": {"role": role, "range": [start, end]}},
    )


def _encoded_range(score: PianoScoreIR, measure_range: Optional[Sequence[int]], role: str) -> Optional[str]:
    if not measure_range:
        return None
    return encode_model_score(_slice_score(score, measure_range[0], measure_range[1], role))


def _instruction(task: str) -> str:
    return {
        "score-dsl-autoencode": "Reconstruct a valid notation-first ScoreDSL score.",
        "whole-piece-generate": "Generate the complete piano score from the shared whole-piece plan.",
        "section-expand-16-64": "Expand the target section while respecting the full plan, motifs, neighbors, and ending target.",
        "masked-span-inpaint": "Repair the masked score span using both neighboring score fragments.",
        "ending-complete": "Complete the ending so the full-piece cadence target is realized.",
        "recapitulation-revise": "Revise the recapitulation so it returns to the planned thematic material.",
        "section-variation-revise": "Revise the target section with coherent variation and contrast while preserving the shared plan.",
    }.get(task, f"Complete the notation-first score task: {task}.")


def _example_character_length(row: Dict[str, Any]) -> int:
    return len(model_prompt(row["model_input"])) + len(row["target_scoredsl"])


def _resolve_source(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize model-ready ScoreDSL training examples")
    parser.add_argument("--curriculum-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--max-example-characters", type=int, default=DEFAULT_MAX_EXAMPLE_CHARACTERS)
    args = parser.parse_args()
    summary = materialize_training_dataset(
        args.curriculum_dir,
        args.output_dir,
        args.dataset_root,
        max_example_characters=args.max_example_characters or None,
    )
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if not summary["source_scores_failed"] else 1)


if __name__ == "__main__":
    main()
