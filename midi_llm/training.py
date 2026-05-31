"""Generate a decision-complete cloud training run plan from a PDMX manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_PHASES = [
    {
        "name": "score-dsl-autoencode",
        "objective": "Learn valid notation grammar and IR round trips",
        "sample_scope": "16 measure notation grammar windows",
        "epochs": 1,
    },
    {
        "name": "section-expand-16-64",
        "objective": "Expand planned sections while reading PiecePlanIR, MotifBank, and neighboring skeleton",
        "sample_scope": "16-64 measure local windows with global context",
        "epochs": 2,
    },
    {
        "name": "whole-piece-generate",
        "objective": "Generate complete 2-6 minute piano miniatures with planned endings",
        "sample_scope": "48-192 measure complete works",
        "epochs": 2,
    },
    {
        "name": "masked-span-inpaint",
        "objective": "Repair boundaries, recapitulations, and endings with bidirectional context",
        "sample_scope": "masked score spans with left and right context",
        "epochs": 1,
    },
    {
        "name": "ending-complete",
        "objective": "Realize planned cadences instead of stopping at arbitrary local-window boundaries",
        "sample_scope": "ending spans with the complete piece plan and future cadence target",
        "epochs": 1,
    },
    {
        "name": "recapitulation-revise",
        "objective": "Return to recognizable thematic material after development and contrast",
        "sample_scope": "recapitulation spans with motif bank and bidirectional neighboring context",
        "epochs": 1,
    },
    {
        "name": "section-variation-revise",
        "objective": "Learn coherent variation and contrast instead of verbatim measure loops",
        "sample_scope": "contrast sections from notation-diverse source scores with bidirectional neighboring context",
        "epochs": 1,
    },
]


def create_run_plan(
    manifest: Path | str,
    output: Path | str,
    curriculum_examples: Optional[Path | str] = None,
    model_dataset_summary: Optional[Path | str] = None,
) -> Dict[str, Any]:
    manifest = Path(manifest)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    split_counts = {
        split: sum(row.get("split") == split for row in rows)
        for split in ("train", "valid", "test")
    }
    quality_tier_counts = Counter(row.get("quality_tier", "unlabeled") for row in rows)
    curriculum = _summarize_curriculum(curriculum_examples)
    run_plan = {
        "pipeline": "score-first-piano-training-v1",
        "base_model": "slseanwu/MIDI-LLM_Llama-3.2-1B",
        "representation": "ModelScoreDSL compact measure-interleaved v3 with rich ScoreDSL artifacts",
        "adaptation": "QLoRA first with compact textual ModelScoreDSL tags in the upstream BPE vocabulary",
        "hardware": "single NVIDIA GPU with 48-80GB VRAM",
        "dataset_policy": (
            "PDMX no_license_conflict + all_valid + best unique arrangement solo-piano core "
            "with tiered quality task gates"
        ),
        "manifest": str(manifest.resolve()),
        "split_counts": split_counts,
        "quality_tier_counts": dict(sorted(quality_tier_counts.items())),
        "curriculum_examples": curriculum,
        "model_dataset": _read_optional_summary(model_dataset_summary),
        "phases": DEFAULT_PHASES,
        "release_gate": {
            "whole_piece_human_review_count": 50,
            "duration_minutes": [2, 6],
            "musicxml_parse_rate_min": 0.98,
            "musescore_render_rate_min": 0.95,
            "control_accuracy_min": 0.90,
            "tempo_overlay_leakage_max": 0,
        },
        "optional_noncommercial_research_track": [
            "SyMuPe/PianoFlow",
            "SyMuPe/PianoCoRe-A*",
        ],
    }
    Path(output).write_text(json.dumps(run_plan, indent=2) + "\n", encoding="utf-8")
    return run_plan


def _summarize_curriculum(path: Optional[Path | str]) -> Dict[str, Any]:
    if not path:
        return {
            "status": "not-prepared",
            "next_command": "python -m midi_llm.curriculum --manifest <manifest> --dataset-root <pdmx-root> --output-dir <curriculum-dir>",
        }
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    counts: Dict[str, int] = {}
    quality_counts: Counter[str] = Counter()
    quality_task_counts: Counter[str] = Counter()
    for row in rows:
        task = row["task"]
        counts[task] = counts.get(task, 0) + 1
        quality_tier = row.get("quality_tier", "unlabeled")
        quality_counts[quality_tier] += 1
        quality_task_counts[f"{quality_tier}:{task}"] += 1
    return {
        "status": "prepared",
        "path": str(path.resolve()),
        "example_count": len(rows),
        "task_counts": dict(sorted(counts.items())),
        "example_quality_tier_counts": dict(sorted(quality_counts.items())),
        "quality_task_counts": dict(sorted(quality_task_counts.items())),
    }


def _read_optional_summary(path: Optional[Path | str]) -> Dict[str, Any]:
    if not path:
        return {
            "status": "not-materialized",
            "next_command": "python -m midi_llm.materialize_training --curriculum-dir <curriculum-dir> --output-dir <model-dataset-dir>",
        }
    path = Path(path)
    return {
        "status": "materialized",
        "path": str(path.resolve()),
        "summary": json.loads(path.read_text(encoding="utf-8")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a cloud training plan from a prepared PDMX manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--curriculum-examples")
    parser.add_argument("--model-dataset-summary")
    args = parser.parse_args()
    plan = create_run_plan(
        args.manifest,
        args.output,
        args.curriculum_examples,
        args.model_dataset_summary,
    )
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
