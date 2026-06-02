"""Select a native classical adapter checkpoint only when upstream parity is preserved."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .native_progress_report import summarize_native_run


def select_native_checkpoint(
    baseline_run: Path | str,
    replay_root: Path | str,
    checkpoint_run_root: Path | str,
) -> Dict[str, Any]:
    """Rank replayed checkpoints and reject every checkpoint that regresses parity."""

    baseline_run = Path(baseline_run)
    replay_root = Path(replay_root)
    checkpoint_run_root = Path(checkpoint_run_root)
    baseline = summarize_native_run(baseline_run)
    rows: List[Dict[str, Any]] = []
    for replay in sorted(replay_root.glob("checkpoint-*"), key=_checkpoint_step):
        if not (replay / "manifest.json").exists():
            continue
        step = _checkpoint_step(replay)
        summary = summarize_native_run(replay)
        rejection_reasons = []
        if summary["syntax_valid_rate"] < baseline["syntax_valid_rate"]:
            rejection_reasons.append("syntax-valid rate regressed")
        if summary["density_drift_rate"] > baseline["density_drift_rate"]:
            rejection_reasons.append("density-drift rate regressed")
        rows.append(
            {
                "step": step,
                "checkpoint_adapter": str(checkpoint_run_root / f"checkpoint-{step}"),
                "replay_run": str(replay),
                "eligible": not rejection_reasons,
                "rejection_reasons": rejection_reasons,
                "summary": summary,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    selected = min(
        eligible,
        key=lambda row: (
            row["summary"]["density_drift_rate"],
            -row["summary"]["natural_stop_rate"],
            -row["summary"]["median_distinct_pitches"],
            row["step"],
        ),
        default=None,
    )
    return {
        "pipeline": "native-classical-checkpoint-parity-selection-v1",
        "baseline_run": str(baseline_run),
        "baseline": baseline,
        "checkpoint_run_root": str(checkpoint_run_root),
        "replay_root": str(replay_root),
        "checkpoints": rows,
        "selected": selected,
        "promotion_decision": "selected-parity-preserving-checkpoint" if selected else "reject-all-checkpoints",
    }


def write_checkpoint_selection(report: Dict[str, Any], output: Path | str) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output


def _checkpoint_step(path: Path) -> int:
    return int(path.name.rsplit("-", maxsplit=1)[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Select parity-preserving native adapter checkpoint")
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--checkpoint-run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = select_native_checkpoint(args.baseline_run, args.replay_root, args.checkpoint_run_root)
    output = write_checkpoint_selection(report, args.output)
    print(output.read_text(encoding="utf-8"))
    raise SystemExit(0 if report["selected"] else 1)


if __name__ == "__main__":
    main()
