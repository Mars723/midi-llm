"""Optional non-commercial SyMuPe research helpers for native MIDI candidates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


QUALITY_MODEL = "SyMuPe/MIDI-Quality-Classifier"
PIANOFLOW_MODEL = "SyMuPe/PianoFlow-base"
LICENSE_NOTICE = (
    "SyMuPe released model weights are CC-BY-NC-SA-4.0. "
    "Use this optional integration only in the acknowledged non-commercial research lane."
)


@dataclass
class CandidateQuality:
    candidate: str
    source: str
    label: str | None
    probabilities: Dict[str, float]


def classify_candidates(run_dir: Path | str, *, device: str = "cuda") -> Path:
    """Classify native MIDI candidates without changing authoritative artifacts."""

    from symupe import AutoClassifier

    run_dir = Path(run_dir)
    classifier = AutoClassifier.from_pretrained(QUALITY_MODEL, device=device)
    rows: List[CandidateQuality] = []
    for midi in _candidate_midis(run_dir):
        result = classifier.predict(midi, show_progress=False)
        rows.append(
            CandidateQuality(
                candidate=midi.parent.name,
                source=str(midi.relative_to(run_dir)),
                label=result.label,
                probabilities=result.probabilities,
            )
        )
    output = run_dir / "symupe-quality-report.json"
    _write_json(output, [asdict(row) for row in rows])
    return output


def perform_candidates(
    run_dir: Path | str,
    *,
    candidates: Iterable[str],
    device: str = "cuda",
    seed: int = 23,
) -> Path:
    """Render separate expressive MIDI layers for selected score MIDI candidates."""

    from symupe import AutoGenerator

    run_dir = Path(run_dir)
    generator = AutoGenerator.from_pretrained(PIANOFLOW_MODEL, device=device)
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        candidate = _normalize_candidate_name(candidate)
        source = run_dir / candidate / "native.mid"
        if not source.exists():
            raise ValueError(f"Native MIDI candidate does not exist: {source}")
        output_dir = run_dir / candidate / "symupe_pianoflow"
        results = generator.perform_score(
            str(source),
            use_score_context=True,
            num_samples=1,
            seed=seed,
        )
        generator.save_performances(results, out_dir=output_dir)
        rows.append(
            {
                "candidate": candidate,
                "source": str(source.relative_to(run_dir)),
                "model": PIANOFLOW_MODEL,
                "artifacts": [
                    str(path.relative_to(run_dir))
                    for path in sorted(output_dir.rglob("*"))
                    if path.is_file()
                ],
            }
        )
    output = run_dir / "symupe-pianoflow-report.json"
    _write_json(output, rows)
    return output


def _candidate_midis(run_dir: Path) -> List[Path]:
    paths = sorted(run_dir.glob("candidate_*/native.mid"))
    if not paths:
        raise ValueError(f"No native MIDI candidates found under {run_dir}")
    return paths


def _normalize_candidate_name(value: str) -> str:
    return value if value.startswith("candidate_") else f"candidate_{value}"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _require_research_license_ack(acknowledged: bool) -> None:
    if not acknowledged:
        raise SystemExit(LICENSE_NOTICE + " Pass --acknowledge-non-commercial-research-license to continue.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run optional SyMuPe research evaluation on native MIDI candidates")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--acknowledge-non-commercial-research-license", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("classify", help="Write SyMuPe MIDI quality classifications")
    perform = subparsers.add_parser("perform", help="Write separate PianoFlow expressive MIDI layers")
    perform.add_argument("--candidates", required=True, help="Comma-separated candidate numbers or names")
    perform.add_argument("--seed", type=int, default=23)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _require_research_license_ack(args.acknowledge_non_commercial_research_license)
    if args.command == "classify":
        output = classify_candidates(args.run_dir, device=args.device)
    else:
        output = perform_candidates(
            args.run_dir,
            candidates=(candidate.strip() for candidate in args.candidates.split(",") if candidate.strip()),
            device=args.device,
            seed=args.seed,
        )
    print(f"SyMuPe research report written to {output.resolve()}")


if __name__ == "__main__":
    main()
