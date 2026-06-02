"""Profile clean Mutopia MIDI compilation outputs before manual promotion."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence

from .import_midi import parse_midi
from .native_backbone import native_midi_stats


def profile_mutopia_midi_candidates(
    manifest: Path | str,
    compile_root: Path | str,
    output_dir: Path | str,
) -> Dict[str, Any]:
    """Write MIDI content profiles and a bounded-length review-only view."""

    manifest = Path(manifest)
    compile_root = Path(compile_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = []
    errors = []
    for row in _read_jsonl(manifest):
        try:
            profiles.append(_profile_row(row, compile_root))
        except Exception as error:  # noqa: BLE001 - preserve per-score failures in the review queue.
            errors.append({**row, "profile_error": str(error)})
    review_candidates = [row for row in profiles if _is_medium_piece_review_candidate(row)]
    intermediate_proxy_review_candidates = [
        row for row in review_candidates if row["difficulty_proxy"] == "intermediate-proxy"
    ]
    _write_jsonl(output_dir / "midi_profiles.jsonl", profiles)
    _write_jsonl(output_dir / "profile_errors.jsonl", errors)
    _write_jsonl(output_dir / "bounded_piece_review_candidates.jsonl", review_candidates)
    _write_jsonl(output_dir / "intermediate_proxy_review_candidates.jsonl", intermediate_proxy_review_candidates)
    summary = {
        "pipeline": "mutopia-clean-midi-content-profile-v1",
        "manifest": str(manifest.resolve()),
        "compile_root": str(compile_root.resolve()),
        "works_requested": len(profiles) + len(errors),
        "works_profiled": len(profiles),
        "works_failed": len(errors),
        "bounded_piece_review_candidates": len(review_candidates),
        "intermediate_proxy_review_candidates": len(intermediate_proxy_review_candidates),
        "difficulty_proxy_counts": _count(review_candidates, "difficulty_proxy"),
        "profiled_composer_counts": _count(profiles, "composer_style"),
        "bounded_piece_review_candidate_composer_counts": _count(review_candidates, "composer_style"),
        "duration_seconds": _summary(row["stats"]["duration_seconds"] for row in profiles),
        "note_counts": _summary(row["stats"]["notes"] for row in profiles),
        "artifacts": {
            "profiles": "midi_profiles.jsonl",
            "errors": "profile_errors.jsonl",
            "bounded_piece_review_candidates": "bounded_piece_review_candidates.jsonl",
            "intermediate_proxy_review_candidates": "intermediate_proxy_review_candidates.jsonl",
        },
        "promotion_policy": "review-only; bounded duration and density are not difficulty labels",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _profile_row(row: Dict[str, Any], compile_root: Path) -> Dict[str, Any]:
    midi_paths = row.get("compiled_midi_paths") or []
    if not midi_paths:
        raise ValueError("candidate has no compiled MIDI path")
    preferred = _preferred_midi_path(midi_paths)
    midi_path = compile_root / preferred
    if not midi_path.is_file():
        raise ValueError(f"compiled MIDI is missing: {midi_path}")
    stats = native_midi_stats(parse_midi(midi_path))
    difficulty_proxy, difficulty_proxy_evidence = _difficulty_proxy(stats)
    return {
        **row,
        "profiled_midi_path": preferred,
        "native_midi_path": preferred,
        "alternate_compiled_midi_paths": sorted(path for path in midi_paths if path != preferred),
        "difficulty_proxy": difficulty_proxy,
        "difficulty_proxy_evidence": difficulty_proxy_evidence,
        "stats": stats,
    }


def _preferred_midi_path(paths: Sequence[str]) -> str:
    return min(paths, key=lambda path: (Path(path).name not in {"score.mid", "score.midi"}, len(path), path))


def _is_medium_piece_review_candidate(row: Dict[str, Any]) -> bool:
    stats = row["stats"]
    return (
        30 <= stats["duration_seconds"] <= 600
        and stats["notes"] >= 64
        and not stats["potential_density_drift"]
    )


def _difficulty_proxy(stats: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    pitch_span = stats["pitch_range"][1] - stats["pitch_range"][0]
    points = 0
    points += 2 if stats["notes_per_second"] >= 10 else 1 if stats["notes_per_second"] >= 6 else 0
    points += 2 if pitch_span >= 60 else 1 if pitch_span >= 42 else 0
    points += 2 if stats["max_active_notes"] >= 8 else 1 if stats["max_active_notes"] >= 5 else 0
    points += 1 if stats["max_notes_at_onset"] >= 7 else 0
    points += 1 if stats["tempo_events"] >= 8 else 0
    label = "advanced-proxy" if points >= 5 else "intermediate-proxy" if points >= 2 else "easy-proxy"
    return label, {
        "policy": "review-priority-only-not-a-formal-difficulty-label",
        "complexity_points": points,
        "notes_per_second": stats["notes_per_second"],
        "pitch_span": pitch_span,
        "max_active_notes": stats["max_active_notes"],
        "max_notes_at_onset": stats["max_notes_at_onset"],
        "tempo_events": stats["tempo_events"],
    }


def _summary(values: Iterable[float | int]) -> Dict[str, float | int | None]:
    values = list(values)
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": median(values), "max": max(values)}


def _count(rows: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "missing") for row in rows).items()))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile clean Mutopia MIDI candidates")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--compile-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(profile_mutopia_midi_candidates(args.manifest, args.compile_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
