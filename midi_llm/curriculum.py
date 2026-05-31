"""Build whole-piece ScoreDSL curriculum indexes from a PDMX manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET
import zipfile


WINDOW_SIZES = (16, 32, 64)
AUTOENCODE_WINDOW_SIZE = 16


def prepare_curriculum(
    manifest: Path | str,
    output_dir: Path | str,
    dataset_root: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Write piece blueprints and task indexes without truncating source works."""

    manifest = Path(manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(dataset_root) if dataset_root else manifest.parent
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    blueprints: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, str]] = []
    task_counts: Counter[str] = Counter()
    quality_tier_counts: Counter[str] = Counter()
    quality_task_counts: Counter[str] = Counter()

    for row in rows:
        source = _resolve_source(root, row.get("path", ""))
        blueprint = _build_blueprint(row, source)
        if blueprint is None:
            unresolved.append(
                {
                    "work_id": row["work_id"],
                    "path": str(source),
                    "reason": "MusicXML/MXL source is unavailable and manifest has no measure count",
                }
            )
            continue
        blueprints.append(blueprint)
        quality_tier_counts[blueprint["quality_tier"]] += 1
        piece_examples = _examples_for_blueprint(blueprint)
        examples.extend(piece_examples)
        task_counts.update(example["task"] for example in piece_examples)
        quality_task_counts.update(f"{example['quality_tier']}:{example['task']}" for example in piece_examples)

    _write_jsonl(output_dir / "piece_blueprints.jsonl", blueprints)
    _write_jsonl(output_dir / "curriculum_examples.jsonl", examples)
    _write_jsonl(output_dir / "unresolved_sources.jsonl", unresolved)
    summary = {
        "pipeline": "score-first-piano-curriculum-v1",
        "manifest": str(manifest.resolve()),
        "dataset_root": str(root.resolve()),
        "manifest_works": len(rows),
        "blueprints_written": len(blueprints),
        "unresolved_sources": len(unresolved),
        "examples_written": len(examples),
        "task_counts": dict(sorted(task_counts.items())),
        "quality_tier_counts": dict(sorted(quality_tier_counts.items())),
        "quality_task_counts": dict(sorted(quality_task_counts.items())),
        "invariants": {
            "whole_piece_examples_keep_complete_source": True,
            "score_dsl_autoencode_window_size": AUTOENCODE_WINDOW_SIZE,
            "section_expand_window_sizes": list(WINDOW_SIZES),
            "section_expand_reads_global_blueprint": True,
            "section_expand_reads_neighbor_context": True,
            "ending_target_is_available_to_local_tasks": True,
        },
        "artifacts": {
            "piece_blueprints": "piece_blueprints.jsonl",
            "curriculum_examples": "curriculum_examples.jsonl",
            "unresolved_sources": "unresolved_sources.jsonl",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_blueprint(row: Dict[str, Any], source: Path) -> Optional[Dict[str, Any]]:
    analysis = _analyze_musicxml(source) if source.exists() else None
    measure_count = (analysis or {}).get("measure_count") or _row_measure_count(row)
    if not measure_count:
        return None
    analysis = analysis or {
        "measure_count": measure_count,
        "key": row.get("key") or "unknown",
        "meter": row.get("meter") or "unknown",
        "tempo_bpm": _optional_int(row.get("tempo")) or None,
        "analysis_provenance": "manifest-metadata-v1",
    }
    sections = _heuristic_sections(measure_count)
    return {
        "blueprint_id": f"blueprint-{row['work_id']}",
        "work_id": row["work_id"],
        "split": row["split"],
        "path": row.get("path", ""),
        "title": row.get("title", ""),
        "composer": row.get("composer", ""),
        "genre": row.get("genre", "unclassified-piano"),
        "genre_evidence": row.get("genre_evidence", {}),
        "form": row.get("form", "free-sectional"),
        "form_evidence": row.get("form_evidence", {}),
        "difficulty": row.get("difficulty", "unknown"),
        "difficulty_score": row.get("difficulty_score"),
        "difficulty_evidence": row.get("difficulty_evidence", {}),
        "quality_tier": row.get("quality_tier", "unlabeled"),
        "quality_score": row.get("quality_score"),
        "quality_evidence": row.get("quality_evidence", {}),
        "tasks": row.get(
            "tasks",
            [
                "score-dsl-autoencode",
                "section-expand-16-64",
                "whole-piece-generate",
                "masked-span-inpaint",
                "ending-complete",
                "recapitulation-revise",
            ],
        ),
        **analysis,
        "sections": sections,
        "motif_references": {
            "theme-a": {"source_section": "A", "operations": ["reuse", "variation", "transpose", "develop"]},
            "theme-b": {"source_section": "B", "operations": ["contrast", "develop"]},
        },
        "future_ending_target": {
            "section": sections[-1]["label"],
            "measure": measure_count,
            "cadence": "authentic",
        },
    }


def _examples_for_blueprint(blueprint: Dict[str, Any]) -> List[Dict[str, Any]]:
    count = blueprint["measure_count"]
    full_range = [1, count]
    tasks = set(blueprint["tasks"])
    examples = []
    if "score-dsl-autoencode" in tasks:
        for start in _window_starts(count, min(AUTOENCODE_WINDOW_SIZE, count)):
            examples.append(
                _example(
                    blueprint,
                    "score-dsl-autoencode",
                    [start, min(count, start + AUTOENCODE_WINDOW_SIZE - 1)],
                )
            )
    if "whole-piece-generate" in tasks:
        examples.append(_example(blueprint, "whole-piece-generate", full_range))
    if "section-expand-16-64" in tasks:
        for size in WINDOW_SIZES:
            if size <= count:
                for start in _window_starts(count, size):
                    examples.append(
                        _example(blueprint, "section-expand-16-64", [start, start + size - 1])
                    )
    repair_size = min(16, count)
    middle_start = max(1, (count - repair_size) // 2 + 1)
    if "masked-span-inpaint" in tasks:
        examples.append(
            _example(
                blueprint,
                "masked-span-inpaint",
                [middle_start, middle_start + repair_size - 1],
                bidirectional=True,
            )
        )
    if "ending-complete" in tasks:
        examples.append(
            _example(
                blueprint,
                "ending-complete",
                [max(1, count - repair_size + 1), count],
                bidirectional=False,
            )
        )
    recap = next((section for section in blueprint["sections"] if section["role"] == "return"), None)
    if recap and "recapitulation-revise" in tasks:
        examples.append(
            _example(
                blueprint,
                "recapitulation-revise",
                [recap["start_measure"], recap["end_measure"]],
                bidirectional=True,
            )
        )
    return examples


def _example(
    blueprint: Dict[str, Any],
    task: str,
    target_range: Sequence[int],
    bidirectional: bool = False,
) -> Dict[str, Any]:
    start, end = target_range
    count = blueprint["measure_count"]
    context = {
        "piece_blueprint": blueprint["blueprint_id"],
        "motif_bank": sorted(blueprint["motif_references"]),
        "left_neighbor_range": [max(1, start - 8), start - 1] if start > 1 else None,
        "right_neighbor_range": [end + 1, min(count, end + 8)] if end < count else None,
        "future_ending_target": blueprint["future_ending_target"],
        "bidirectional": bidirectional,
    }
    identity = f"{blueprint['work_id']}|{task}|{start}|{end}"
    return {
        "example_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
        "work_id": blueprint["work_id"],
        "split": blueprint["split"],
        "quality_tier": blueprint["quality_tier"],
        "path": blueprint["path"],
        "task": task,
        "target_range": [start, end],
        "complete_piece_measure_count": count,
        "context": context,
    }


def _window_starts(measure_count: int, size: int) -> List[int]:
    last = measure_count - size + 1
    return sorted({1, max(1, (last + 1) // 2), last})


def _heuristic_sections(measure_count: int) -> List[Dict[str, Any]]:
    if measure_count < 24:
        return [
            {
                "label": "Whole",
                "role": "statement",
                "start_measure": 1,
                "end_measure": measure_count,
                "motif_refs": ["theme-a"],
                "cadence": "authentic",
            }
        ]
    a_end = max(8, round(measure_count * 0.35))
    b_end = max(a_end + 8, round(measure_count * 0.65))
    b_end = min(b_end, measure_count - 8)
    return [
        {
            "label": "A",
            "role": "statement",
            "start_measure": 1,
            "end_measure": a_end,
            "motif_refs": ["theme-a"],
            "cadence": "half",
        },
        {
            "label": "B",
            "role": "contrast",
            "start_measure": a_end + 1,
            "end_measure": b_end,
            "motif_refs": ["theme-b", "theme-a"],
            "cadence": "half",
        },
        {
            "label": "A'",
            "role": "return",
            "start_measure": b_end + 1,
            "end_measure": measure_count,
            "motif_refs": ["theme-a"],
            "cadence": "authentic",
        },
    ]


def _analyze_musicxml(path: Path) -> Optional[Dict[str, Any]]:
    try:
        root = _read_xml_root(path)
    except (ET.ParseError, OSError, KeyError, zipfile.BadZipFile):
        return None
    part = next(iter(_children(root, "part")), None)
    if part is None:
        return None
    measures = list(_children(part, "measure"))
    if not measures:
        return None
    first = measures[0]
    fifths = _find_text(first, ".//key/fifths")
    mode = _find_text(first, ".//key/mode") or "major"
    beats = _find_text(first, ".//time/beats")
    beat_type = _find_text(first, ".//time/beat-type")
    tempo = next(
        (element.attrib.get("tempo") for element in _iter_named(root, "sound") if element.attrib.get("tempo")),
        None,
    )
    return {
        "measure_count": len(measures),
        "key": _key_name(_optional_int(fifths), mode),
        "meter": f"{beats}/{beat_type}" if beats and beat_type else "unknown",
        "tempo_bpm": round(float(tempo)) if tempo else None,
        "analysis_provenance": "musicxml-heuristic-v1",
    }


def _read_xml_root(path: Path) -> ET.Element:
    if path.suffix.lower() != ".mxl":
        return ET.parse(path).getroot()
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".xml", ".musicxml")) and not name.startswith("META-INF/")
        ]
        if not candidates:
            raise KeyError("MXL archive has no MusicXML score")
        return ET.fromstring(archive.read(candidates[0]))


def _children(element: ET.Element, name: str) -> Iterable[ET.Element]:
    return (child for child in element if _local_name(child.tag) == name)


def _iter_named(element: ET.Element, name: str) -> Iterable[ET.Element]:
    return (child for child in element.iter() if _local_name(child.tag) == name)


def _find_text(element: ET.Element, suffix: str) -> Optional[str]:
    names = suffix.replace(".//", "").split("/")
    for candidate in element.iter():
        if _local_name(candidate.tag) != names[-1]:
            continue
        return candidate.text
    return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _key_name(fifths: Optional[int], mode: str) -> str:
    if fifths is None:
        return "unknown"
    major = ("Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#")
    minor = ("Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#")
    index = max(-7, min(7, fifths)) + 7
    normalized_mode = "minor" if mode.lower() == "minor" else "major"
    tonic = minor[index] if normalized_mode == "minor" else major[index]
    return f"{tonic} {normalized_mode}"


def _resolve_source(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _row_measure_count(row: Dict[str, Any]) -> Optional[int]:
    for key in ("measure_count", "n_measures", "measures", "num_measures"):
        value = _optional_int(row.get(key))
        if value:
            return value
    return None


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build whole-piece score-first curriculum indexes")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root")
    args = parser.parse_args()
    summary = prepare_curriculum(args.manifest, args.output_dir, args.dataset_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
