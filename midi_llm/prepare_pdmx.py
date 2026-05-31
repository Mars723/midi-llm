"""Create labeled, deduplicated PDMX manifests for score-first training."""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


GENRE_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("theme-and-variations", ("theme and variations", "variations", "variation", "variationes", "variationen")),
    ("nocturne", ("nocturne", "notturno")),
    ("impromptu", ("impromptu",)),
    ("prelude", ("prelude", "preludio", "praeludium")),
    ("minuet", ("minuet", "menuet", "minuetto")),
    ("waltz", ("waltz", "valse", "walzer")),
    ("etude", ("etude", "study", "studi")),
)
DIFFICULTY_LABELS = ("easy", "intermediate", "advanced")
QUALITY_FILTERS = ("metadata-curated", "canonical-core")
GENERIC_GENRES = ("classical-piano", "unclassified-piano")
CANONICAL_CLASSICAL_COMPOSERS = (
    "albinoni",
    "alkan",
    "arensky",
    "bach",
    "balakirev",
    "bartok",
    "beethoven",
    "borodin",
    "brahms",
    "burgmuller",
    "chopin",
    "clementi",
    "corelli",
    "couperin",
    "czerny",
    "debussy",
    "dvorak",
    "faure",
    "frescobaldi",
    "glazunov",
    "gounod",
    "grieg",
    "handel",
    "hanon",
    "haydn",
    "heller",
    "kabalevsky",
    "korsakov",
    "kuhlau",
    "liszt",
    "lully",
    "mendelssohn",
    "monteverdi",
    "moszkowski",
    "mussorgsky",
    "mozart",
    "offenbach",
    "pachelbel",
    "poulenc",
    "prokofiev",
    "purcell",
    "rachmaninoff",
    "rameau",
    "ravel",
    "satie",
    "scarlatti",
    "schubert",
    "schumann",
    "scriabin",
    "shostakovich",
    "strauss",
    "taneyev",
    "telemann",
    "tchaikovsky",
    "verdi",
    "vivaldi",
)
ARRANGEMENT_TITLE_MARKERS = (
    "2 pianos",
    "4 hands",
    "accordion",
    "arr",
    "arranged",
    "arrangement",
    "concerto",
    "easy piano",
    "edit",
    "flute",
    "for solo piano",
    "four hands",
    "live recording",
    "medley",
    "movie",
    "opera",
    "orchestra",
    "piano solo",
    "reduction",
    "simple piano",
    "soundtrack",
    "symphony",
    "theme from",
    "two pianos",
    "violin",
    "cello",
)


def prepare_manifest(
    metadata_csv: Path | str,
    output_dir: Path | str,
    *,
    genres: Sequence[str] = (),
    difficulty: Optional[str] = None,
    quality: Optional[str] = None,
    min_measures: Optional[int] = None,
    max_measures: Optional[int] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Write a no-license-conflict PDMX manifest with auditable labels."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(metadata_csv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [
        row
        for row in rows
        if _subset_enabled(row, "all_valid")
        and _subset_enabled(row, "no_license_conflict")
        and _is_solo_piano(row)
        and _is_preferred_arrangement(row)
    ]
    difficulty_scores = _difficulty_scores(eligible)
    requested_genres = set(genres)
    accepted = []
    seen = set()
    for row, difficulty_score in zip(eligible, difficulty_scores):
        work_id = _work_id(row)
        if work_id in seen:
            continue
        seen.add(work_id)
        genre, genre_evidence = _genre_label(row)
        form, form_evidence = _form_label(row, genre)
        difficulty_label = _difficulty_label(difficulty_score)
        quality_tier, quality_score, quality_evidence = _quality_label(row, genre)
        measure_count = _measure_count(row)
        if requested_genres and genre not in requested_genres:
            continue
        if difficulty and difficulty_label != difficulty:
            continue
        if quality and not _quality_matches(quality_tier, quality):
            continue
        if min_measures is not None and (measure_count is None or measure_count < min_measures):
            continue
        if max_measures is not None and (measure_count is None or measure_count > max_measures):
            continue
        accepted.append(
            {
                "work_id": work_id,
                "split": _split(work_id),
                "path": _score_path(row),
                "title": row.get("title") or row.get("song_name") or "",
                "composer": row.get("composer") or row.get("composer_name") or "",
                "measure_count": measure_count,
                "genre": genre,
                "genre_evidence": genre_evidence,
                "form": form,
                "form_evidence": form_evidence,
                "difficulty": difficulty_label,
                "difficulty_score": round(difficulty_score, 4),
                "difficulty_evidence": _difficulty_evidence(row),
                "quality_tier": quality_tier,
                "quality_score": quality_score,
                "quality_evidence": quality_evidence,
                "source_genres": row.get("genres") or "",
                "source_tracks": row.get("tracks") or row.get("instrumentation") or "",
                "tasks": _tasks_for_quality(quality_tier),
            }
        )
    accepted.sort(key=lambda row: row["work_id"])
    if limit is not None:
        accepted = accepted[:limit]
    manifest_path = output_dir / "pdmx_score_first_manifest.jsonl"
    _write_jsonl(manifest_path, accepted)
    summary = {
        "pipeline": "score-first-pdmx-labeling-v2",
        "input_rows": len(rows),
        "eligible_deduplicated_no_license_conflict_solo_piano_works": len(eligible),
        "accepted_unique_solo_piano_works": len(accepted),
        "train": sum(row["split"] == "train" for row in accepted),
        "valid": sum(row["split"] == "valid" for row in accepted),
        "test": sum(row["split"] == "test" for row in accepted),
        "genre_counts": dict(sorted(Counter(row["genre"] for row in accepted).items())),
        "form_counts": dict(sorted(Counter(row["form"] for row in accepted).items())),
        "difficulty_counts": dict(sorted(Counter(row["difficulty"] for row in accepted).items())),
        "quality_counts": dict(sorted(Counter(row["quality_tier"] for row in accepted).items())),
        "filters": {
            "genres": sorted(requested_genres),
            "difficulty": difficulty,
            "quality": quality,
            "min_measures": min_measures,
            "max_measures": max_measures,
            "limit": limit,
            "official_pdmx_subsets": ["no_license_conflict", "all_valid"],
            "official_pdmx_preferred_arrangement": True,
        },
        "difficulty_policy": {
            "score": "0.65 * note-density-percentile + 0.35 * normalized-source-complexity",
            "easy": "score < 0.30",
            "intermediate": "0.30 <= score < 0.75",
            "advanced": "score >= 0.75",
        },
        "quality_policy": {
            "score": (
                "2 * named-controlled-genre + source-classical + known-composer + professional-user + "
                "well-rated + 0.5 * viewed + 0.5 * favorited + 0.5 * best-path"
            ),
            "metadata-curated": "score >= 4.0",
            "canonical-core": (
                "metadata-curated and canonical-classical-composer and no explicit arrangement marker in title"
            ),
            "curriculum": {
                "broad": ["score-dsl-autoencode"],
                "metadata-curated": [
                    "score-dsl-autoencode",
                    "section-expand-16-64",
                    "masked-span-inpaint",
                    "ending-complete",
                    "section-variation-revise",
                ],
                "canonical-core": [
                    "score-dsl-autoencode",
                    "section-expand-16-64",
                    "whole-piece-generate",
                    "masked-span-inpaint",
                    "ending-complete",
                    "recapitulation-revise",
                    "section-variation-revise",
                ],
            },
        },
        "artifacts": {"manifest": manifest_path.name},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _subset_enabled(row: Dict[str, str], name: str) -> bool:
    for key in (f"subset:{name}", name):
        if key in row:
            return _truthy(row.get(key, ""))
    return True


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() not in ("", "0", "false", "no", "none", "nan")


def _is_solo_piano(row: Dict[str, str]) -> bool:
    if row.get("is_solo_piano"):
        return _truthy(row["is_solo_piano"])
    tracks = str(row.get("tracks", "")).strip()
    if tracks:
        # PDMX encodes General MIDI programs with "-" separators. Program 0 is piano.
        return tracks == "0"
    text = " ".join(
        str(row.get(key, ""))
        for key in ("instrumentation", "instruments", "instrument", "is_piano")
    ).lower()
    return "piano" in text and not any(
        word in text for word in ("violin", "cello", "flute", "voice", "orchestra", "guitar", "drum")
    )


def _is_preferred_arrangement(row: Dict[str, str]) -> bool:
    if "is_best_unique_arrangement" in row:
        return _truthy(row.get("is_best_unique_arrangement", ""))
    if "subset:deduplicated" in row:
        return _truthy(row.get("subset:deduplicated", ""))
    return True


def _genre_label(row: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
    fields = ("title", "song_name", "subtitle", "tags", "groups", "genres")
    normalized = {field: _normalize_text(row.get(field, "")) for field in fields}
    for genre, patterns in GENRE_PATTERNS:
        for field in fields:
            for pattern in patterns:
                if _phrase_present(normalized[field], _normalize_text(pattern)):
                    return genre, {"field": field, "matched": pattern}
    source_genres = normalized["genres"].split()
    if "classical" in source_genres:
        return "classical-piano", {"field": "genres", "matched": "classical"}
    return "unclassified-piano", {"field": "", "matched": ""}


def _form_label(row: Dict[str, str], genre: str) -> Tuple[str, Dict[str, str]]:
    title = _normalize_text(row.get("title") or row.get("song_name") or "")
    for pattern in ("rondo", "rondeau"):
        if _phrase_present(title, pattern):
            return "rondo", {"field": "title", "matched": pattern, "policy": "title-keyword"}
    if genre == "theme-and-variations":
        return "theme-and-variations", {"field": "genre", "matched": genre, "policy": "genre-heuristic"}
    if genre == "nocturne":
        return "ABA", {"field": "genre", "matched": genre, "policy": "genre-heuristic"}
    if genre == "minuet":
        return "ternary", {"field": "genre", "matched": genre, "policy": "genre-heuristic"}
    return "free-sectional", {"field": "", "matched": "", "policy": "conservative-default"}


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    ascii_text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text))


def _phrase_present(text: str, phrase: str) -> bool:
    return bool(text and phrase and re.search(rf"(?:^| )({re.escape(phrase)})(?: |$)", text))


def _difficulty_scores(rows: Sequence[Dict[str, str]]) -> List[float]:
    densities = [_optional_float(row.get("notes_per_bar")) or 0.0 for row in rows]
    sorted_densities = sorted(densities)
    return [
        round(0.65 * _percentile_rank(sorted_densities, density) + 0.35 * _normalized_complexity(row), 6)
        for row, density in zip(rows, densities)
    ]


def _percentile_rank(sorted_values: Sequence[float], value: float) -> float:
    if len(sorted_values) < 2:
        return 0.5
    below = bisect_left(sorted_values, value)
    equal = bisect_right(sorted_values, value) - below
    return (below + max(0, equal - 1) / 2) / (len(sorted_values) - 1)


def _normalized_complexity(row: Dict[str, str]) -> float:
    complexity = _optional_float(row.get("complexity"))
    if complexity is None:
        return 0.5
    return max(0.0, min(1.0, complexity / 3.0))


def _difficulty_label(score: float) -> str:
    if score < 0.30:
        return "easy"
    if score < 0.75:
        return "intermediate"
    return "advanced"


def _difficulty_evidence(row: Dict[str, str]) -> Dict[str, Optional[float]]:
    return {
        "source_complexity": _optional_float(row.get("complexity")),
        "notes_per_bar": _optional_float(row.get("notes_per_bar")),
        "n_notes": _optional_float(row.get("n_notes")),
        "measure_count": _optional_float(row.get("song_length.bars")),
    }


def _quality_label(row: Dict[str, str], genre: str) -> Tuple[str, float, Dict[str, Any]]:
    source_genres = _normalize_text(row.get("genres", "")).split()
    rating = _optional_float(row.get("rating")) or 0.0
    views = _optional_float(row.get("n_views")) or 0.0
    favorites = _optional_float(row.get("n_favorites")) or 0.0
    evidence = {
        "named_controlled_genre": genre not in GENERIC_GENRES,
        "source_classical": "classical" in source_genres,
        "known_composer": _known_composer(row),
        "professional_user": _truthy(row.get("is_user_pro", "")),
        "well_rated": _truthy(row.get("is_rated", "")) and rating >= 4.0,
        "viewed": views >= 100,
        "favorited": favorites >= 2,
        "best_path": _truthy(row.get("is_best_path", "")),
        "canonical_classical_composer": _canonical_classical_composer(row),
        "piano_native_title": _piano_native_title(row),
    }
    score = (
        2.0 * evidence["named_controlled_genre"]
        + evidence["source_classical"]
        + evidence["known_composer"]
        + evidence["professional_user"]
        + evidence["well_rated"]
        + 0.5 * evidence["viewed"]
        + 0.5 * evidence["favorited"]
        + 0.5 * evidence["best_path"]
    )
    if score >= 4.0 and evidence["canonical_classical_composer"] and evidence["piano_native_title"]:
        tier = "canonical-core"
    elif score >= 4.0:
        tier = "metadata-curated"
    else:
        tier = "broad"
    return tier, round(score, 2), evidence


def _quality_matches(tier: str, requested: str) -> bool:
    if requested == "metadata-curated":
        return tier in ("metadata-curated", "canonical-core")
    return tier == requested


def _tasks_for_quality(tier: str) -> List[str]:
    tasks = ["score-dsl-autoencode"]
    if tier in ("metadata-curated", "canonical-core"):
        tasks.extend(["section-expand-16-64", "masked-span-inpaint", "ending-complete", "section-variation-revise"])
    if tier == "canonical-core":
        tasks.extend(["whole-piece-generate", "recapitulation-revise"])
    return tasks


def _canonical_classical_composer(row: Dict[str, str]) -> bool:
    composer = _normalize_text(row.get("composer") or row.get("composer_name") or "")
    return any(_phrase_present(composer, pattern) for pattern in CANONICAL_CLASSICAL_COMPOSERS)


def _piano_native_title(row: Dict[str, str]) -> bool:
    title = _normalize_text(row.get("title") or row.get("song_name") or "")
    return not any(_phrase_present(title, _normalize_text(marker)) for marker in ARRANGEMENT_TITLE_MARKERS)


def _known_composer(row: Dict[str, str]) -> bool:
    composer = _normalize_text(row.get("composer") or row.get("composer_name") or "")
    placeholders = {
        "",
        "na",
        "anon",
        "anonymous",
        "unknown",
        "traditional",
        "trad",
        "untitled",
        "march",
        "reel",
        "jig",
        "air",
    }
    return composer not in placeholders and len(composer) >= 3


def _work_id(row: Dict[str, str]) -> str:
    official_identity = str(row.get("best_unique_arrangement", "")).strip().lower()
    composer = str(row.get("composer") or row.get("composer_name") or "").strip().lower()
    title = str(row.get("title") or row.get("song_name") or "").strip().lower()
    fallback_path = _score_path(row).strip().lower()
    identity = official_identity or (f"{composer}|{title}" if composer or title else fallback_path)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _score_path(row: Dict[str, str]) -> str:
    return str(
        row.get("mxl")
        or row.get("mxl_path")
        or row.get("musicxml_path")
        or row.get("path")
        or ""
    )


def _split(work_id: str) -> str:
    bucket = int(work_id[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "valid"
    return "test"


def _measure_count(row: Dict[str, str]) -> Optional[int]:
    for key in ("measure_count", "n_measures", "measures", "num_measures", "song_length.bars"):
        value = _optional_float(row.get(key))
        if value is not None:
            return int(value)
    return None


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "NA"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a labeled no-license-conflict PDMX training manifest")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--genres", help="Comma-separated inferred genres to retain")
    parser.add_argument("--difficulty", choices=DIFFICULTY_LABELS)
    parser.add_argument("--quality", choices=QUALITY_FILTERS)
    parser.add_argument("--min-measures", type=int)
    parser.add_argument("--max-measures", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    genres = tuple(part.strip() for part in args.genres.split(",") if part.strip()) if args.genres else ()
    summary = prepare_manifest(
        args.metadata_csv,
        args.output_dir,
        genres=genres,
        difficulty=args.difficulty,
        quality=args.quality,
        min_measures=args.min_measures,
        max_measures=args.max_measures,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
