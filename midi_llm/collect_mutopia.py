"""Collect conservatively filtered Mutopia piano sources with provenance."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .prepare_pdmx import (
    CANONICAL_CLASSICAL_COMPOSERS,
    _composer_period,
    _composer_style_label,
    _form_label,
    _genre_label,
    _normalize_text,
    _split,
    _style_tags,
)


MUTOPIA_REPOSITORY = "https://github.com/MutopiaProject/MutopiaProject"
HISTORICAL_KEYBOARD_INSTRUMENTS = {"clavichord", "clavier", "harpsichord", "piano"}
LICENSE_URLS = {
    "public-domain": "https://creativecommons.org/publicdomain/mark/1.0/",
    "cc-by-2.5": "https://creativecommons.org/licenses/by/2.5/",
    "cc-by-3.0": "https://creativecommons.org/licenses/by/3.0/",
    "cc-by-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-sa-2.5": "https://creativecommons.org/licenses/by-sa/2.5/",
    "cc-by-sa-3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
    "cc-by-sa-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
}


def collect_mutopia_sources(source_root: Path | str, output_dir: Path | str) -> Dict[str, Any]:
    """Write all filtered candidates and one preferred source per Mutopia work."""

    source_root = Path(source_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = _git_commit(source_root)
    candidates = []
    for path in sorted((source_root / "ftp").rglob("*.ly")):
        row = _source_row(path, source_root, commit)
        if row is not None:
            candidates.append(row)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[row["variant_group_id"]].append(row)
    selected = []
    for variants in grouped.values():
        variants.sort(key=_variant_selection_key)
        selected.append({**variants[0], "variant_count": len(variants), "alternate_paths": [row["path"] for row in variants[1:]]})
    selected.sort(key=lambda row: row["work_id"])
    _write_jsonl(output_dir / "mutopia_piano_candidates.jsonl", candidates)
    _write_jsonl(output_dir / "mutopia_piano_manifest.jsonl", selected)
    summary = {
        "pipeline": "mutopia-piano-source-collection-v1",
        "source_dataset": "Mutopia",
        "repository": MUTOPIA_REPOSITORY,
        "repository_commit": commit,
        "lilypond_files_scanned": sum(1 for _ in (source_root / "ftp").rglob("*.ly")),
        "filtered_keyboard_candidates": len(candidates),
        "selected_unique_works": len(selected),
        "duplicate_variants_removed": len(candidates) - len(selected),
        "instrument_tier_counts": _count(selected, "instrument_tier"),
        "composer_style_counts": _count(selected, "composer_style"),
        "composer_period_counts": _count(selected, "composer_period"),
        "genre_counts": _count(selected, "genre"),
        "source_license_counts": _count(selected, "source_license"),
        "license_evidence_counts": _count(selected, "license_evidence"),
        "artifacts": {
            "candidates": "mutopia_piano_candidates.jsonl",
            "manifest": "mutopia_piano_manifest.jsonl",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _source_row(path: Path, source_root: Path, commit: str) -> Dict[str, Any] | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    metadata = _metadata(text)
    instrument = metadata.get("mutopiainstrument") or metadata.get("instrument") or ""
    instrument_tier = _instrument_tier(instrument)
    if instrument_tier is None:
        return None
    relative = path.relative_to(source_root).as_posix()
    sha256 = _sha256(path)
    footer = metadata.get("footer") or ""
    variant_group_id = footer or _fallback_group_id(metadata, relative)
    composer_style = _composer_style(metadata)
    composer_period = _composer_period(composer_style)
    title = metadata.get("mutopiatitle") or metadata.get("title") or path.stem
    source_style = metadata.get("style") or ""
    genre, genre_evidence = _genre_label({"title": title, "genres": source_style})
    form, form_evidence = _form_label({"title": title}, genre)
    source_license, source_license_url, license_evidence = _license(text, metadata)
    work_id = hashlib.sha256(f"Mutopia|{variant_group_id}".encode("utf-8")).hexdigest()[:20]
    return {
        "work_id": work_id,
        "variant_group_id": variant_group_id,
        "split": _split(work_id),
        "source_dataset": "Mutopia",
        "source_repository": MUTOPIA_REPOSITORY,
        "source_repository_commit": commit,
        "source_url": f"{MUTOPIA_REPOSITORY}/blob/{commit}/{relative}",
        "path": relative,
        "score_format": "lilypond",
        "source_sha256": sha256,
        "source_license": source_license,
        "source_license_url": source_license_url,
        "license_evidence": license_evidence,
        "title": title,
        "composer": metadata.get("composer") or metadata.get("mutopiacomposer") or "",
        "mutopia_composer": metadata.get("mutopiacomposer") or "",
        "composer_style": composer_style,
        "composer_period": composer_period,
        "style_tags": _style_tags(composer_style, composer_period, genre),
        "genre": genre,
        "genre_evidence": genre_evidence,
        "form": form,
        "form_evidence": form_evidence,
        "difficulty": "unreviewed",
        "quality_tier": "supplemental-unreviewed",
        "instrument": instrument,
        "instrument_tier": instrument_tier,
        "source_edition": metadata.get("source") or "",
        "maintainer": metadata.get("maintainer") or "",
        "mutopia_footer": footer,
        "tasks": [],
    }


def _metadata(text: str) -> Dict[str, str]:
    fields = (
        "title",
        "composer",
        "instrument",
        "mutopiatitle",
        "mutopiacomposer",
        "mutopiainstrument",
        "mutopiaopus",
        "source",
        "style",
        "license",
        "maintainer",
        "footer",
    )
    values = {}
    for field in fields:
        match = re.search(rf"^\s*{re.escape(field)}\s*=\s*\"([^\"]*)\"", text, re.MULTILINE)
        if match:
            values[field] = match.group(1).strip()
    return values


def _instrument_tier(instrument: str) -> str | None:
    normalized = _normalize_text(instrument)
    if normalized == "piano":
        return "solo-piano"
    parts = {part for part in normalized.split() if part not in {"and"}}
    if parts and parts <= HISTORICAL_KEYBOARD_INSTRUMENTS and "piano" in parts:
        return "historical-keyboard-compatible"
    if normalized in HISTORICAL_KEYBOARD_INSTRUMENTS:
        return "historical-keyboard-compatible"
    return None


def _composer_style(metadata: Dict[str, str]) -> str:
    style, _ = _composer_style_label({"composer": metadata.get("composer", "")})
    if style != "unclassified-composer":
        return style
    code = _normalize_text(metadata.get("mutopiacomposer", "")).replace(" ", "")
    for candidate in CANONICAL_CLASSICAL_COMPOSERS:
        if code.startswith(candidate):
            return candidate
    return "unclassified-composer"


def _license(text: str, metadata: Dict[str, str]) -> Tuple[str, str, str]:
    combined = f"{metadata.get('license', '')}\n{text}".casefold()
    match = re.search(r"licenses/(by(?:-sa)?)/([0-9.]+)", combined)
    if match:
        key = f"cc-{match.group(1)}-{match.group(2).rstrip('.')}"
        return key, LICENSE_URLS.get(key, f"https://creativecommons.org/licenses/{match.group(1)}/{match.group(2)}/"), "declared-in-lilypond-source"
    if "public domain" in combined or "licenses/publicdomain" in combined:
        return "public-domain", LICENSE_URLS["public-domain"], "declared-in-lilypond-source"
    normalized = _normalize_text(metadata.get("license", ""))
    if normalized == "cc by sa":
        return "cc-by-sa-unspecified", "", "declared-in-lilypond-source"
    return "legacy-mutopia-unspecified", "", "requires-manual-license-review"


def _fallback_group_id(metadata: Dict[str, str], relative: str) -> str:
    title = _normalize_text(metadata.get("mutopiatitle") or metadata.get("title") or "")
    composer = _normalize_text(metadata.get("mutopiacomposer") or metadata.get("composer") or "")
    opus = _normalize_text(metadata.get("mutopiaopus") or "")
    return f"path:{relative}" if not (title or composer or opus) else f"metadata:{composer}|{title}|{opus}"


def _variant_selection_key(row: Dict[str, Any]) -> Tuple[int, int, int, str]:
    name = Path(row["path"]).stem.casefold()
    variant_markers = ("midi", "-mid", "_mid", "letter", "a4", "book", "-all", "_all")
    return (
        sum(marker in name for marker in variant_markers),
        row["path"].count("/"),
        len(row["path"]),
        row["path"],
    )


def _git_commit(source_root: Path) -> str:
    head = source_root / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = source_root / ".git" / value.removeprefix("ref: ")
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()
        packed = source_root / ".git" / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(f" {value.removeprefix('ref: ')}"):
                    return line.split()[0]
        return "unknown"
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _count(rows: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "missing") for row in rows).items()))


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect conservatively filtered Mutopia keyboard sources")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(collect_mutopia_sources(args.source_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
