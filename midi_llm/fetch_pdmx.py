"""Download official PDMX resources from Zenodo with checksums and resume support."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set
from urllib.request import Request, urlopen


ZENODO_RECORD = "15571083"
ZENODO_BASE = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"
RESOURCES: Dict[str, Dict[str, Any]] = {
    "PDMX.csv": {"size": 225399738, "md5": "30392ccf38bb63ce70e7afae70f9c88c"},
    "subset_paths.tar.gz": {"size": 29258714, "md5": "092eee416ece8060f77d08575b94a43d"},
    "mxl.tar.gz": {"size": 1894335797, "md5": "49ffd75ecf5489c0be6d41182eb11ff7"},
    "mid.tar.gz": {"size": 214395208, "md5": "d920a21b2fcd99a56d9c381b39debbb2"},
}
PARALLEL_CHUNK_BYTES = 32 * 1024 * 1024


def fetch_resources(
    output_dir: Path | str,
    *,
    resources: Sequence[str] = ("PDMX.csv", "subset_paths.tar.gz"),
    extract: Sequence[str] = (),
    extract_members: Optional[Mapping[str, Set[str]]] = None,
    connections: int = 1,
) -> Dict[str, Any]:
    """Download selected official files and optionally extract tar archives."""

    if connections < 1:
        raise ValueError("connections must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in resources:
        if name not in RESOURCES:
            raise ValueError(f"Unknown PDMX resource: {name}")
        destination = output_dir / name
        _download(
            f"{ZENODO_BASE}/{name}/content",
            destination,
            RESOURCES[name]["size"],
            connections=connections,
        )
        checksum = _md5(destination)
        if checksum != RESOURCES[name]["md5"]:
            raise ValueError(f"Checksum mismatch for {destination}: {checksum}")
        results.append({"resource": name, "path": str(destination.resolve()), "md5": checksum})
    extracted = []
    for name in extract:
        if name not in resources:
            raise ValueError(f"Cannot extract resource that was not fetched: {name}")
        archive = output_dir / name
        if not tarfile.is_tarfile(archive):
            raise ValueError(f"Resource is not a tar archive: {archive}")
        members = (extract_members or {}).get(name)
        count = _extract_tar(archive, output_dir, members)
        extracted.append({"resource": name, "members": count, "selective": members is not None})
    summary = {
        "pipeline": "pdmx-official-zenodo-fetch-v1",
        "record": ZENODO_RECORD,
        "connections": connections,
        "resources": results,
        "extracted": extracted,
    }
    (output_dir / "fetch_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _download(url: str, destination: Path, expected_size: int, *, connections: int = 1) -> None:
    current_size = destination.stat().st_size if destination.exists() else 0
    if current_size == expected_size:
        return
    if current_size > expected_size:
        destination.unlink()
        current_size = 0
    if connections > 1:
        _download_parallel(url, destination, current_size, expected_size, connections)
        return
    request = Request(url, headers={"Range": f"bytes={current_size}-"} if current_size else {})
    with urlopen(request) as response:
        resumed = current_size and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        with destination.open(mode) as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    actual_size = destination.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"Size mismatch for {destination}: expected {expected_size}, got {actual_size}")


def _download_parallel(url: str, destination: Path, current_size: int, expected_size: int, connections: int) -> None:
    if connections < 1:
        raise ValueError("connections must be positive")
    if current_size == expected_size:
        return
    parts_dir = destination.parent / f".{destination.name}.parts"
    parts_dir.mkdir(exist_ok=True)
    existing = _normalize_existing_parts(parts_dir, current_size, expected_size)
    gaps = _missing_ranges(current_size, expected_size - 1, [(start, end) for start, end, _ in existing])
    pending = [
        (start, end, _part_path(parts_dir, start, end))
        for gap_start, gap_end in gaps
        for start, end in _split_range(gap_start, gap_end, PARALLEL_CHUNK_BYTES)
    ]
    ranges = sorted(existing + pending)
    _validate_coverage(ranges, current_size, expected_size)
    with ThreadPoolExecutor(max_workers=connections) as pool:
        list(pool.map(lambda item: _download_range(url, *item), ranges))
    with destination.open("ab") as output:
        for start, end, part in ranges:
            expected_part_size = end - start + 1
            if part.stat().st_size != expected_part_size:
                raise ValueError(f"Size mismatch for {part}")
            with part.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    output.write(chunk)
            part.unlink()
    parts_dir.rmdir()
    actual_size = destination.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"Size mismatch for {destination}: expected {expected_size}, got {actual_size}")


def _normalize_existing_parts(parts_dir: Path, lower: int, upper: int):
    parts = []
    for part in parts_dir.glob("*.part"):
        start, end = _part_range(part)
        size = part.stat().st_size
        if end < lower:
            part.unlink()
            continue
        if start < lower or end >= upper:
            raise ValueError(f"Part falls outside expected range: {part}")
        expected_size = end - start + 1
        if size > expected_size:
            raise ValueError(f"Part is larger than its declared range: {part}")
        if not size:
            part.unlink()
            continue
        actual_end = start + size - 1
        if actual_end != end:
            normalized = _part_path(parts_dir, start, actual_end)
            if normalized.exists() and normalized != part:
                raise ValueError(f"Part normalization collision: {normalized}")
            part.rename(normalized)
            part = normalized
        parts.append((start, actual_end, part))
    parts.sort()
    _validate_non_overlapping(parts)
    return parts


def _missing_ranges(lower: int, upper: int, covered):
    gaps = []
    cursor = lower
    for start, end in sorted(covered):
        if start > cursor:
            gaps.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
    if cursor <= upper:
        gaps.append((cursor, upper))
    return gaps


def _split_range(start: int, end: int, chunk_size: int):
    while start <= end:
        chunk_end = min(end, start + chunk_size - 1)
        yield start, chunk_end
        start = chunk_end + 1


def _validate_coverage(ranges, lower: int, upper: int) -> None:
    cursor = lower
    for start, end, _ in ranges:
        if start != cursor:
            raise ValueError(f"Parallel download coverage is not contiguous at byte {cursor}")
        cursor = end + 1
    if cursor != upper:
        raise ValueError(f"Parallel download coverage ends at byte {cursor}, expected {upper}")


def _validate_non_overlapping(ranges) -> None:
    previous_end = -1
    for start, end, part in ranges:
        if start <= previous_end:
            raise ValueError(f"Parallel download parts overlap at {part}")
        previous_end = end


def _part_path(parts_dir: Path, start: int, end: int) -> Path:
    return parts_dir / f"{start:020d}-{end:020d}.part"


def _part_range(path: Path):
    raw_start, raw_end = path.name.removesuffix(".part").split("-", maxsplit=1)
    return int(raw_start), int(raw_end)


def _download_range(url: str, start: int, end: int, destination: Path) -> None:
    expected_size = end - start + 1
    if destination.exists() and destination.stat().st_size > expected_size:
        destination.unlink()
    for _ in range(4):
        current_size = destination.stat().st_size if destination.exists() else 0
        if current_size == expected_size:
            return
        request = Request(url, headers={"Range": f"bytes={start + current_size}-{end}"})
        with urlopen(request) as response:
            if getattr(response, "status", None) != 206:
                raise ValueError(f"Server did not honor Range request for {start + current_size}-{end}")
            with destination.open("ab") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
    actual_size = destination.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"Size mismatch for {destination}: expected {expected_size}, got {actual_size}")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_tar(archive: Path, output_dir: Path, members: Optional[Set[str]] = None) -> int:
    root = output_dir.resolve()
    with tarfile.open(archive) as handle:
        requested = {_normalize_archive_path(member) for member in members} if members is not None else None
        selected = [
            member
            for member in handle.getmembers()
            if requested is None or _normalize_archive_path(member.name) in requested
        ]
        if requested is not None:
            found = {_normalize_archive_path(member.name) for member in selected}
            missing = sorted(requested - found)
            if missing:
                raise ValueError(f"Archive is missing {len(missing)} requested members; first missing: {missing[0]}")
        for member in selected:
            destination = (output_dir / member.name).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(f"Archive member escapes output directory: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not supported: {member.name}")
        try:
            handle.extractall(output_dir, members=selected, filter="fully_trusted")
        except TypeError:
            # Python 3.11 does not expose the tar extraction filter argument.
            handle.extractall(output_dir, members=selected)
    return sum(member.isfile() for member in selected)


def _normalize_archive_path(value: str) -> str:
    return value.replace("\\", "/").removeprefix("./")


def _manifest_paths(path: Path | str) -> Set[str]:
    paths = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line).get("path", "")
            if value:
                paths.add(_normalize_archive_path(str(value)))
    return paths


def _resources_from_args(args: argparse.Namespace) -> Iterable[str]:
    yield "PDMX.csv"
    yield "subset_paths.tar.gz"
    if args.include_mxl:
        yield "mxl.tar.gz"
    if args.include_midi:
        yield "mid.tar.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch official PDMX resources from Zenodo")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-mxl", action="store_true")
    parser.add_argument("--include-midi", action="store_true")
    parser.add_argument("--extract-subsets", action="store_true")
    parser.add_argument("--extract-mxl", action="store_true")
    parser.add_argument("--mxl-manifest", help="Extract only MXL paths referenced by this JSONL manifest")
    parser.add_argument("--connections", type=int, default=4)
    args = parser.parse_args()
    resources = tuple(_resources_from_args(args))
    extract = []
    if args.extract_subsets:
        extract.append("subset_paths.tar.gz")
    if args.extract_mxl:
        if not args.include_mxl:
            parser.error("--extract-mxl requires --include-mxl")
        extract.append("mxl.tar.gz")
    extract_members = {}
    if args.mxl_manifest:
        if not args.extract_mxl:
            parser.error("--mxl-manifest requires --extract-mxl")
        extract_members["mxl.tar.gz"] = _manifest_paths(args.mxl_manifest)
    print(
        json.dumps(
            fetch_resources(
                args.output_dir,
                resources=resources,
                extract=extract,
                extract_members=extract_members,
                connections=args.connections,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
