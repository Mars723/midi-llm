"""Package and verify the minimal ScoreDSL dataset needed by a GPU host."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from typing import Any, Dict, Iterable, List


DATASET_FILES = (
    "train.jsonl",
    "valid.jsonl",
    "test.jsonl",
    "summary.json",
    "materialization_errors.jsonl",
    "oversized_examples.jsonl",
)
MANIFEST_NAME = "bundle_manifest.json"
DATASET_ARCHIVE_DIR = "model_dataset"


def package_cloud_dataset(dataset_dir: Path | str, output: Path | str) -> Dict[str, Any]:
    """Create a checksum-addressed archive without local score caches."""

    dataset_dir = Path(dataset_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for name in DATASET_FILES:
        path = dataset_dir / name
        if not path.is_file():
            raise ValueError(f"Required model dataset file is missing: {path}")
        files.append(
            {
                "path": f"{DATASET_ARCHIVE_DIR}/{name}",
                "bytes": path.stat().st_size,
                "sha256": _sha256_path(path),
            }
        )
    manifest = {
        "pipeline": "score-first-cloud-bundle-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": _git_commit(),
        "dataset_archive_dir": DATASET_ARCHIVE_DIR,
        "excluded_local_cache": "scores/",
        "files": files,
    }
    encoded_manifest = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    with tarfile.open(output, "w:gz") as archive:
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(encoded_manifest)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(encoded_manifest))
        for entry in files:
            archive.add(dataset_dir / Path(entry["path"]).name, arcname=entry["path"], recursive=False)
    result = {
        **manifest,
        "archive": str(output.resolve()),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": _sha256_path(output),
    }
    return result


def verify_cloud_bundle(bundle: Path | str) -> Dict[str, Any]:
    """Verify archive paths and payload digests before extraction."""

    bundle = Path(bundle)
    with tarfile.open(bundle) as archive:
        members = archive.getmembers()
        _validate_members(members)
        manifest_member = archive.getmember(MANIFEST_NAME)
        extracted = archive.extractfile(manifest_member)
        if extracted is None:
            raise ValueError(f"Bundle manifest is not a file: {MANIFEST_NAME}")
        manifest = json.loads(extracted.read())
        members_by_name = {member.name: member for member in members}
        checked = []
        for entry in manifest["files"]:
            path = entry["path"]
            member = members_by_name.get(path)
            if member is None or not member.isfile():
                raise ValueError(f"Bundle is missing dataset file: {path}")
            payload = archive.extractfile(member)
            if payload is None:
                raise ValueError(f"Bundle dataset entry is not readable: {path}")
            digest = _sha256_stream(payload)
            if digest != entry["sha256"]:
                raise ValueError(f"Checksum mismatch for bundled dataset file: {path}")
            if member.size != entry["bytes"]:
                raise ValueError(f"Size mismatch for bundled dataset file: {path}")
            checked.append(path)
    return {
        "pipeline": manifest["pipeline"],
        "bundle": str(bundle.resolve()),
        "archive_bytes": bundle.stat().st_size,
        "archive_sha256": _sha256_path(bundle),
        "verified_files": checked,
        "source_git_commit": manifest.get("source_git_commit"),
    }


def extract_cloud_bundle(bundle: Path | str, output_dir: Path | str) -> Dict[str, Any]:
    """Verify and safely extract the model dataset on a remote host."""

    verification = verify_cloud_bundle(bundle)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle) as archive:
        members = archive.getmembers()
        _validate_members(members)
        try:
            archive.extractall(output_dir, members=members, filter="fully_trusted")
        except TypeError:
            archive.extractall(output_dir, members=members)
    return {
        **verification,
        "extracted_to": str(output_dir.resolve()),
        "dataset_dir": str((output_dir / DATASET_ARCHIVE_DIR).resolve()),
    }


def _validate_members(members: Iterable[tarfile.TarInfo]) -> None:
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Archive member escapes output directory: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"Archive links are not supported: {member.name}")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256_path(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package or extract a ScoreDSL cloud training dataset")
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package", help="Create a checksum-addressed tar.gz archive")
    package.add_argument("--dataset-dir", required=True)
    package.add_argument("--output", required=True)
    verify = commands.add_parser("verify", help="Verify a cloud dataset archive")
    verify.add_argument("--bundle", required=True)
    extract = commands.add_parser("extract", help="Verify and safely extract a cloud dataset archive")
    extract.add_argument("--bundle", required=True)
    extract.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "package":
        result = package_cloud_dataset(args.dataset_dir, args.output)
    elif args.command == "verify":
        result = verify_cloud_bundle(args.bundle)
    else:
        result = extract_cloud_bundle(args.bundle, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
