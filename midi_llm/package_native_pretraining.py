"""Package the upstream-native classical pretraining dataset for a GPU host."""

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


BUNDLE_MANIFEST = "bundle_manifest.json"
PREFLIGHT_JSON = "preflight_report.json"
PREFLIGHT_MD = "preflight_report.md"
DATASET_DIR = "training_dataset"
REQUIRED_DATASET_FILES = (
    "train.jsonl",
    "valid.jsonl",
    "test.jsonl",
    "summary.json",
    "materialization_errors.jsonl",
    "oversized_complete_pieces.jsonl",
)


def package_native_pretraining_bundle(
    dataset_dir: Path | str,
    output: Path | str,
    *,
    mutopia_summary: Path | str | None = None,
    pianocore_summary: Path | str | None = None,
    asap_summary: Path | str | None = None,
    maestro_summary: Path | str | None = None,
    gallery_index: Path | str | None = None,
) -> Dict[str, Any]:
    """Create a checksum-addressed tarball for native MIDI QLoRA pretraining."""

    dataset_dir = Path(dataset_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_files = _dataset_entries(dataset_dir)
    report = _preflight_report(
        dataset_dir,
        mutopia_summary=mutopia_summary,
        pianocore_summary=pianocore_summary,
        asap_summary=asap_summary,
        maestro_summary=maestro_summary,
        gallery_index=gallery_index,
    )
    manifest = {
        "pipeline": "native-classical-pretraining-bundle-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": _git_commit(),
        "default_training_dataset": DATASET_DIR,
        "files": dataset_files,
        "preflight_report": PREFLIGHT_JSON,
        "training_policy": {
            "default_dataset": "PDMX composition-core upstream-native tokens",
            "mutopia_review_tokens_in_default_training": False,
            "non_commercial_research_sources_in_default_training": False,
            "start_training_without_gpu": False,
        },
    }
    encoded_manifest = _json_bytes(manifest)
    encoded_report = _json_bytes(report)
    encoded_markdown = _markdown_report(report).encode("utf-8")
    with tarfile.open(output, "w:gz") as archive:
        _add_bytes(archive, BUNDLE_MANIFEST, encoded_manifest)
        _add_bytes(archive, PREFLIGHT_JSON, encoded_report)
        _add_bytes(archive, PREFLIGHT_MD, encoded_markdown)
        for entry in dataset_files:
            archive.add(dataset_dir / entry["source_relative_path"], arcname=entry["path"], recursive=False)
    result = {
        **manifest,
        "archive": str(output.resolve()),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": _sha256_path(output),
        "preflight": report,
    }
    (output.with_suffix(output.suffix + ".manifest.json")).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def verify_native_pretraining_bundle(bundle: Path | str) -> Dict[str, Any]:
    """Verify archive structure, file digests, and referenced token coverage."""

    bundle = Path(bundle)
    with tarfile.open(bundle) as archive:
        members = archive.getmembers()
        _validate_members(members)
        manifest = _read_json_member(archive, BUNDLE_MANIFEST)
        report = _read_json_member(archive, PREFLIGHT_JSON)
        members_by_name = {member.name: member for member in members}
        checked = []
        for entry in manifest["files"]:
            member = members_by_name.get(entry["path"])
            if member is None or not member.isfile():
                raise ValueError(f"Bundle is missing dataset file: {entry['path']}")
            payload = archive.extractfile(member)
            if payload is None:
                raise ValueError(f"Bundle entry is not readable: {entry['path']}")
            digest = _sha256_stream(payload)
            if digest != entry["sha256"]:
                raise ValueError(f"Checksum mismatch for bundled file: {entry['path']}")
            if member.size != entry["bytes"]:
                raise ValueError(f"Size mismatch for bundled file: {entry['path']}")
            checked.append(entry["path"])
        expected_tokens = _token_paths_from_archive(archive, members_by_name)
        missing_tokens = sorted(path for path in expected_tokens if path not in members_by_name)
        if missing_tokens:
            raise ValueError(f"Bundle is missing {len(missing_tokens)} referenced token files; first: {missing_tokens[0]}")
    return {
        "pipeline": manifest["pipeline"],
        "bundle": str(bundle.resolve()),
        "archive_bytes": bundle.stat().st_size,
        "archive_sha256": _sha256_path(bundle),
        "verified_files": len(checked),
        "verified_referenced_tokens": len(expected_tokens),
        "ready_for_gpu_smoke": bool(report["gate"]["ready_for_gpu_smoke"]),
        "source_git_commit": manifest.get("source_git_commit"),
    }


def _dataset_entries(dataset_dir: Path) -> List[Dict[str, Any]]:
    entries = []
    for name in REQUIRED_DATASET_FILES:
        path = dataset_dir / name
        if not path.is_file():
            raise ValueError(f"Required native dataset file is missing: {path}")
        entries.append(_entry(path, name))
    token_paths = _token_paths_from_dataset(dataset_dir)
    for token in token_paths:
        entries.append(_entry(dataset_dir / token, token))
    return entries


def _entry(path: Path, source_relative_path: str) -> Dict[str, Any]:
    return {
        "path": f"{DATASET_DIR}/{source_relative_path}",
        "source_relative_path": source_relative_path,
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def _preflight_report(dataset_dir: Path, **paths: Path | str | None) -> Dict[str, Any]:
    summary = json.loads((dataset_dir / "summary.json").read_text(encoding="utf-8"))
    errors = _read_jsonl(dataset_dir / "materialization_errors.jsonl")
    oversized = _read_jsonl(dataset_dir / "oversized_complete_pieces.jsonl")
    token_paths = _token_paths_from_dataset(dataset_dir)
    references = {key: _load_optional_summary(value) for key, value in paths.items()}
    ready = (
        summary.get("examples_written", 0) > 0
        and summary.get("examples_failed", 1) == 0
        and summary.get("segment_event_token_counts", {}).get("max", 999999) <= 7800
        and not errors
    )
    return {
        "pipeline": "native-classical-pretraining-preflight-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir.resolve()),
        "gate": {
            "ready_for_gpu_smoke": ready,
            "ready_for_gpu_pilot": ready,
            "reason": "PDMX composition-core native token dataset passes parsing, split, and context checks.",
            "not_started": "GPU training has intentionally not been launched.",
        },
        "default_training_dataset": {
            "source": "PDMX",
            "policy": "commercial composition backbone candidate",
            "examples_written": summary.get("examples_written"),
            "native_segments_written": summary.get("native_segments_written"),
            "split_counts": summary.get("split_counts"),
            "composer_style_counts": summary.get("composer_style_counts"),
            "genre_counts": summary.get("genre_counts"),
            "segment_event_token_counts": summary.get("segment_event_token_counts"),
            "complete_piece_event_token_counts": summary.get("complete_piece_event_token_counts"),
            "referenced_token_files": len(token_paths),
            "examples_failed": summary.get("examples_failed"),
            "examples_excluded_oversized": summary.get("examples_excluded_oversized"),
            "oversized_review_queue": len(oversized),
        },
        "excluded_from_default_training": {
            "mutopia": "review-only until human approval of PDF gallery and license attribution handling",
            "maestro": "CC BY-NC-SA; non-commercial performance overlay only",
            "pianocore": "CC BY-NC-SA; metadata-only research inventory",
            "asap": "CC BY-NC-SA; metadata-only research inventory",
        },
        "references": references,
        "next_gpu_commands": [
            "tar -xzf native-classical-pretraining-v1.tar.gz -C /root/midllm-local/classical-native-v1",
            "MIDI_LLM_NATIVE_DATASET=/root/midllm-local/classical-native-v1/training_dataset bash scripts/run_native_classical_stages.sh dry-run",
            "MIDI_LLM_NATIVE_DATASET=/root/midllm-local/classical-native-v1/training_dataset bash scripts/run_native_classical_stages.sh smoke",
            "MIDI_LLM_NATIVE_DATASET=/root/midllm-local/classical-native-v1/training_dataset bash scripts/run_native_classical_stages.sh conservative-pilot",
        ],
    }


def _markdown_report(report: Dict[str, Any]) -> str:
    dataset = report["default_training_dataset"]
    excluded = report["excluded_from_default_training"]
    lines = [
        "# Native Classical Pretraining Preflight",
        "",
        f"- Ready for GPU smoke: `{report['gate']['ready_for_gpu_smoke']}`",
        f"- Ready for GPU pilot: `{report['gate']['ready_for_gpu_pilot']}`",
        f"- Training started: `false`",
        "",
        "## Default Training Dataset",
        "",
        f"- Source: {dataset['source']}",
        f"- Examples: {dataset['examples_written']}",
        f"- Native segments: {dataset['native_segments_written']}",
        f"- Token files: {dataset['referenced_token_files']}",
        f"- Max segment event tokens: {dataset['segment_event_token_counts']['max']}",
        f"- Failed examples: {dataset['examples_failed']}",
        "",
        "## Excluded From Default Training",
        "",
    ]
    lines.extend(f"- {name}: {reason}" for name, reason in excluded.items())
    lines.extend(["", "## Next GPU Commands", ""])
    lines.extend(f"```bash\n{command}\n```" for command in report["next_gpu_commands"])
    return "\n".join(lines) + "\n"


def _token_paths_from_dataset(dataset_dir: Path) -> List[str]:
    paths = set()
    for split in ("train", "valid", "test"):
        for row in _read_jsonl(dataset_dir / f"{split}.jsonl"):
            for segment in row.get("native_segments", []):
                token_path = segment.get("native_model_tokens_path")
                if not token_path:
                    raise ValueError(f"Native segment in {split}.jsonl has no token path")
                if PurePosixPath(token_path).is_absolute() or ".." in PurePosixPath(token_path).parts:
                    raise ValueError(f"Unsafe token path in split: {token_path}")
                if not (dataset_dir / token_path).is_file():
                    raise ValueError(f"Referenced token file is missing: {token_path}")
                paths.add(token_path)
    return sorted(paths)


def _token_paths_from_archive(archive: tarfile.TarFile, members_by_name: Dict[str, tarfile.TarInfo]) -> set[str]:
    paths = set()
    for split in ("train", "valid", "test"):
        member_name = f"{DATASET_DIR}/{split}.jsonl"
        member = members_by_name.get(member_name)
        if member is None:
            continue
        payload = archive.extractfile(member)
        if payload is None:
            raise ValueError(f"Bundle split is unreadable: {member_name}")
        for line in payload.read().decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for segment in row.get("native_segments", []):
                paths.add(f"{DATASET_DIR}/{segment['native_model_tokens_path']}")
    return paths


def _load_optional_summary(value: Path | str | None) -> Dict[str, Any] | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_file():
        return {"missing": str(path)}
    base = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256_path(path)}
    if path.suffix.casefold() != ".json":
        return base
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "pipeline",
        "works",
        "works_requested",
        "works_profiled",
        "bounded_piece_review_candidates",
        "intermediate_proxy_review_candidates",
        "works_with_pdf",
        "metadata_rows",
        "unique_works",
        "performance_rows",
        "unique_title_composer_works",
        "commercial_use_allowed",
        "composition_backbone_eligible",
    )
    return {**base, **{key: data[key] for key in keys if key in data}}


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    archive.addfile(info, io.BytesIO(payload))


def _read_json_member(archive: tarfile.TarFile, name: str) -> Dict[str, Any]:
    member = archive.getmember(name)
    payload = archive.extractfile(member)
    if payload is None:
        raise ValueError(f"Bundle member is unreadable: {name}")
    return json.loads(payload.read())


def _validate_members(members: Iterable[tarfile.TarInfo]) -> None:
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Archive member escapes output directory: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"Archive links are not supported: {member.name}")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _json_bytes(data: Dict[str, Any]) -> bytes:
    return (json.dumps(data, indent=2) + "\n").encode("utf-8")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
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
    parser = argparse.ArgumentParser(description="Package or verify a native classical pretraining bundle")
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package")
    package.add_argument("--dataset-dir", required=True)
    package.add_argument("--output", required=True)
    package.add_argument("--mutopia-summary")
    package.add_argument("--pianocore-summary")
    package.add_argument("--asap-summary")
    package.add_argument("--maestro-summary")
    package.add_argument("--gallery-index")
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "package":
        result = package_native_pretraining_bundle(
            args.dataset_dir,
            args.output,
            mutopia_summary=args.mutopia_summary,
            pianocore_summary=args.pianocore_summary,
            asap_summary=args.asap_summary,
            maestro_summary=args.maestro_summary,
            gallery_index=args.gallery_index,
        )
    else:
        result = verify_native_pretraining_bundle(args.bundle)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
