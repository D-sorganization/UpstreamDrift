"""Collect identity-matched hosted structural slices without mutating sources."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    build_registered_cases,
    load_available_checkpoints,
    plan_sha256,
)

_SCHEMA = "articulated-structural-factorial-collection/1.0.0"
_SESSION_SCHEMA = "articulated-structural-factorial-session/1.0.0"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class StructuralSliceSource:
    """Immutable provenance supplied with one downloaded workflow artifact."""

    run_id: int
    artifact_name: str
    directory: Path

    def __post_init__(self) -> None:
        if isinstance(self.run_id, bool) or not isinstance(self.run_id, int):
            raise TypeError("run_id must be an integer")
        if self.run_id <= 0:
            raise ValueError("run_id must be positive")
        if not isinstance(self.artifact_name, str):
            raise TypeError("artifact_name must be a string")
        if not self.artifact_name.strip():
            raise ValueError("artifact_name must be nonempty")
        if not isinstance(self.directory, Path):
            raise TypeError("directory must be a pathlib.Path")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _session_bytes(
    source: StructuralSliceSource, launch: Mapping[str, object]
) -> bytes:
    path = source.directory / "execution-session.json"
    try:
        content = path.read_bytes()
        record = json.loads(content)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("execution session is unreadable") from exc
    if not isinstance(record, dict):
        raise ValueError("execution session must be a JSON object")
    runtime_identity = record.get("runtime_identity_sha256")
    if (
        record.get("schema_version") != _SESSION_SCHEMA
        or record.get("execution_revision") != launch.get("execution_revision")
        or not isinstance(runtime_identity, str)
        or _SHA256.fullmatch(runtime_identity) is None
    ):
        raise ValueError("execution session identity is invalid for this launch")
    return content


def _allowed_names(source: StructuralSliceSource) -> set[str]:
    return {
        "execution-session.json",
        *(path.name for path in source.directory.glob("case-*.json")),
        *(path.name for path in source.directory.glob("case-*.npz")),
    }


def collect_structural_slices(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    sources: Sequence[StructuralSliceSource],
    output_dir: Path,
) -> dict[str, object]:
    """Atomically collect non-overlapping, byte-identical execution slices."""

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    if not sources:
        raise ValueError("at least one slice source is required")
    if output_dir.exists():
        raise FileExistsError("output directory must not already exist")
    ordered = tuple(sorted(sources, key=lambda source: source.run_id))
    if len({source.run_id for source in ordered}) != len(ordered):
        raise ValueError("slice run IDs must be unique")

    session = _session_bytes(ordered[0], launch)
    source_records: list[dict[str, object]] = []
    checkpoint_sources: dict[str, StructuralSliceSource] = {}
    checkpoint_files: dict[str, tuple[Path, ...]] = {}
    for source in ordered:
        if not source.directory.is_dir():
            raise ValueError("slice source directory does not exist")
        if _session_bytes(source, launch) != session:
            raise ValueError("all slices must have identical execution-session bytes")
        visible = {path.name for path in source.directory.iterdir() if path.is_file()}
        unexpected = visible - _allowed_names(source)
        if unexpected:
            raise ValueError("slice contains unexpected files")
        checkpoints = load_available_checkpoints(
            plan=plan, launch=launch, checkpoint_dir=source.directory
        )
        if not checkpoints:
            raise ValueError("slice contains no registered checkpoints")
        files: list[dict[str, str]] = []
        for checkpoint in checkpoints:
            name = checkpoint.path.name
            if name in checkpoint_sources:
                raise ValueError(f"overlapping checkpoint across slices: {name}")
            paths = [checkpoint.path]
            sidecar = checkpoint.path.with_suffix(".npz")
            if sidecar.is_file():
                paths.append(sidecar)
            checkpoint_sources[name] = source
            checkpoint_files[name] = tuple(paths)
            files.extend({"name": path.name, "sha256": _sha256(path)} for path in paths)
        source_records.append(
            {
                "run_id": source.run_id,
                "artifact_name": source.artifact_name,
                "checkpoint_count": len(checkpoints),
                "files": sorted(files, key=lambda item: item["name"]),
            }
        )

    registered_count = len(build_registered_cases(plan))
    manifest: dict[str, object] = {
        "schema_version": _SCHEMA,
        "classification": "execution_collection_not_scientific_summary",
        "plan_sha256": plan_sha256(plan),
        "execution_revision": launch.get("execution_revision"),
        "execution_session_sha256": hashlib.sha256(session).hexdigest(),
        "sources": source_records,
        "combined_checkpoint_count": len(checkpoint_files),
        "registered_case_count": registered_count,
        "missing_case_count": registered_count - len(checkpoint_files),
        "complete": len(checkpoint_files) == registered_count,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f"{output_dir.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        (temporary / "execution-session.json").write_bytes(session)
        for name in sorted(checkpoint_files):
            for source_path in checkpoint_files[name]:
                shutil.copyfile(source_path, temporary / source_path.name)
        _write_json(temporary / "collection-manifest.json", manifest)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _read_mapping(path: Path, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Collect explicit downloaded artifacts and print the retained manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        required=True,
        metavar=("RUN_ID", "ARTIFACT_NAME", "DIRECTORY"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    sources = tuple(
        StructuralSliceSource(int(run_id), artifact_name, Path(directory))
        for run_id, artifact_name, directory in args.source
    )
    manifest = collect_structural_slices(
        plan=_read_mapping(args.plan, name="plan"),
        launch=_read_mapping(args.launch, name="launch"),
        sources=sources,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["StructuralSliceSource", "collect_structural_slices", "main"]
