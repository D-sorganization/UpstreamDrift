"""Materialize a byte-exact structural checkpoint prefix without outcomes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
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

_SCHEMA = "articulated-structural-factorial-prefix-view/1.0.0"
_SESSION_SCHEMA = "articulated-structural-factorial-session/1.0.0"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"cannot hash {path}") from exc


def _read_mapping(path: Path, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _session(source_dir: Path, launch: Mapping[str, object]) -> tuple[Path, str]:
    path = source_dir / "execution-session.json"
    session = _read_mapping(path, name="execution session")
    runtime_identity = session.get("runtime_identity_sha256")
    if (
        session.get("schema_version") != _SESSION_SCHEMA
        or session.get("execution_revision") != launch.get("execution_revision")
        or not isinstance(runtime_identity, str)
        or _SHA256.fullmatch(runtime_identity) is None
    ):
        raise ValueError("execution session identity does not match the launch")
    return path, _sha256(path)


def materialize_structural_prefix_view(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    source_dir: Path,
    prefix_stop_exclusive: int,
    output_dir: Path,
) -> dict[str, object]:
    """Create an atomic, source-preserving prefix for interim exact replay audit."""

    if not isinstance(source_dir, Path) or not source_dir.is_dir():
        raise ValueError("source_dir must be an existing pathlib.Path directory")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    if output_dir.exists():
        raise FileExistsError("output directory must not already exist")
    registered = build_registered_cases(plan)
    if (
        isinstance(prefix_stop_exclusive, bool)
        or not isinstance(prefix_stop_exclusive, int)
        or prefix_stop_exclusive <= 0
        or prefix_stop_exclusive > len(registered)
    ):
        raise ValueError("prefix stop must satisfy 0 < stop <= registered case count")

    checkpoints = load_available_checkpoints(
        plan=plan, launch=launch, checkpoint_dir=source_dir
    )
    case_indices = {case.case_key: index for index, case in enumerate(registered)}
    observed = [case_indices[checkpoint.case.case_key] for checkpoint in checkpoints]
    if observed != list(range(len(observed))):
        raise ValueError("source checkpoints must form a contiguous prefix from zero")
    if len(checkpoints) < prefix_stop_exclusive:
        raise ValueError("prefix stop exceeds the available source checkpoint prefix")
    session_path, session_sha256 = _session(source_dir, launch)

    selected = checkpoints[:prefix_stop_exclusive]
    source_files = [session_path]
    for checkpoint in selected:
        source_files.append(checkpoint.path)
        sidecar = checkpoint.path.with_suffix(".npz")
        if sidecar.is_file():
            source_files.append(sidecar)
    file_records = [
        {"name": path.name, "sha256": _sha256(path)}
        for path in sorted(source_files, key=lambda item: item.name)
    ]
    source_manifest = source_dir / "collection-manifest.json"
    manifest: dict[str, object] = {
        "schema_version": _SCHEMA,
        "classification": "operational_prefix_view_not_scientific_summary",
        "plan_sha256": plan_sha256(plan),
        "execution_revision": launch.get("execution_revision"),
        "execution_session_sha256": session_sha256,
        "source_directory_name": source_dir.resolve().name,
        "source_collection_manifest_sha256": (
            _sha256(source_manifest) if source_manifest.is_file() else None
        ),
        "source_checkpoint_count": len(checkpoints),
        "prefix_case_stop_exclusive": prefix_stop_exclusive,
        "complete_source_exposed": prefix_stop_exclusive == len(checkpoints),
        "files": file_records,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f"{output_dir.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        for source_path in source_files:
            shutil.copyfile(source_path, temporary / source_path.name)
        _write_json(temporary / "prefix-view-manifest.json", manifest)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Materialize one explicit prefix and print its operational manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--prefix-stop-exclusive", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = materialize_structural_prefix_view(
        plan=_read_mapping(args.plan, name="plan"),
        launch=_read_mapping(args.launch, name="launch"),
        source_dir=args.source_dir,
        prefix_stop_exclusive=args.prefix_stop_exclusive,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "materialize_structural_prefix_view"]
