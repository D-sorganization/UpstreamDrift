"""Write a run manifest next to a run's outputs (Issue #14432 / Epic #14436).

The emitter is engine-neutral: an engine run path writes its artifacts into one
run directory and then calls :func:`emit_run_manifest`, which hashes each
artifact, validates the manifest against the run-manifest v1 schema and writes
it beside them. Engine adapters share this emitter rather than copying it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from src.sim_run.manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    RunManifest,
    validate_run_manifest,
)

MANIFEST_FILENAME = "run_manifest.json"


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    """SHA-256 hex digest of ``value`` as compact, key-sorted JSON."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _output_digests(run_dir: Path, outputs: Iterable[Path]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for output in outputs:
        path = Path(output).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"run output {path.name!r} does not exist")
        if not path.is_relative_to(run_dir):
            raise ValueError(
                f"run output {path.name!r} must be inside the run directory"
            )
        digests[path.relative_to(run_dir).as_posix()] = sha256_file(path)
    return digests


def emit_run_manifest(
    run_dir: Path,
    *,
    outputs: Iterable[Path],
    model_id: str,
    revision: str,
    parameters: Mapping[str, Any],
    input_sha256: str,
    seed: int,
    engine_name: str,
    engine_version: str,
    ud_commit: str,
    created_at: datetime | None = None,
    input_label: str | None = None,
) -> RunManifest:
    """Hash ``outputs``, validate the manifest and write it into ``run_dir``.

    Raises:
        FileNotFoundError: if an output does not exist.
        ValueError: if an output lies outside ``run_dir``.
        RunManifestError: if any provenance field is invalid.

    Postcondition: ``run_dir / MANIFEST_FILENAME`` holds the returned manifest.
    """
    root = Path(run_dir).resolve()
    data = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "model": {"model_id": model_id, "revision": revision},
        "parameters": dict(parameters),
        "input_data": {"sha256": input_sha256, "label": input_label},
        "seed": seed,
        "engine": {"name": engine_name, "version": engine_version},
        "ud_commit": ud_commit,
        "created_at": created_at or datetime.now(timezone.utc),
        "outputs": _output_digests(root, outputs),
    }
    manifest = validate_run_manifest(data)
    text = json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n"
    (root / MANIFEST_FILENAME).write_text(text, encoding="utf-8")
    return manifest
