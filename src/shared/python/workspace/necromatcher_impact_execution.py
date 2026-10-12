"""Impact execution provenance extends refit stamps without importing native SDKs."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, distribution
import json
from pathlib import Path
from typing import Any

from src.shared.python.version_info import get_repo_root
from .artifact_handoff import compute_file_sha256
from .necromatcher_fit_jobs import fit_execution_stamp


def _digest(record: object) -> str:
    encoded = json.dumps(record, sort_keys=True, allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _flight_kernel_record() -> dict[str, Any]:
    """Pin installed Python/native flight files; metadata lookup executes no SDK."""
    try:
        installed = distribution("upstream-physics")
    except PackageNotFoundError:
        return {"version": "not-installed", "files": {}}
    if installed.files is None:
        raise ValueError("Installed flight kernel has no file inventory")
    hashes = {
        str(relative): compute_file_sha256(Path(str(installed.locate_file(relative))))
        for relative in sorted(installed.files, key=str)
        if relative.suffix.lower() in {".py", ".pyd", ".so", ".dll"}
    }
    return {"version": installed.version, "files": hashes}


def impact_execution_stamp() -> dict[str, Any]:
    """Retain refit provenance and pin actual impact/flight implementation bytes.

    Python physics sources and the two canonical viewer validators extend the
    source map. The installed flight distribution's version and executable-file
    hashes extend runtime provenance, including same-version binary replacement.
    Absence is recorded explicitly, never repaired through a simulation fallback.
    This is software identity evidence, not historical scientific qualification.
    """
    stamp = fit_execution_stamp()
    root = get_repo_root()
    sources = dict(stamp["source_files"])
    extra = [
        *sorted((root / "src/shared/python/physics").rglob("*.py")),
        root / "src/api/routes/_ball_flight_trajectory_import.py",
        root / "src/launchers/_shot_tracer_trajectory_import.py",
    ]
    for path in extra:
        sources[path.relative_to(root).as_posix()] = compute_file_sha256(path)
    runtime = {**stamp["runtime"], "upstream_physics": _flight_kernel_record()}
    return {
        **stamp,
        "source_files": sources,
        "source_sha256": _digest(sources),
        "runtime": runtime,
        "runtime_sha256": _digest(runtime),
    }
