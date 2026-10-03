"""Owned terminal child measurements, authenticated to the canonical request."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from src.shared.python.estimation import SolverTelemetry
from src.shared.python.motion_matching.jobs.io_atomic import (
    atomic_write_json,
    read_text,
)


def _request_binding(path: Path, request: dict[str, Any]) -> dict[str, str]:
    """Reject redirected paths before writing any child measurement."""
    if not isinstance(path, Path) or path.name != "request.json":
        raise ValueError("Telemetry requires the owned run request path")
    root = path.parent
    library = Path(request["library_root"])
    if not re.fullmatch("[a-f0-9]{32}", root.name) or root.parent != library / "runs":
        raise ValueError("Telemetry requires an owned run within the library")
    if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
        raise ValueError("Telemetry owned run must be link-free and regular")
    raw = path.read_bytes()
    if json.loads(raw) != request:
        raise ValueError("Telemetry request binding changed")
    return {
        "run_id": root.name,
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "source_fit_id": request["source_fit_id"],
        "source_fit_hash": request["source_fit_hash"],
        "source_sha256": request["execution_stamp"]["source_sha256"],
        "runtime_sha256": request["execution_stamp"]["runtime_sha256"],
    }


def _write(path: Path, request: dict[str, Any], value: SolverTelemetry) -> None:
    binding = _request_binding(path, request)
    target = path.parent / "worker-telemetry.json"
    if target.exists():
        raise ValueError("Owned terminal telemetry already exists")
    atomic_write_json(
        target,
        {
            "schema": "necromatcher/worker-telemetry/1",
            "binding": binding,
            "telemetry": value.to_record(),
        },
    )


def write_worker_telemetry(
    request_path: Path,
    request: dict[str, Any],
    fit: dict[str, Any] | None,
    elapsed_s: float,
    terminal_reason: str,
) -> SolverTelemetry:
    """Persist child-entry scope; missing solver results stay explicitly unknown."""
    original = fit["evidence"]["original_fit"] if fit is not None else None
    if original is not None and original.get("solver_telemetry") is not None:
        value = SolverTelemetry.from_record(original["solver_telemetry"])
    else:
        value = SolverTelemetry(
            termination_reason=terminal_reason,
            unavailable_reason="optimizer_not_run"
            if original is not None and original.get("optimizer_ran") is False
            else "solver_result_unavailable",
        )
    value = replace(value, worker_elapsed_s=elapsed_s)
    _write(request_path, request, value)
    if original is not None:
        original["solver_telemetry"] = value.to_record()
    return value


def read_worker_telemetry(
    run_root: Path, request: dict[str, Any]
) -> SolverTelemetry | None:
    """Read only measurements bound to this exact run, request and source/runtime."""
    path = run_root / "worker-telemetry.json"
    if path.is_symlink():
        raise ValueError("Telemetry must be a regular owned file")
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("Telemetry must be a regular owned file")
    record = json.loads(read_text(path))
    if (
        set(record) != {"schema", "binding", "telemetry"}
        or record["schema"] != "necromatcher/worker-telemetry/1"
    ):
        raise ValueError("Malformed terminal telemetry record")
    if record["binding"] != _request_binding(run_root / "request.json", request):
        raise ValueError("Telemetry source/run/request binding differs")
    return SolverTelemetry.from_record(record["telemetry"])


def record_missing_worker_telemetry(
    run_root: Path, request: dict[str, Any], reason: str
) -> None:
    """Parent failure cannot impersonate missing child measurements or timings."""
    if read_worker_telemetry(run_root, request) is None:
        _write(
            run_root / "request.json",
            request,
            SolverTelemetry(
                termination_reason=reason,
                unavailable_reason="child_terminal_receipt_unavailable",
            ),
        )
