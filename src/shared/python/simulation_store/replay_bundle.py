"""Hash-verified import of retained R2025b results for offline playback.

This adapter never loads an executable model, starts MATLAB or runs physics.
Manifest paths are relative to the manifest directory; numerical arrays are owned
and read-only. This version supports the archived native Simscape MAT/NPZ wire.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

SCHEMA = "upstreamdrift/saved-simscape-replay/1"
_ARTIFACTS = {"model", "candidate", "trajectory", "target", "report"}


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
    if not isinstance(document, dict):
        raise ValueError("Expected JSON object")
    return document


def _artifact(root: Path, entry: Any) -> Path:
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
        raise ValueError("Artifact requires path and sha256")
    name = entry["path"]
    posix = PurePosixPath(name)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError("Artifact path must be a relative subdirectory entry")
    resolved = (root / Path(posix)).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError("Artifact path escapes manifest directory")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"Artifact hash mismatch: {name}")
    return resolved


def _array(source: Any, shape: tuple[int, ...], name: str) -> NDArray[Any]:
    if source is None:
        raise ValueError(f"Missing array: {name}")
    result = np.array(source, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} requires finite shape {shape}")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class SavedSimscapeReplay:
    """Verified replay metadata and immutable owned numerical samples."""

    run_id: str
    status: str
    model: Mapping[str, Any]
    candidate: Mapping[str, Any]
    report: Mapping[str, Any]
    coordinate_names: tuple[str, ...]
    arrays: Mapping[str, NDArray[Any]]
    manifest_path: Path
    artifact_paths: Mapping[str, Path] = field(default_factory=dict)


def _load_trajectory_arrays(
    traj_path: Path, names: list[str], duration: float, n_expected: Any, k_markers: int
) -> tuple[dict[str, NDArray[Any]], NDArray[Any], int]:
    raw = loadmat(traj_path)
    time = np.asarray(raw.get("time_s"), dtype=float).reshape(-1)
    if time.size < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Replay clock requires finite increasing samples")
    if not np.isclose(time[0], 0, atol=1e-12, rtol=0) or not np.isclose(
        time[-1], duration, atol=1e-9, rtol=0
    ):
        raise ValueError("Manifest horizon differs from original replay clock")
    n, d = len(time), len(names)
    if k_markers == 0 or n_expected != n:
        raise ValueError("Missing markers or inconsistent sample count")
    arrays: dict[str, NDArray[Any]] = {"time_s": _array(time, (n,), "time_s")}
    for key in ("q", "qd", "qdd", "omega"):
        arrays[key] = _array(raw.get(key), (n, d), key)
    tau = np.array(raw.get("tau"), dtype=float, copy=True)
    if tau.shape != (n, d):
        raise ValueError("tau requires the recorded coordinate shape")
    tau_valid = np.isfinite(tau)
    tau.flags.writeable = False
    tau_valid.flags.writeable = False
    arrays.update(tau=tau, tau_valid=tau_valid)
    arrays["markers_m"] = _array(raw.get("prediction"), (n, k_markers, 3), "prediction")
    return arrays, time, n


def _load_target_arrays(
    target_path: Path, time: NDArray[Any], n: int, k_markers: int
) -> dict[str, NDArray[Any]]:
    with np.load(target_path, allow_pickle=False) as target:
        if not np.array_equal(target["time_s"], time):
            raise ValueError("Target clock differs from retained replay")
        valid = np.array(target["valid"], copy=True)
        points = np.array(target["target_m"], dtype=float, copy=True)
    if (
        valid.dtype != np.bool_
        or valid.shape != (n, k_markers)
        or points.shape != (n, k_markers, 3)
    ):
        raise ValueError("Invalid target/mask shape or type")
    if not valid.any() or not np.isfinite(points[valid]).all():
        raise ValueError("Observed target markers must be finite")
    valid.flags.writeable = False
    points.flags.writeable = False
    return {"target_m": points, "valid": valid}


def load_simscape_bundle(path: Path | str) -> SavedSimscapeReplay:
    """Verify all artifact identities and clocks before exposing archived states."""
    manifest_path = Path(path).resolve()
    doc = _json(manifest_path)
    if doc.get("schema_version") != SCHEMA or doc.get("engine") != "simscape":
        raise ValueError("Unsupported saved replay schema or engine")
    run_id = doc.get("run_id")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not all(c.isalnum() or c in "_-" for c in run_id)
    ):
        raise ValueError("Invalid run_id")
    status = doc.get("status")
    if status not in {"accepted", "rejected", "unqualified"}:
        raise ValueError("Invalid qualification status")
    entries = doc.get("artifacts")
    if not isinstance(entries, dict) or set(entries) != _ARTIFACTS:
        raise ValueError("Missing or unknown replay artifact roles")
    files = {
        key: _artifact(manifest_path.parent, entry) for key, entry in entries.items()
    }
    model, candidate, report = (
        _json(files[key]) for key in ("model", "candidate", "report")
    )
    if report.get("matlab_release") != "2025b":
        raise ValueError("Native replay requires R2025b evidence")
    gates = report.get("gates")
    if (
        not isinstance(gates, dict)
        or not gates
        or any(type(v) is not bool for v in gates.values())
    ):
        raise ValueError("Report requires explicit boolean gates")
    # Acceptance here is recorded prefix status, not full-swing qualification.
    if status == "accepted" and not all(gates.values()):
        raise ValueError("Accepted status contradicts recorded gates")
    names = candidate.get("coordinate_names")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(n, str) for n in names)
    ):
        raise ValueError("Candidate requires explicit coordinate_names")
    if len(set(names)) != len(names) or set(names) != set(
        model.get("coordinate_order", [])
    ):
        raise ValueError("Model/candidate coordinate inventory mismatch")
    duration = doc.get("duration_s")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (float, int))
        or not np.isfinite(duration)
    ):
        raise ValueError("Invalid replay duration")
    if not np.isclose(
        report.get("duration_s", -1), duration, atol=1e-9, rtol=0
    ) or not np.isclose(candidate.get("duration_s", -1), duration, atol=1e-9, rtol=0):
        raise ValueError("Report/candidate horizon mismatch")
    k = len(candidate.get("marker_labels", []))
    arrays, time, n = _load_trajectory_arrays(
        files["trajectory"], names, float(duration), report.get("n_samples"), k
    )
    arrays.update(_load_target_arrays(files["target"], time, n, k))
    return SavedSimscapeReplay(
        run_id,
        status,
        MappingProxyType(model),
        MappingProxyType(candidate),
        MappingProxyType(report),
        tuple(names),
        MappingProxyType(arrays),
        manifest_path,
        MappingProxyType(files),
    )
