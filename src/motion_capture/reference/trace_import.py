"""Bounded Trace v2 import through the shared simulation serializer."""

from __future__ import annotations

import json
from math import prod
from pathlib import Path

import numpy as np

from .importers import MotionDraft
from .model import MAX_REFERENCE_BYTES, MAX_SAMPLES, ReferenceSource


def _preflight(path: Path) -> None:
    import h5py

    with h5py.File(path, "r") as handle:
        if not str(handle.attrs.get("schema_version", "")).startswith("2."):
            raise ValueError("Reference analysis requires a Trace v2 file")
        total = 0
        for name in handle:
            if not isinstance(handle.get(name, getlink=True), h5py.HardLink):
                raise ValueError(
                    "Trace reference cannot contain external or soft links"
                )
            dataset = handle[name]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError("Trace reference requires root datasets")
            total += prod(dataset.shape) * dataset.dtype.itemsize
            if total > MAX_REFERENCE_BYTES:
                raise ValueError("Decoded trace exceeds 64 MB; export a shorter motion")


def _names(encoded: object, count: int) -> tuple[str, ...]:
    if encoded is None:
        return tuple(f"marker_{index}" for index in range(count))
    try:
        names = json.loads(encoded) if isinstance(encoded, str) else None
    except json.JSONDecodeError as exc:
        raise ValueError("Trace marker names must be a JSON string array") from exc
    if (
        not isinstance(names, list)
        or len(names) != count
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            "Trace marker names must be unique strings matching marker columns"
        )
    return tuple(names)


def load_trace_draft(path: Path, digest: str) -> MotionDraft:
    """Preserve trace coordinates and time, requiring explicit mapping of unknown axes."""
    from src.shared.python.simulation_backends.protocol import Trace
    from src.shared.python.simulation_backends.trace_io import read_trace

    _preflight(path)
    trace = read_trace(path)
    if not isinstance(trace, Trace):
        raise ValueError(
            "Choose and export one rollout from the batch before importing"
        )
    points = trace.markers
    if points is None or points.shape[1] == 0:
        raise ValueError(
            "This trace has no marker trajectories; export backend marker kinematics first"
        )
    frames, count, _ = points.shape
    if not 0 < frames <= 100_000 or count > 256 or frames * count > MAX_SAMPLES:
        raise ValueError("Trace marker/frame count exceeds reference limits")
    if (
        not np.isfinite(trace.t).all()
        or np.any(np.diff(trace.t) <= 0)
        or np.isinf(points).any()
    ):
        raise ValueError(
            "Trace requires increasing finite times and finite-or-missing marker coordinates"
        )
    names = _names(trace.meta.get("marker_names_json"), count)
    identity = str(trace.meta.get("model_identity", "")).strip()
    identity = f"{trace.backend} / {identity}" if identity else trace.backend
    source = ReferenceSource(path=str(path), sha256=digest, format="simulation-trace/2")
    return MotionDraft(
        source,
        names,
        tuple(trace.t),
        points.copy(),
        "m",
        True,
        trace.meta.get("frame") == "world_Zup",
        identity,
    )
