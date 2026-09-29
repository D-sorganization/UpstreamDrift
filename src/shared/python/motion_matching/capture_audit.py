"""Repeatable pre-fit capture audit using the existing C3D adapter (#9921)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.motion_capture.provenance import sha256_of
from src.shared.python.motion_pipeline.sources import C3DAdapter

from .provenance import git_commit_short


def audit_capture(path: Path) -> dict[str, Any]:
    """Return JSON-safe per-marker coverage and extents in source axes/metres.

    No interpolation, event detection, axis inference, marker exclusions or
    joint-centre assumptions are applied. The source file must remain unchanged
    throughout import. An audit is never evidence of a successful model fit.
    """
    path = Path(path)
    if path.suffix.lower() != ".c3d" or not path.is_file():
        raise ValueError("path must name an existing C3D capture")
    digest = sha256_of(path)
    trajectory = C3DAdapter().load(path)
    if sha256_of(path) != digest:
        raise ValueError("capture changed while being audited")
    frames = trajectory.frames
    time = np.array([frame.timestamp for frame in frames])
    if len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("capture needs at least two strictly increasing finite times")
    metadata = trajectory.metadata
    if not metadata.get("units_declared", False):
        raise ValueError("capture must declare spatial units before fitting")
    names = metadata.get("source_labels", [])
    if not names or len(set(names)) != len(names):
        raise ValueError("capture needs unique source marker labels")
    markers: dict[str, Any] = {}
    for name in names:
        positions, indices = [], []
        for index, frame in enumerate(frames):
            marker = frame.markers.get(name)
            if marker is not None and not marker.occluded:
                point = (marker.x, marker.y, marker.z)
                if np.isfinite(point).all():
                    positions.append(point)
                    indices.append(index)
        xyz = np.asarray(positions, dtype=float)
        markers[name] = {
            "valid_samples": len(positions),
            "missing_samples": len(frames) - len(positions),
            "first_valid_frame": indices[0] if indices else None,
            "last_valid_frame": indices[-1] if indices else None,
            "first_position_m": xyz[0].tolist() if positions else None,
            "range_m": np.ptp(xyz, axis=0).tolist() if positions else None,
        }
    return {
        "schema": "simscape-capture-audit/1.0.0",
        "qualification": "capture-audit-only",
        "source_file": path.name,
        "source_sha256": digest,
        "revision": git_commit_short(),
        "source_units": metadata.get("units"),
        "coordinate_units": "m",
        "coordinate_axes": "source-unregistered",
        "frame_count": len(frames),
        "duration_s": float(time[-1] - time[0]),
        "sample_interval_s": float(np.median(np.diff(time))),
        "markers": markers,
    }


def main() -> None:
    """Write one new audit JSON; an existing evidence file is never overwritten."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = audit_capture(args.capture)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
