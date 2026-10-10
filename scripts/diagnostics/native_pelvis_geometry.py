"""Bounded in-sample pelvis geometry probe; output belongs in a private folder.

Uses the existing frozen capture loader and shared calibration/IK. First-frame
marker offsets impose a reference gauge, not measured anatomical calibration.
No dynamics, contact, grip or independent holdout acceptance is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import numpy as np

from src.engines.physics_engines.opensim.python.tour_matching.native_marker_geometry import (
    NativeMarkerGeometry,
)
from src.shared.python.motion_matching.full_body_ik import (
    compute_marker_rms_trajectory,
    solve_full_body_ik_trajectory,
)
from src.shared.python.motion_matching.marker_calibration import (
    Offsets,
    Pose,
    static_marker_offsets,
)
from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_SEGMENTS,
    TOUR_CAPTURES,
    TourCapture,
    capture_kind,
)

PELVIS_COORDINATES = tuple(
    f"/jointset/ground_pelvis/{name}"
    for name in (
        "pelvis_tilt",
        "pelvis_list",
        "pelvis_rotation",
        "pelvis_tx",
        "pelvis_ty",
        "pelvis_tz",
    )
)


def sample_observations(capture: TourCapture) -> tuple[TourCapture, np.ndarray]:
    """Select three original times, without turning them into independent trials."""
    if capture.frames < 3:
        raise ValueError("At least three frames are required for the bounded probe")
    pelvis = capture.subset(MARKER_SEGMENTS["pelvis"])
    indices = np.array([0, (pelvis.frames - 1) // 2, pelvis.frames - 1])
    if not pelvis.valid[indices].all():
        raise ValueError("The predeclared pelvis samples must all be observed")
    return TourCapture(
        pelvis.time_s[indices],
        pelvis.labels,
        pelvis.points_m[indices],
        pelvis.valid[indices],
        pelvis.source_sha256,
    ), indices


def run_probe(
    model_path: Path, source_sha256: str, capture: TourCapture
) -> dict[str, object]:
    """Execute bounded position-level fitting and retain native evidence limits."""
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != source_sha256:
        raise ValueError("Source model hash differs from the pinned request")
    sampled, indices = sample_observations(capture)
    native = NativeMarkerGeometry(model_path, PELVIS_COORDINATES)
    body = "/bodyset/pelvis"
    provisional = dict.fromkeys(sampled.labels, (body, (0.0, 0.0, 0.0)))
    initial = native.initial_coordinates
    pose0 = native.frame_poses(provisional, initial)
    calibration = TourCapture(
        sampled.time_s[:1],
        sampled.labels,
        sampled.points_m[:1],
        sampled.valid[:1],
        sampled.source_sha256,
    )
    offsets = static_marker_offsets(
        calibration, dict.fromkeys(sampled.labels, body), [pose0]
    )

    def pose_fn(q: np.ndarray) -> dict[str, Pose]:
        return native.frame_poses(offsets, q)

    started = time.perf_counter()
    receipt = _native_context(native, sampled, capture, indices)
    receipt["initial_q"] = initial.tolist()
    receipt["declared_marker_offsets"] = offsets
    try:
        q = solve_full_body_ik_trajectory(
            pose_fn,
            offsets,
            sampled,
            initial,
            reg_weight=1e-3,
            max_nfev=30,
            coordinate_bounds=native.coordinate_bounds,
        )
    except (ValueError, RuntimeError) as error:
        receipt.update(
            status="failed-not-qualified",
            error_type=type(error).__name__,
            error=str(error),
            elapsed_s=time.perf_counter() - started,
        )
        return receipt
    per_frame, per_marker, rms = compute_marker_rms_trajectory(
        pose_fn, offsets, sampled, q
    )
    achieved, station_discrepancy = _verify_native_stations(native, offsets, q)
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != source_sha256:
        raise ValueError("Source model changed during evaluation")
    receipt.update(
        {
            "requested_q": q.tolist(),
            "achieved_q": achieved,
            "assembly_accuracy": native.assembly_accuracy,
            "coordinate_tolerance": native.coordinate_tolerance,
            "calibration_frames": [0],
            "offset_radius_m": {
                label: float(np.linalg.norm(offset))
                for label, (_, offset) in offsets.items()
            },
            "marker_rms_m": rms,
            "per_frame_rms_m": per_frame.tolist(),
            "per_marker_rms_m": per_marker,
            "native_station_discrepancy_m": station_discrepancy,
            "elapsed_s": time.perf_counter() - started,
            "minimum_coordinate_bound_margin": float(
                np.minimum(
                    q - native.coordinate_bounds[0], native.coordinate_bounds[1] - q
                ).min()
            ),
        }
    )
    return receipt


def _verify_native_stations(
    native: NativeMarkerGeometry,
    offsets: Offsets,
    q: np.ndarray,
) -> tuple[list[list[float]], list[float]]:
    """Compare native station positions against native rigid frame transforms."""
    achieved = []
    station_discrepancy = []
    for row in q:
        poses = native.frame_poses(offsets, row)
        predicted = np.array(
            [
                poses[body][0] @ np.asarray(offset) + poses[body][1]
                for body, offset in offsets.values()
            ]
        )
        stations = native.marker_positions(row, offsets)
        station_discrepancy.append(float(np.max(np.abs(predicted - stations))))
        achieved.append(native.achieved_coordinates.tolist())
    return achieved, station_discrepancy


def _native_context(
    native: NativeMarkerGeometry,
    sampled: TourCapture,
    capture: TourCapture,
    indices: np.ndarray,
) -> dict[str, object]:
    """Retain native/source identity even if the fitting evaluation fails."""
    receipt: dict[str, object] = {
        "status": "exploratory-in-sample-pelvis-geometry-not-qualified",
        "source_sha256": native.source_sha256,
        "loaded_sha256": native.loaded_sha256,
        "native_geometry_identity": native.identity_sha256,
        "runtime_version": native.runtime_version,
        "runtime_extension_sha256": native.runtime_extension_sha256,
        "provider_sha256": native.provider_sha256,
        "shared_source_sha256": _shared_source_hashes(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "native_inventory": native.inventory,
        "capture_sha256": sampled.source_sha256,
        "original_frame_indices": indices.tolist(),
        "observation_times_s": sampled.time_s.tolist(),
        "full_capture_frames": capture.frames,
        "fit_samples": sampled.frames,
        "coordinate_paths": list(native.coordinate_order),
        "coordinate_bounds": [values.tolist() for values in native.coordinate_bounds],
    }
    receipt.update(_qualification_limits())
    return receipt


def _qualification_limits() -> dict[str, object]:
    return {
        "solver": {
            "provider": "solve_full_body_ik_trajectory",
            "method": "trf-with-unchanged-native-source-bounds",
            "cost_only_stopping": "disabled-for-active-bound-seeds",
            "max_nfev": 30,
            "reg_weight": 1e-3,
            "termination_qualification": "not-reported-by-provider",
        },
        "binding_basis": "Existing pelvic segment grouping; explicit source pelvis body. First-frame offsets impose an unqualified reference gauge, not donor-landmark equivalence.",
        "unqualified": [
            "anatomical-offsets",
            "world-ground-calibration",
            "capture-to-model-fixed-frame-registration",
            "anthropometry",
            "independent-trials",
            "noise-events",
            "nonpelvic-mappings",
            "wrists-hand-muscles",
            "physical-club-grip",
            "contact",
            "passive-muscle-loads",
            "reserves",
            "dynamics",
            "full-state-replay",
            "full-horizon-continuous-motion",
        ],
    }


def _shared_source_hashes() -> dict[str, str]:
    """Bind reused calculation implementations without exposing private paths."""
    names = (
        "src.shared.python.motion_matching.full_body_ik",
        "src.shared.python.motion_matching.marker_calibration",
        "src.shared.python.motion_matching.tour_capture_contract",
        "src.engines.physics_engines.opensim.python.tour_matching.trc",
    )
    hashes = {}
    for name in names:
        filename = importlib.import_module(name).__file__
        if filename is None:
            raise ValueError(f"Source file is unavailable for {name}")
        hashes[name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    return hashes


def load_isolated_capture(
    capture_path: Path, capture_python: Path
) -> tuple[TourCapture, dict[str, object]]:
    """Reuse TRC across DLL-incompatible SDK processes, retaining source clock."""
    from src.engines.physics_engines.opensim.python.tour_matching.trc import read_trc

    source_hash = hashlib.sha256(capture_path.read_bytes()).hexdigest()
    spec = TOUR_CAPTURES[capture_kind(source_hash)]
    environment = os.environ.copy()
    root = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = str(root)
    with tempfile.TemporaryDirectory(prefix="native-pelvis-capture-") as folder:
        trc = Path(folder) / "observations.trc"
        result = subprocess.run(
            [
                str(capture_python),
                "-m",
                "scripts.diagnostics.frozen_capture_trc",
                str(capture_path),
                str(trc),
            ],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        metadata = json.loads(result.stdout)
        if (
            metadata["capture_sha256"] != source_hash
            or metadata["trc_sha256"] != hashlib.sha256(trc.read_bytes()).hexdigest()
        ):
            raise ValueError("Isolated capture/TRC identity mismatch")
        decoded = read_trc(trc)
    original_time = np.arange(spec.frames) / spec.rate_hz
    if (
        decoded.labels != spec.labels
        or decoded.frames != spec.frames
        or np.max(np.abs(decoded.time_s - original_time)) > 5.1e-10
    ):
        raise ValueError("TRC differs from frozen capture specification")
    capture = TourCapture(
        original_time, decoded.labels, decoded.points_m, decoded.valid, source_hash
    )
    return capture, metadata


def main() -> None:
    """Write a private diagnostic receipt or a retained failure receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        capture, transfer = load_isolated_capture(args.capture, args.capture_python)
        receipt = run_probe(args.model, args.model_sha256, capture)
        receipt["capture_transfer"] = transfer
    except (ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        receipt = {
            "status": "failed-not-qualified",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        raise
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if receipt["status"] == "failed-not-qualified":
        raise RuntimeError("Native geometry probe failed; see private receipt")


if __name__ == "__main__":
    main()
