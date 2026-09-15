"""Measure an existing model without asserting Shadow Tracker readiness (#10124).

Run as ``python3 -m scripts.shadow_tracker.model_probe --output receipt.json``.
The passive short-window experiment uses the existing measured tour capture for
the existing rollout API; it does not fabricate silhouette-derived marker data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.shadow_tracker.pilot_metrics import closure_metrics, replay_difference

_MODEL = "docs/development/full_body_models/full_body_spec_v1.json"
_CALIBRATION = "docs/development/full_body_models/evidence/fb4_calibration/mujoco"
_TRANSLATIONS = ("TranslationInputX", "TranslationInputY", "TranslationInputZ")


def _execute(
    model_bytes: bytes,
    q0: np.ndarray,
    times: np.ndarray,
    capture: Any,
    offsets: dict[str, Any],
    rtol: float,
) -> tuple[dict[str, Any], Any]:
    from src.engines.physics_engines.mujoco.python.full_body_ik import MujocoFullBodyIK
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.shared.python.motion_matching.contact_law import GroundPlane
    from src.shared.python.motion_matching.full_body_forward_dynamics import (
        RolloutOptions,
        calibrate_ground_height_at_address,
        simulate_full_body_forward,
    )

    started = time.perf_counter()
    model = NativeMujocoFullBodyModel(model_bytes)
    names = model.coordinate_order
    ground_height = calibrate_ground_height_at_address(model, q0)
    ground = model.ground_plane
    model.ground_plane = GroundPlane(normal=ground.normal, height_m=ground_height)
    initial_ground = model.ground_plane
    ik = MujocoFullBodyIK(model_bytes.decode("utf-8"))
    ik.pose_fn(q0)
    ik_closure = float(np.linalg.norm(ik.closure_residuals(q0)))
    model.accelerations(
        dict(zip(names, q0, strict=True)),
        dict.fromkeys(names, 0.0),
        dict.fromkeys(names, 0.0),
    )
    initial_displacement, _ = model.closure_errors()
    native = model.model
    addresses = [(int(native.joint(name).qposadr[0]), name) for name in names]
    native_order = [name for _, name in sorted(addresses)]
    theta = np.zeros((len(names), 7))
    result = simulate_full_body_forward(
        model=model,
        ik_adapter=ik,
        theta=theta,
        time_grid=times,
        initial_state=(q0.copy(), np.zeros_like(q0)),
        marker_offsets=offsets,
        capture=capture,
        options=RolloutOptions(rtol=rtol, atol=rtol / 100),
    )
    details: dict[str, Any] = {
        "solver_status": result.status,
        "frames_returned": len(result.time_s),
        "time_grid_exact": bool(np.array_equal(times, result.time_s)),
        "rtol": rtol,
        "atol": rtol / 100,
        "ground_height_m": ground_height,
        "ground_unchanged": model.ground_plane == initial_ground,
        "wall_time_s": time.perf_counter() - started,
        "contact": result.contact_audit.as_dict(),
        "max_closure_translation_m": result.max_closure_translation_m,
        "max_closure_rotation_rad": result.max_closure_rotation_rad,
        "legacy_mixed_unit_closure_value": result.max_closure_residual_m,
        "native_qpos_order": native_order,
        "declared_order_matches_native_qpos": names == native_order,
        "ik_initial_grip_translation_m": ik_closure,
        "dynamics_initial_grip_translation_m": float(
            np.linalg.norm(initial_displacement[:3])
        ),
    }
    if result.status == "success":
        errors = []
        for q, v in zip(result.q, result.qd, strict=True):
            model.accelerations(
                dict(zip(names, q, strict=True)),
                dict(zip(names, v, strict=True)),
                dict.fromkeys(names, 0.0),
            )
            displacement, _ = model.closure_errors()
            errors.append(displacement)
        details.update(closure_metrics(errors))
    details["native_state"] = {"nq": native.nq, "nv": native.nv}
    return details, result


def run_probe(root: Path, frames: int = 10) -> dict[str, Any]:
    """Run three fresh passive rollouts and return auditable, unqualified evidence.

    Requires the existing model, measured C3D and IK assets. Postcondition: no
    source assets or physics models are modified; the receipt never claims
    scientific qualification, even when integration and repeatability succeed.
    """
    if type(frames) is not int or not 2 <= frames <= 100:
        raise ValueError("frames must be an integer between 2 and 100")
    from src.shared.python.motion_matching.tour_capture_contract import (
        load_tour_capture,
    )

    paths = [
        _MODEL,
        f"{_CALIBRATION}/ik_trajectory.npz",
        f"{_CALIBRATION}/calibrated_offsets.json",
        "data/C3D_TA_Driver.c3d",
    ]
    assets = {name: (root / name).read_bytes() for name in paths}
    model_bytes = assets[_MODEL]
    specification = json.loads(model_bytes)
    names = specification["coordinate_order"]
    with np.load(root / paths[1]) as archive:
        q0 = archive["q"][0].copy()
    offsets = json.loads(assets[paths[2]])["marker_offsets"]
    capture = load_tour_capture(root / paths[3])
    if frames > capture.frames:
        raise ValueError("frames exceed available capture")
    times = capture.time_s[:frames] - capture.time_s[0]
    outputs = [
        _execute(model_bytes, q0, times, capture, offsets, rtol)
        for rtol in (1e-5, 1e-5, 1e-6)
    ]
    runs = [item[0] for item in outputs]
    receipt: dict[str, Any] = {
        "schema_version": "shadow-tracker/model-probe/1.0.0",
        "scientifically_qualified": False,
        "canonical_adapter_verified": False,
        "experiment": "passive-zero-control-zero-initial-velocity-short-window",
        "duration_s": float(times[-1]),
        "coordinate_order": names,
        "native_state": runs[0]["native_state"],
        "body_count": len(specification["bodies"]),
        "joint_count": len(specification["joints"]),
        "input_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in assets.items()
        },
        "runs": runs,
    }
    if all(run["solver_status"] == "success" for run in runs):
        first, repeat, refined = (item[1] for item in outputs)
        indices = tuple(names.index(name) for name in _TRANSLATIONS)
        for name, candidate in (("repeat", repeat), ("refined", refined)):
            if not np.array_equal(first.time_s, candidate.time_s):
                raise ValueError("Replay time grids differ")
            receipt[f"{name}_q"] = replay_difference(first.q, candidate.q, indices)
            receipt[f"{name}_v"] = replay_difference(first.qd, candidate.qd, indices)
    return receipt


def main() -> int:
    """Write an experiment receipt; exit status reports execution, not validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()
    receipt = run_probe(args.root, args.frames)
    import mujoco

    receipt["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }
    receipt["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.root, text=True
    ).strip()
    receipt["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sources = (
        "scripts/shadow_tracker/model_probe.py",
        "scripts/shadow_tracker/pilot_metrics.py",
        "src/engines/physics_engines/mujoco/python/full_body_model.py",
        "src/engines/physics_engines/mujoco/python/full_body_ik.py",
        "src/engines/physics_engines/mujoco/python/full_body_mjcf.py",
        "src/engines/physics_engines/mujoco/python/native_model.py",
        "src/shared/python/motion_matching/full_body_forward_dynamics.py",
        "src/shared/python/motion_matching/contact_law.py",
    )
    receipt["source_sha256"] = {
        path: hashlib.sha256((args.root / path).read_bytes()).hexdigest()
        for path in sources
    }
    receipt["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return 0 if all(run["solver_status"] == "success" for run in receipt["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
