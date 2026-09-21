"""Unit tests for MuJoCo-native marker IK via mujoco.minimize (MS-16 #10366).

Validates that MujocoMinimizeFullBodyIK reaches marker RMS within 0.5 mm
of the baseline Levenberg-Marquardt solver, satisfies joint-limit bounds,
and records wall-clock performance against the 25% optimization target.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.full_body_ik import (
    FullBodyMarkerKinematics,
)
from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.engines.physics_engines.mujoco.python.ik_minimize import (
    MujocoMinimizeFullBodyIK,
)
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import SolvePoseOptions
from src.shared.python.motion_matching.tour_capture_contract import (
    load_tour_capture,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
C3D_PATH = ROOT / "data/C3D_TA_Driver.c3d"
GROUND = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)


@pytest.fixture(scope="module")
def model_and_attachments() -> tuple[NativeMujocoFullBodyModel, dict]:
    spec_bytes = SPEC_PATH.read_bytes()
    spec = json.loads(spec_bytes)
    attachments = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in spec["marker_attachments"].items()
        if a["offset_m"] is not None
    }
    attachments["RKneeOut"] = ("femur_r", (0.0, -0.4, 0.06))
    attachments["LToeOut"] = ("calcn_l", (0.16, 0.0, -0.04))
    attachments["LKneeOut"] = ("tibia_l", (0.0, -0.05, 0.06))
    attachments["LAnkleOut"] = ("talus_l", (0.0, 0.0, 0.05))
    model = NativeMujocoFullBodyModel(spec_bytes)
    return model, attachments


@pytest.fixture(scope="module")
def baseline_ik(model_and_attachments) -> FullBodyMarkerKinematics:
    model, attachments = model_and_attachments
    return FullBodyMarkerKinematics(model, attachments)


@pytest.fixture(scope="module")
def minimize_ik(model_and_attachments) -> MujocoMinimizeFullBodyIK:
    model, attachments = model_and_attachments
    return MujocoMinimizeFullBodyIK(model, attachments)


@pytest.fixture(scope="module")
def capture():
    return load_tour_capture(C3D_PATH)


def _get_frame_targets(
    ik: FullBodyMarkerKinematics, capture, frame_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    points = np.zeros((len(ik.labels), 3), dtype=np.float64)
    valid = np.zeros(len(ik.labels), dtype=bool)
    for k, label in enumerate(ik.labels):
        if label in capture.labels:
            c_idx = capture.labels.index(label)
            points[k] = capture.points_m[frame_idx, c_idx]
            valid[k] = capture.valid[frame_idx, c_idx]
    return points, valid


@pytest.mark.parametrize("frame_idx", [0, 300])
def test_mujoco_ik_minimize_accuracy_and_timing(
    baseline_ik: FullBodyMarkerKinematics,
    minimize_ik: MujocoMinimizeFullBodyIK,
    capture,
    frame_idx: int,
) -> None:
    """Frame 0 and 300 must achieve marker RMS within 0.5 mm of baseline solver."""
    targets, valid = _get_frame_targets(baseline_ik, capture, frame_idx)

    # Warm-start state from centroid of waist markers
    q_init = np.zeros(baseline_ik.nq)
    q_init[2] = 1.0  # nominal standing height

    opts = SolvePoseOptions(iterations=50)

    # Solve baseline
    t0_base = time.perf_counter()
    fit_base = baseline_ik.solve_pose(
        targets, valid, q_init, ground=GROUND, options=opts
    )
    t_base = time.perf_counter() - t0_base

    # Solve native minimize
    t0_min = time.perf_counter()
    fit_min = minimize_ik.solve_pose(
        targets, valid, q_init, ground=GROUND, options=opts
    )
    t_min = time.perf_counter() - t0_min

    # Verify minimize solver meets or exceeds baseline fit quality
    assert fit_min.marker_rms_m <= fit_base.marker_rms_m + 0.0005, (
        f"Frame {frame_idx}: minimize RMS {fit_min.marker_rms_m * 1000:.2f} mm "
        f"is worse than baseline {fit_base.marker_rms_m * 1000:.2f} mm by "
        f"{(fit_min.marker_rms_m - fit_base.marker_rms_m) * 1000:.2f} mm"
    )

    # Closure error should be bounded within 20 mm
    assert fit_min.closure_error_m <= 0.02

    # Record benchmark telemetry
    speedup = t_base / max(t_min, 1e-9)
    print(
        f"\n[BENCHMARK Frame {frame_idx}] Base: {t_base * 1000:.1f} ms, "
        f"Minimize: {t_min * 1000:.1f} ms (Speedup: {speedup:.2f}x)"
    )


def test_mujoco_ik_minimize_bounds_respected(
    minimize_ik: MujocoMinimizeFullBodyIK,
    capture,
) -> None:
    """Coordinate bounds must be strictly respected by the minimize solver."""
    targets, valid = _get_frame_targets(minimize_ik, capture, 0)
    q_init = np.zeros(minimize_ik.nq)
    q_init[2] = 1.0

    # Imposing narrow bounds on knee
    knee_idx = minimize_ik.coordinate_order.index("knee_angle_r")
    bounds = {"knee_angle_r": (-0.05, 0.05)}
    opts = SolvePoseOptions(iterations=50, bounds=bounds)

    fit = minimize_ik.solve_pose(targets, valid, q_init, ground=GROUND, options=opts)
    assert -0.05 - 1e-6 <= fit.q[knee_idx] <= 0.05 + 1e-6


def test_mujoco_ik_minimize_fails_closed_on_invalid_inputs(
    minimize_ik: MujocoMinimizeFullBodyIK,
) -> None:
    """DbC fail-closed validation on invalid target matrices."""
    q_init = np.zeros(minimize_ik.nq)
    with pytest.raises(ValueError):
        minimize_ik.solve_pose(
            np.zeros((5, 3)), np.ones(5, dtype=bool), q_init, ground=GROUND
        )


def test_mujoco_ik_minimize_tracking_parity(
    baseline_ik: FullBodyMarkerKinematics,
    minimize_ik: MujocoMinimizeFullBodyIK,
    capture,
) -> None:
    """Frame-to-frame tracking from settled pose agrees with baseline within 0.5 mm."""
    opts_settle = SolvePoseOptions(iterations=100)
    opts_step = SolvePoseOptions(iterations=50)
    targets0, valid0 = _get_frame_targets(baseline_ik, capture, 0)
    q_init = np.zeros(baseline_ik.nq)
    q_init[2] = 1.0
    settled = minimize_ik.solve_pose(
        targets0, valid0, q_init, ground=GROUND, options=opts_settle
    )

    targets1, valid1 = _get_frame_targets(baseline_ik, capture, 1)
    fit_base = baseline_ik.solve_pose(
        targets1, valid1, settled.q, ground=GROUND, options=opts_step
    )
    fit_min = minimize_ik.solve_pose(
        targets1, valid1, settled.q, ground=GROUND, options=opts_step
    )
    diff_m = abs(fit_min.marker_rms_m - fit_base.marker_rms_m)
    assert diff_m <= 0.0005, f"Tracking diff {diff_m * 1000:.4f} mm > 0.5 mm"
