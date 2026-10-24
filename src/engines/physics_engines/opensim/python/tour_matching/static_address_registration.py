"""Address-hold helpers for tour-marker registration (#13434).

``quiet_segment_frames`` finds the held address at the start of a capture,
``waist_cluster_frame`` builds the pelvis axes from the waist belt, and
``solve_pelvis_rotations`` turns those axes into OpenSim pelvis coordinates.
They seed the heading of the OS-3b calibration and the OS-9 prior-constrained
registration.

A trunk-only static re-registration (lumbar locked, trunk markers re-placed
from the hold) was tried here first and rejected: it left the limb markers on
the mirrored OS-3b solution and doubled the swing IK error (#13434).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.tour_capture_contract import TourCapture


TRUNK_MARKERS: tuple[str, ...] = (
    "WaistLeft",
    "WaistRight",
    "WaistLBack",
    "WaistRBack",
    "BackTop",
    "BackLeft",
    "BackRight",
    "HeadTop",
    "HeadFront",
    "HeadSide",
    "LShoulderTop",
    "LShoulderBack",
    "RShoulderTop",
    "RShoulderBack",
)

# ---------------------------------------------------------------------------
# Law of Demeter (LoD) single-hop OpenSim helpers
# ---------------------------------------------------------------------------


def _get_body_transform(
    model: Any, state: Any, body_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Single-hop query for body transform (rotation matrix 3x3, origin 3) in ground."""
    body = model.getBodySet().get(body_name)
    xform = body.getTransformInGround(state)
    rot = xform.R().asMat33()
    r = np.array([[rot.get(i, j) for j in range(3)] for i in range(3)], dtype=float)
    p = np.array(xform.p().to_numpy(), dtype=float)
    return r, p


# ---------------------------------------------------------------------------
# Quiet Segment & Waist Cluster
# ---------------------------------------------------------------------------


def quiet_segment_frames(
    capture: TourCapture,
    trunk_markers: Sequence[str] = TRUNK_MARKERS,
    threshold_mps: float = 0.05,
    min_duration_s: float = 0.05,
) -> np.ndarray:
    """Find initial quiet frames where median trunk marker speed < threshold_mps.

    Requires at least min_duration_s of quiet segment from the first frame.
    """
    require(capture.frames > 1, "Capture must have at least 2 frames")
    require(threshold_mps > 0.0, "Speed threshold must be positive")
    require(min_duration_s > 0.0, "Minimum duration must be positive")

    indices = [capture.index(m) for m in trunk_markers if m in capture.labels]
    require(len(indices) > 0, "At least one trunk marker must exist in capture")

    dt = np.diff(capture.time_s)
    pts = capture.points_m[:, indices, :]
    diffs = np.diff(pts, axis=0)
    speeds = np.linalg.norm(diffs / dt[:, None, None], axis=-1)

    valid_mask = capture.valid[:-1, indices] & capture.valid[1:, indices]
    masked_speeds = np.where(valid_mask, speeds, np.nan)
    median_speeds = np.nanmedian(masked_speeds, axis=1)

    exceed = np.where(median_speeds >= threshold_mps)[0]
    last_quiet_idx = int(exceed[0]) if len(exceed) > 0 else len(median_speeds)

    quiet_count = last_quiet_idx + 1
    duration = float(capture.time_s[quiet_count - 1] - capture.time_s[0])
    require(
        duration >= min_duration_s,
        f"Quiet segment duration {duration:.4f} s < minimum {min_duration_s:.4f} s",
    )

    frames = np.arange(quiet_count, dtype=int)
    ensure(len(frames) >= 2, "Quiet segment must span at least 2 frames")
    return frames


def waist_cluster_frame(waist_points: Mapping[str, np.ndarray]) -> np.ndarray:
    """Compute OpenSim pelvis body orientation from waist cluster anatomical axes.

    Anterior: a = normalize(mean(WaistLeft, WaistRight) - mean(WaistLBack, WaistRBack))
    Left: l = normalize(mean(WaistLeft, WaistLBack) - mean(WaistRight, WaistRBack))
    OpenSim pelvis convention: x anterior (a), z right (-l), y superior (z cross x).
    """
    wl = np.asarray(waist_points["WaistLeft"], dtype=float)
    wr = np.asarray(waist_points["WaistRight"], dtype=float)
    wlb = np.asarray(waist_points["WaistLBack"], dtype=float)
    wrb = np.asarray(waist_points["WaistRBack"], dtype=float)

    front = 0.5 * (wl + wr)
    back = 0.5 * (wlb + wrb)
    a = front - back
    norm_a = float(np.linalg.norm(a))
    require(norm_a > 1e-6, "Anterior waist vector norm must be positive")
    a = a / norm_a

    l_mid = 0.5 * (wl + wlb)
    r_mid = 0.5 * (wr + wrb)
    l_raw = l_mid - r_mid
    l_orth = l_raw - float(np.dot(l_raw, a)) * a
    norm_l = float(np.linalg.norm(l_orth))
    require(norm_l > 1e-6, "Lateral waist vector norm must be positive")
    l_orth = l_orth / norm_l

    x = a
    z = -l_orth
    y = np.cross(z, x)
    norm_y = float(np.linalg.norm(y))
    require(norm_y > 1e-6, "Superior waist vector norm must be positive")
    y = y / norm_y

    R = np.column_stack([x, y, z])
    det = float(np.linalg.det(R))
    ensure(abs(det - 1.0) < 1e-6, "Waist cluster frame must be right-handed (det = +1)")
    ensure(
        bool(np.allclose(R.T @ R, np.eye(3), atol=1e-6)),
        "Waist cluster frame must be orthonormal",
    )
    return R


# ---------------------------------------------------------------------------
# Coordinate Solvers & Helpers
# ---------------------------------------------------------------------------


def solve_pelvis_rotations(
    model: Any, target_R: np.ndarray
) -> tuple[dict[str, float], float]:
    """Solve pelvis_tilt, pelvis_list, pelvis_rotation to match target_R in ground."""
    require(target_R.shape == (3, 3), "target_R must be 3x3")
    state = model.initSystem()
    coords = model.getCoordinateSet()
    c_tilt = coords.get("pelvis_tilt")
    c_list = coords.get("pelvis_list")
    c_rot = coords.get("pelvis_rotation")

    def residual(angles: np.ndarray) -> np.ndarray:
        c_tilt.setValue(state, float(angles[0]), False)
        c_list.setValue(state, float(angles[1]), False)
        c_rot.setValue(state, float(angles[2]), False)
        model.realizePosition(state)
        r_current, _ = _get_body_transform(model, state, "pelvis")
        return Rotation.from_matrix(r_current.T @ target_R).as_rotvec()

    seed = np.zeros(3)
    sol = least_squares(residual, seed, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    res_norm = float(np.linalg.norm(sol.fun))
    require(res_norm < 1e-6, f"Pelvis rotation residual {res_norm:.2e} >= 1e-6 rad")

    out_coords = {
        "pelvis_tilt": float(sol.x[0]),
        "pelvis_list": float(sol.x[1]),
        "pelvis_rotation": float(sol.x[2]),
    }
    return out_coords, res_norm
