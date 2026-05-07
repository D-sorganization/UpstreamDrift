"""Drake vs Simscape grip-trajectory equivalence test (closes #4123).

Cross-engine parity spec §2.2 requires every physics engine to reproduce
the Simscape ground-truth grip trajectory to within **5 mm RMSE** of
position and **1 degree** of orientation at three canonical poses:
**address**, **top-of-backswing**, and **impact**.

This module pins that contract for the Drake float-pathway forward-sim
wrapper landed in PR #4169
(`src.engines.physics_engines.drake.python.motion_matching.simulate.
simulate_with_coefficients`).

Design
------
* Three fixed poses are loaded from a checked-in Simscape CSV under
  ``src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/Scripts/
  Dataset Generator/golf_swing_dataset_20251030/trial_001_*.csv``.
* The grip anchor in Simscape land is the **club butt** position
  (``LHCalcsLogs_ButtPosition_*``); the grip orientation is reconstructed
  from the shaft axis (butt → clubhead) plus the clubface unit vector
  (``ClubLogs_CFUnitVector_*``) so that the comparison is well-defined
  even though Simscape does not expose a grip quaternion directly.
* Drake is invoked with ``theta = 0`` (i.e. gravity + initial pose only)
  per the issue body — this is the cleanest equivalence check because it
  factors out actuator-polynomial dynamics and isolates the URDF /
  forward-kinematics stack.
* The test deliberately fails loudly when Drake disagrees with the
  Simscape reference: per the issue body, *"if Drake's simulation can't
  reproduce the Simscape ground truth within tolerance, the test fails —
  that's the whole point"*. The PR body documents any genuine divergence
  and links follow-up issues for fixes.

The test is gated on ``@pytest.mark.requires_drake`` and additionally
performs a runtime ``importorskip`` on
``src.engines.physics_engines.drake.python.motion_matching.simulate``
because that module ships in PR #4169 (still open at the time this
fixture lands) and is not present on a fresh ``main`` checkout.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Fixture data — Simscape ground truth CSV
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SIMSCAPE_CSV: Path = (
    REPO_ROOT
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "matlab"
    / "Scripts"
    / "Dataset Generator"
    / "golf_swing_dataset_20251030"
    / "trial_001_20251030_202704.csv"
)

# Column indices in the trial_001 CSV (1956 columns total). Discovered by
# inspecting the header in scripts/inspect_drake_vs_simscape.py.
COL_TIME = 0
COL_BUTT = (768, 769, 770)  # LHCalcsLogs_ButtPosition_{1,2,3} — grip anchor
COL_CHEAD = (305, 306, 307)  # ClubLogs_CHGlobalPosition_{1,2,3}
COL_CFACE = (302, 303, 304)  # ClubLogs_CFUnitVector_dim{1,2,3}

#: Pose name → row index in the 31-sample CSV (0.01 s grid, 0..0.30 s).
#: Address is the first sample. Top-of-backswing for this trial is at the
#: row with the maximum butt-Z (club is highest above the ground). Impact
#: is the row whose clubhead-speed is closest to the canonical 100 mph
#: gate; for this fitted-but-noisy fixture the cleanest comparable index
#: is the row immediately preceding the chaotic blow-up at the end.
POSES: dict[str, int] = {
    "address": 0,
    "top_of_backswing": 10,
    "impact": 20,
}

#: Acceptance gates from cross-engine §2.2 / issue #4123.
RMSE_POSITION_GATE_M = 5.0e-3  # 5 mm
ORIENT_GATE_RAD = np.deg2rad(1.0)  # 1 degree geodesic


@dataclass(frozen=True)
class SimscapePose:
    """A single Simscape reference pose (row of the CSV)."""

    name: str
    t: float
    butt: NDArray[np.float64]  # (3,) world position (m)
    grip_quat: NDArray[np.float64]  # (4,) [w,x,y,z]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``vec`` divided by its 2-norm; raise if zero/non-finite."""
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-12:
        msg = f"_normalize received a degenerate vector: {vec!r}"
        raise ValueError(msg)
    return np.asarray(vec, dtype=np.float64) / norm


def _rotation_matrix_to_quaternion(rmat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a 3x3 rotation matrix to a ``[w, x, y, z]`` quaternion.

    Uses the numerically stable branch-by-trace algorithm (Shepperd 1978).
    """
    if rmat.shape != (3, 3):
        msg = f"rmat must be (3,3); got {rmat.shape}"
        raise ValueError(msg)
    m = rmat
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _grip_quat_from_simscape_row(
    butt: NDArray[np.float64],
    clubhead: NDArray[np.float64],
    cface: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reconstruct a grip quaternion from Simscape's shaft + clubface.

    The grip frame's body axes are defined as:
      * **z_grip** = unit vector from butt → clubhead (down the shaft);
      * **x_grip** = unit clubface normal (``CFUnitVector``), projected
        onto the plane orthogonal to ``z_grip`` and re-normalised;
      * **y_grip** = ``z_grip × x_grip`` (right-handed completion).

    This matches the Drake URDF's ``club_grip`` body frame convention
    documented in DRAKE_PARITY_SPEC.md §4.2.
    """
    z_axis = _normalize(np.asarray(clubhead, dtype=np.float64) - np.asarray(butt))
    cface_unit = _normalize(np.asarray(cface, dtype=np.float64))
    # Project clubface vector to the shaft-orthogonal plane.
    x_axis = cface_unit - np.dot(cface_unit, z_axis) * z_axis
    x_axis = _normalize(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rmat = np.column_stack((x_axis, y_axis, z_axis))
    return _rotation_matrix_to_quaternion(rmat)


def _quaternion_geodesic_angle_rad(
    q_a: NDArray[np.float64],
    q_b: NDArray[np.float64],
) -> float:
    """Geodesic angle (radians) between two unit quaternions ``q_a``, ``q_b``.

    Quaternions are expected in ``[w, x, y, z]`` order. The result is in
    ``[0, pi]`` and treats ``q`` and ``-q`` as the same rotation.
    """
    a = np.asarray(q_a, dtype=np.float64)
    b = np.asarray(q_b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def _load_simscape_poses(csv_path: Path) -> list[SimscapePose]:
    """Load the three canonical poses from the Simscape CSV."""
    if not csv_path.is_file():
        msg = f"Simscape ground-truth CSV not found at {csv_path}"
        raise FileNotFoundError(msg)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # discard header
        rows = list(reader)
    out: list[SimscapePose] = []
    for name, idx in POSES.items():
        if idx >= len(rows):
            msg = f"pose {name!r} index {idx} out of range (CSV has {len(rows)} rows)"
            raise IndexError(msg)
        row = rows[idx]
        t = float(row[COL_TIME])
        butt = np.array([float(row[c]) for c in COL_BUTT], dtype=np.float64)
        chead = np.array([float(row[c]) for c in COL_CHEAD], dtype=np.float64)
        cface = np.array([float(row[c]) for c in COL_CFACE], dtype=np.float64)
        quat = _grip_quat_from_simscape_row(butt, chead, cface)
        out.append(SimscapePose(name=name, t=t, butt=butt, grip_quat=quat))
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simscape_poses() -> list[SimscapePose]:
    """The three reference poses (address / top-of-backswing / impact)."""
    return _load_simscape_poses(SIMSCAPE_CSV)


@pytest.fixture(scope="module")
def drake_simulate_module():
    """Import the Drake forward-sim wrapper or skip.

    Skips when either ``pydrake`` itself is missing OR when the
    ``simulate`` module has not been merged to ``main`` yet (PR #4169).
    """
    pytest.importorskip("pydrake", reason="pydrake is required for #4123")
    return pytest.importorskip(
        "src.engines.physics_engines.drake.python.motion_matching.simulate",
        reason=(
            "Drake simulate_with_coefficients ships in PR #4169 — skipping "
            "until that lands on main"
        ),
    )


@pytest.fixture(scope="module")
def drake_simout(drake_simulate_module):
    """Run Drake forward-sim once with theta=0 and reuse for all three poses."""
    sim_mod = drake_simulate_module
    options = sim_mod.SimOptions(
        simulation_time_s=0.30,
        sample_rate_hz=1000.0,
        time_step_s=1.0e-3,
    )
    # Theta sized to the URDF actuator count; pad generously so the
    # simulate wrapper's actuator-vs-theta DOF check picks the URDF's
    # native count and runs at zero torque.
    n_actuators_max = 64  # safe upper bound for the canonical humanoid
    theta = np.zeros(n_actuators_max * sim_mod.COEFFS_PER_JOINT, dtype=np.float64)
    return sim_mod.simulate_with_coefficients(theta, options=options)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_drake
@pytest.mark.integration
def test_simscape_fixture_loads(simscape_poses: list[SimscapePose]) -> None:
    """The Simscape ground-truth fixture parses cleanly."""
    assert len(simscape_poses) == 3, "Expected three canonical poses"
    names = {p.name for p in simscape_poses}
    assert names == {"address", "top_of_backswing", "impact"}
    for p in simscape_poses:
        assert np.all(np.isfinite(p.butt)), f"non-finite butt for {p.name}"
        assert np.all(np.isfinite(p.grip_quat)), f"non-finite quat for {p.name}"
        assert abs(np.linalg.norm(p.grip_quat) - 1.0) < 1e-6


@pytest.mark.requires_drake
@pytest.mark.integration
@pytest.mark.parametrize("pose_name", list(POSES.keys()))
def test_drake_grip_matches_simscape(
    pose_name: str,
    simscape_poses: list[SimscapePose],
    drake_simout,
) -> None:
    """Drake's grip pose at ``pose_name`` matches Simscape within the gates.

    The acceptance criterion (cross-engine §2.2):

    * grip-position **RMSE < 5 mm** vs the Simscape reference, and
    * grip-orientation **geodesic angle < 1 degree** vs the Simscape
      reconstructed quaternion.

    The RMSE is computed across the three Cartesian axes for the matched
    sample, which collapses to the per-pose Euclidean error magnitude
    (``sqrt(mean((p_drake - p_simscape)**2))``). This is the same
    statistic the parity-leaderboard in §3 of the spec scores against.
    """
    pose = next(p for p in simscape_poses if p.name == pose_name)

    # Snap Simscape time to the closest Drake output sample.
    drake_t = drake_simout.time
    j = int(np.argmin(np.abs(drake_t - pose.t)))

    drake_grip = np.asarray(drake_simout.grip[j], dtype=np.float64)
    drake_quat = np.asarray(drake_simout.grip_quat[j], dtype=np.float64)

    # Drake may emit NaN columns for grip if the URDF didn't expose the
    # ``club_grip`` body — fail informatively in that case.
    if not np.all(np.isfinite(drake_grip)):
        pytest.fail(
            f"Drake produced non-finite grip at t={pose.t:.3f}s "
            f"(pose={pose.name!r}); SimOut.metadata={drake_simout.metadata!r}"
        )

    # ------------ Position RMSE ----------------------------------------
    delta = drake_grip - pose.butt
    rmse_pos = float(np.sqrt(np.mean(delta**2)))
    err_msg_pos = (
        f"[#4123 equivalence] pose={pose.name!r} t={pose.t:.3f}s  "
        f"grip-position RMSE={rmse_pos * 1e3:.2f} mm exceeds the 5 mm gate. "
        f"drake={drake_grip.tolist()}  simscape={pose.butt.tolist()}  "
        f"delta={delta.tolist()} m"
    )
    assert rmse_pos < RMSE_POSITION_GATE_M, err_msg_pos

    # ------------ Orientation geodesic ---------------------------------
    if np.all(np.isfinite(drake_quat)) and np.linalg.norm(drake_quat) > 0.0:
        angle = _quaternion_geodesic_angle_rad(drake_quat, pose.grip_quat)
        err_msg_ori = (
            f"[#4123 equivalence] pose={pose.name!r} t={pose.t:.3f}s  "
            f"grip-orientation geodesic={np.rad2deg(angle):.3f} deg "
            f"exceeds the 1.000 deg gate. drake_quat={drake_quat.tolist()}  "
            f"simscape_quat={pose.grip_quat.tolist()}"
        )
        assert angle < ORIENT_GATE_RAD, err_msg_ori
    else:
        pytest.fail(
            f"Drake produced a non-finite grip_quat at t={pose.t:.3f}s "
            f"(pose={pose.name!r}); cannot evaluate orientation gate."
        )
