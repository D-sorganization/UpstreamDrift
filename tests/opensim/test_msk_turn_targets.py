"""Pelvis/upper-trunk turn targets and planted feet for the OSV-9 retarget (#12042)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python import msk_club as mc
from src.engines.physics_engines.opensim.python import msk_club_calibration as cal
from src.engines.physics_engines.opensim.python import msk_turn_targets as tt
from src.shared.python.swing_comparison.events import SwingEvents
from src.shared.python.swing_comparison.turn import marker_turn_lines, model_turn_lines

pytestmark = pytest.mark.unit

SWING = mc.REPO_ROOT / "tests" / "fixtures" / "club_face" / "swing_q_driver.npz"
#: Tracked segment turn must reproduce a synthetic marker turn this closely
#: (new bound; the source landmarks held at address pull against the turn and
#: leave 0.2 deg on the pelvis and 1.4 deg on the trunk at 15 and 35 deg).
TURN_TOL_DEG = 2.0
#: Grip tolerances of ``test_msk_club_tracking`` (unchanged, not loosened).
GRIP_TOL_M = 0.005
TRAIL_TOL_M = 0.02


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array(
        [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    )


def _synthetic_markers(
    pelvis_turn_deg: np.ndarray, trunk_turn_deg: np.ndarray
) -> dict[str, np.ndarray]:
    """Waist and back marker pairs rigidly turned about vertical (native world).

    The golfer faces -X with the target toward -Y, so the left markers sit at
    -Y. A backswing turn is clockwise seen from above: yaw = -turn.
    """
    base = {
        "WaistLeft": ([0.0, -0.15, 1.0], pelvis_turn_deg),
        "WaistRight": ([0.0, 0.15, 1.0], pelvis_turn_deg),
        "BackLeft": ([0.1, -0.12, 1.35], trunk_turn_deg),
        "BackRight": ([0.1, 0.12, 1.35], trunk_turn_deg),
    }
    return {
        name: np.array([_rot_z(-turn) @ np.array(p) for turn in turns])
        for name, (p, turns) in base.items()
    }


def _events(t: np.ndarray) -> SwingEvents:
    n = len(t) - 1
    return SwingEvents(0, float(t[0]), n, float(t[n]), n, float(t[n]), n, float(t[n]))


# ------------------------------------------------------------- pure targets
def test_turn_targets_sample_the_shared_marker_lines() -> None:
    t = np.linspace(0.0, 0.5, 51)
    pelvis, trunk = 40.0 * t / 0.5, 90.0 * t / 0.5
    lines = marker_turn_lines(_synthetic_markers(pelvis, trunk), t, _events(t))
    targets = tt.turn_targets_from_lines(lines, [0.0, 0.25, 0.5, 0.6])
    np.testing.assert_allclose(targets.pelvis_deg[:3], [0.0, 20.0, 40.0], atol=1e-9)
    np.testing.assert_allclose(
        targets.upper_trunk_deg[:3], [0.0, 45.0, 90.0], atol=1e-9
    )
    assert np.isnan(targets.pelvis_deg[3])  # outside the capture: no target, not 0


def test_turn_targets_contracts() -> None:
    with pytest.raises(ValueError, match="equal length"):
        tt.TurnTargets(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="finite or NaN"):
        tt.TurnTargets(np.array([np.inf]), np.zeros(1))
    with pytest.raises(ValueError, match="weight_rad"):
        tt.TurnTargets(np.zeros(1), np.zeros(1), weight_rad=0.0)
    t = np.linspace(0.0, 0.5, 51)
    lines = marker_turn_lines(
        _synthetic_markers(np.zeros(51), np.zeros(51)), t, _events(t)
    )
    with pytest.raises(ValueError, match="times_s"):
        tt.turn_targets_from_lines(lines, [[0.0]])


class _FakeProbe:
    """Pose probe whose pelvis and torso are turned by set yaw angles."""

    def __init__(self) -> None:
        self.yaw = {"pelvis": 0.0, "upper_trunk": 0.0}

    def body(self, name: str) -> np.ndarray:
        pose = np.eye(4)
        # Rajagopal world: Y up; native yaw psi is a rotation about +y by +psi
        # (native z = Rajagopal y, native x = -Rajagopal x).
        segment = (
            "upper_trunk" if name in ("torso", "humerus_l", "humerus_r") else "pelvis"
        )
        a = self.yaw[segment]
        rot = np.array(
            [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        )
        local = {
            "femur_l": [0, 0.9, -0.09],
            "femur_r": [0, 0.9, 0.09],
            "humerus_l": [0, 1.4, -0.18],
            "humerus_r": [0, 1.4, 0.18],
        }.get(name, [0, 1.2, 0])
        pose[:3, :3] = rot
        pose[:3, 3] = rot @ np.array(local, dtype=float)
        return pose


def test_segment_yaw_follows_the_native_line_convention() -> None:
    probe = _FakeProbe()
    base = tt.segment_yaws(probe)
    probe.yaw = {"pelvis": math.radians(-30.0), "upper_trunk": math.radians(-80.0)}
    turned = tt.segment_yaws(probe)
    for segment, expected in (("pelvis", 30.0), ("upper_trunk", 80.0)):
        turn = -math.degrees(tt.wrap_angle(turned[segment] - base[segment]))
        assert turn == pytest.approx(expected, abs=1e-9)
    # The reporting points give the same turn through the shared module.
    points = [tt.model_turn_points(_FakeProbe())] * 2 + [
        tt.model_turn_points(probe)
    ] * 2
    stacked = {k: np.array([p[k] for p in points]) for k in points[0]}
    t = np.array([0.0, 0.1, 0.2, 0.3])
    lines = model_turn_lines(stacked, t, _events(t))
    assert lines.pelvis.turn_deg[-1] == pytest.approx(30.0, abs=1e-9)
    assert lines.upper_trunk.turn_deg[-1] == pytest.approx(80.0, abs=1e-9)
    assert lines.x_factor.turn_deg[-1] == pytest.approx(50.0, abs=1e-9)


def test_turn_residuals_are_weighted_wrapped_yaw_errors() -> None:
    reference = {"pelvis": 0.0, "upper_trunk": 3.0}
    targets = {"pelvis": 40.0, "upper_trunk": float("nan")}
    yaws = {"pelvis": math.radians(-35.0), "upper_trunk": -3.0}
    res = tt.turn_residuals(yaws, reference, targets, 0.02)
    assert res[0] == pytest.approx(math.radians(5.0) / 0.02)
    assert res[1] == 0.0  # no target on this frame
    wrapped = tt.turn_residuals(
        {"pelvis": math.pi - 0.01, "upper_trunk": 0.0},
        {"pelvis": -math.pi + 0.01, "upper_trunk": 0.0},
        {"pelvis": 0.0, "upper_trunk": 0.0},
        1.0,
    )
    assert wrapped[0] == pytest.approx(-0.02)


def test_tracker_objective_includes_the_pelvis_and_thorax_terms() -> None:
    pytest.importorskip("scipy")
    from src.engines.physics_engines.opensim.python import msk_club_tracking as mt

    class _Probe(_FakeProbe):
        def set(self, values: dict[str, float]) -> None:
            self.yaw = {
                "pelvis": values["pelvis_rotation"],
                "upper_trunk": values["lumbar_rotation"],
            }

    names = ["pelvis_rotation", "lumbar_rotation"]
    tracker = mt._Tracker(_Probe(), names, {s: np.eye(4) for s in "LR"})
    tracker.grips = {s: np.eye(4) for s in "LR"}
    q = np.array([-0.5, -1.2])
    plain = tracker(q)
    tracker.set_turn_reference({})
    tracker.turn = {"pelvis": 20.0, "upper_trunk": 60.0}
    with_turn = tracker(q)
    assert with_turn.size == plain.size + len(tt.TARGET_SEGMENTS)
    expected = [
        (-0.5 + math.radians(20.0)) / tt.TURN_WEIGHT_RAD,
        (-1.2 + math.radians(60.0)) / tt.TURN_WEIGHT_RAD,
    ]
    np.testing.assert_allclose(with_turn[-2:], expected, rtol=1e-9)


# ------------------------------------------------------------- planted feet
@dataclass
class _Capture:
    time_s: np.ndarray
    labels: tuple[str, ...]
    points_m: np.ndarray
    valid: np.ndarray


def _foot_capture(flare_deg: dict[str, float]) -> _Capture:
    """Static capture (Y-up) with feet flared by ``flare_deg`` (toe-out +)."""
    native: dict[str, np.ndarray] = {
        "LWristTop": np.array([-0.4, -0.05, 0.8]),
        "RWristTop": np.array([-0.4, 0.0, 0.8]),
    }
    for prefix, y0, out in (("L", -0.2, -1.0), ("R", 0.2, 1.0)):
        a = math.radians(flare_deg[prefix])
        axis = np.array([-math.cos(a), out * math.sin(a), 0.0])  # toward the ball
        lateral = np.array([-axis[1], axis[0], 0.0])
        lateral = lateral if lateral[1] * out > 0 else -lateral
        ankle_out = np.array([0.0, y0, 0.08]) + 0.04 * lateral
        native[f"{prefix}AnkleOut"] = ankle_out
        toe_mid = ankle_out - 0.04 * lateral + 0.16 * axis
        native[f"{prefix}ToeIn"] = toe_mid - 0.03 * lateral
        native[f"{prefix}ToeOut"] = toe_mid + 0.03 * lateral
    labels = tuple(native)
    frames = 20
    pts = np.array([[native[n] for n in labels]] * frames)
    y_up = np.stack([pts[..., 0], pts[..., 2], -pts[..., 1]], axis=-1)
    return _Capture(
        np.arange(frames) / 360.0, labels, y_up, np.ones(pts.shape[:2], bool)
    )


def test_planted_feet_follow_the_capture_foot_markers() -> None:
    feet = tt.planted_feet_from_capture(_foot_capture({"L": 25.0, "R": 5.0}))
    for suffix, out, flare in (("l", -1.0, 25.0), ("r", 1.0, 5.0)):
        axis = feet.axis[suffix]
        toe_out = math.degrees(math.atan2(out * axis[1], -axis[0]))
        assert toe_out == pytest.approx(flare, abs=0.5)
        assert feet.lateral[suffix][1] * out > 0
    targets = feet.targets(floor_native_z=0.0)
    assert set(targets) == {f"{b}_{s}" for b in cal.FOOT_TEMPLATE for s in "lr"}
    for name, point in targets.items():
        assert point[1] == pytest.approx(cal.FOOT_TEMPLATE[name[:-2]][1])
    lead = targets["toes_l"] - targets["calcn_l"]
    lead_native = cal.opensim_to_native_vector(lead)
    assert math.degrees(math.atan2(-lead_native[1], -lead_native[0])) > 15.0


def test_planted_feet_contracts() -> None:
    cap = _foot_capture({"L": 0.0, "R": 0.0})
    cut = _Capture(cap.time_s, cap.labels[:-1], cap.points_m[:, :-1], cap.valid[:, :-1])
    with pytest.raises(ValueError, match="foot markers"):
        tt.planted_feet_from_capture(cut)
    with pytest.raises(ValueError, match="'l' and 'r'"):
        tt.PlantedFeet({"l": np.zeros(3)}, {}, {}, 1)


# ------------------------------------------------------------- OpenSim tracking
def test_retarget_reproduces_a_synthetic_marker_turn() -> None:
    """Tracked pelvis and thorax turn equal a known synthetic marker turn."""
    pytest.importorskip("opensim")
    pytest.importorskip("scipy")
    from src.engines.physics_engines.opensim.python import msk_club_tracking as mt

    t = np.array([0.0, 0.02, 0.04, 0.06])
    pelvis, trunk = np.array([0.0, 5.0, 10.0, 15.0]), np.array([0.0, 12.0, 24.0, 35.0])
    lines = marker_turn_lines(_synthetic_markers(pelvis, trunk), t, _events(t))
    targets = tt.turn_targets_from_lines(lines, t)
    rows = np.load(SWING)["q"][[0, 0, 0, 0]]  # the source holds its address pose
    model = mc.MODELS_DIR / "golf_humanoid.osim"
    frames = mt.track_swing(model, rows, club="driver", turn_targets=targets)
    points = {
        k: np.array([f.turn_points[k] for f in frames]) for k in frames[0].turn_points
    }
    tracked = model_turn_lines(points, t, _events(t))
    np.testing.assert_allclose(tracked.pelvis.turn_deg, pelvis, atol=TURN_TOL_DEG)
    np.testing.assert_allclose(tracked.upper_trunk.turn_deg, trunk, atol=TURN_TOL_DEG)
    for frame in frames:
        assert frame.lead_grip_error_m <= GRIP_TOL_M
        assert frame.trail_grip_gap_m <= TRAIL_TOL_M


def test_track_swing_rejects_misaligned_turn_targets() -> None:
    from src.engines.physics_engines.opensim.python import msk_club_tracking as mt

    targets = tt.TurnTargets(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="turn_targets"):
        mt.track_swing(
            mc.MODELS_DIR / "golf_humanoid.osim",
            np.zeros((2, 44)),
            turn_targets=targets,
        )
