"""Address foot-progression residual and marker-attachment symmetry (OSV-4, #11730)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline import address_feet
from src.shared.python.motion_matching.pipeline.address_feet import (
    FootTargets,
    build_foot_targets,
    model_feet_deg,
    refine_foot_progression,
)
from src.shared.python.motion_matching.pipeline.constants import (
    LEG_SEEDS,
    square_forefoot_seeds,
)

pytestmark = [pytest.mark.unit]


class _FakeKin:
    """Two-foot kinematics whose foot yaw is pelvis yaw plus hip rotation.

    Right-handed capture world: golfer faces -x, target -y, left (lead) foot at
    -y. ``body_poses`` returns calcn and toes origins on the ground.
    """

    coordinate_order = ("pelvis_yaw", "hip_rotation_r", "hip_rotation_l")

    def __init__(self, sign_r: float = 1.0, sign_l: float = -1.0) -> None:
        self.sign = {"r": sign_r, "l": sign_l}

    def _yaw(self, q: np.ndarray, side: str) -> float:
        i = self.coordinate_order.index(f"hip_rotation_{side}")
        return q[0] + self.sign[side] * q[i]

    def body_poses(self, q, bodies):  # noqa: ANN001 - test double
        out = {}
        for side in "rl":
            # out direction: lead (left) toward -y, trail (right) toward +y
            out_dir = np.array([0.0, -1.0, 0.0] if side == "l" else [0.0, 1.0, 0.0])
            fwd = np.array([-1.0, 0.0, 0.0])
            yaw = self._yaw(q, side)
            axis = np.cos(yaw) * fwd + np.sin(yaw) * out_dir
            base = np.array([0.0, -0.25 if side == "l" else 0.25, 0.05])
            out[f"calcn_{side}"] = (np.eye(3), base)
            out[f"toes_{side}"] = (np.eye(3), base + 0.16 * axis)
        return {b: out[b] for b in bodies}


def _targets(left: float, right: float, **kw) -> FootTargets:
    return FootTargets(
        target_deg={"left": left, "right": right},
        is_default={"left": False, "right": False},
        source="capture",
        **kw,
    )


def test_model_feet_deg_reads_signed_toe_out() -> None:
    kin = _FakeKin()
    q = np.array([0.0, np.radians(10.0), np.radians(-5.0)])
    angles = model_feet_deg(kin, q, _targets(0.0, 0.0))
    assert angles["right"] == pytest.approx(10.0, abs=1e-9)
    assert angles["left"] == pytest.approx(5.0, abs=1e-9)  # sign_l = -1


def test_refine_reaches_targets_within_tolerance_for_either_sign_convention() -> None:
    for sign_l in (-1.0, 1.0):
        kin = _FakeKin(sign_l=sign_l)
        start = SimpleNamespace(q=np.array([np.radians(30.0), 0.0, 0.0]))

        def resolve(q0, prior):  # noqa: ANN001 - prior pins hip rotation: keep q0
            assert prior["hip_rotation_r"] > 0 and prior["hip_rotation_l"] > 0
            return SimpleNamespace(q=q0.copy(), marker_rms_m=0.01)

        fit = refine_foot_progression(kin, start, _targets(18.0, 4.0), resolve)
        angles = model_feet_deg(kin, fit.q, _targets(18.0, 4.0))
        assert angles["left"] == pytest.approx(18.0, abs=0.5)
        assert angles["right"] == pytest.approx(4.0, abs=0.5)


def test_refine_is_a_noop_when_already_on_target() -> None:
    kin = _FakeKin()
    q = np.array([0.0, np.radians(4.0), np.radians(-18.0)])
    start = SimpleNamespace(q=q)
    calls: list[int] = []

    def resolve(q0, prior):  # noqa: ANN001
        calls.append(1)
        return SimpleNamespace(q=q0)

    fit = refine_foot_progression(kin, start, _targets(18.0, 4.0), resolve)
    assert fit is start
    assert not calls


def test_refine_rejects_missing_hip_rotation_coordinates() -> None:
    kin = _FakeKin()
    kin.coordinate_order = ("pelvis_yaw", "x", "y")
    with pytest.raises(ValueError, match="hip_rotation"):
        refine_foot_progression(
            kin, SimpleNamespace(q=np.zeros(3)), _targets(20.0, 20.0), lambda a, b: a
        )


def test_foot_targets_validates() -> None:
    with pytest.raises(ValueError, match="left"):
        FootTargets(target_deg={"right": 1.0}, is_default={}, source="capture")
    with pytest.raises(ValueError, match="weight"):
        _targets(1.0, 1.0, weight=-1.0)


def test_build_foot_targets_modes() -> None:
    lane = SimpleNamespace(points=None, valid=None, labels=())
    assert build_foot_targets(lane, "off") is None
    forced = build_foot_targets(lane, "default")
    assert forced is not None
    assert forced.target_deg == {"left": 20.0, "right": 20.0}
    assert forced.is_default == {"left": True, "right": True}
    with pytest.raises(ValueError, match="foot_progression"):
        build_foot_targets(lane, "banana")


def test_build_foot_targets_capture_falls_back_to_flagged_default_without_wrists() -> (
    None
):
    pts = np.zeros((10, 6, 3))
    lane = SimpleNamespace(
        points=pts,
        valid=np.ones((10, 6), bool),
        labels=("LAnkleOut", "LToeIn", "LToeOut", "RAnkleOut", "RToeIn", "RToeOut"),
    )
    targets = build_foot_targets(lane, "capture")
    assert targets is not None
    assert targets.is_default == {"left": True, "right": True}
    assert targets.target_deg["left"] == 20.0
    assert "wrist" in targets.notes


def test_marker_attachments_left_mirrors_right_in_the_foot_frame() -> None:
    for marker in ("AnkleOut", "ToeIn", "ToeOut"):
        body_r, off_r = LEG_SEEDS[f"R{marker}"]
        body_l, off_l = LEG_SEEDS[f"L{marker}"]
        assert body_r.endswith("_r") and body_l.endswith("_l")
        assert off_l[0] == pytest.approx(off_r[0], abs=0.0105)  # x forward
        assert off_l[2] == pytest.approx(-off_r[2], abs=1e-9)  # z lateral mirrors
    # toe markers are exactly symmetric
    for marker in ("ToeIn", "ToeOut"):
        _, off_r = LEG_SEEDS[f"R{marker}"]
        _, off_l = LEG_SEEDS[f"L{marker}"]
        assert off_l == (off_r[0], off_r[1], -off_r[2])


def test_square_forefoot_removes_the_toe_marker_stagger() -> None:
    # the stock seeds stagger ToeIn 0.02 m ahead of ToeOut: a 12.5 deg yaw bias
    for side in "RL":
        stock_in, stock_out = (
            LEG_SEEDS[f"{side}ToeIn"][1],
            LEG_SEEDS[f"{side}ToeOut"][1],
        )
        assert abs(stock_in[0] - stock_out[0]) > 0.015
    squared = square_forefoot_seeds(LEG_SEEDS)
    for side in "RL":
        t_in, t_out = squared[f"{side}ToeIn"][1], squared[f"{side}ToeOut"][1]
        assert t_in[0] == pytest.approx(t_out[0], abs=1e-12)
        assert t_in[0] == pytest.approx(
            0.5 * (LEG_SEEDS[f"{side}ToeIn"][1][0] + LEG_SEEDS[f"{side}ToeOut"][1][0])
        )
        assert t_in[2] == LEG_SEEDS[f"{side}ToeIn"][1][2]  # lateral untouched
    # non-toe markers are untouched and the input mapping is not mutated
    assert squared["RKneeOut"] == LEG_SEEDS["RKneeOut"]
    assert LEG_SEEDS["RToeIn"][1][0] == 0.19


def test_module_exports() -> None:
    assert hasattr(address_feet, "FootTargets")


def test_cli_foot_progression_option_defaults_to_off() -> None:
    from src.shared.python.motion_matching.pipeline.cli import build_parser

    parser = build_parser()
    assert parser.parse_args([]).foot_progression == "off"
    assert parser.parse_args(["--foot-progression", "capture"]).foot_progression == (
        "capture"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--foot-progression", "banana"])


def test_lane_leg_seeds_only_square_the_forefoot_when_enabled() -> None:
    from src.shared.python.motion_matching.pipeline.lane import Lane

    lane = Lane.__new__(Lane)
    lane.feet = None
    assert lane.leg_seeds() == dict(LEG_SEEDS)
    lane.feet = _targets(20.0, 20.0)
    squared = lane.leg_seeds()
    assert squared["RToeIn"][1][0] == squared["RToeOut"][1][0]
    assert squared["RAnkleOut"] == LEG_SEEDS["RAnkleOut"]
