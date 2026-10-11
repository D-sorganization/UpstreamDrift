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


@pytest.mark.parametrize("sign_l", [-1.0, 1.0])
def test_seed_hip_rotation_is_sign_safe_for_mirrored_and_unmirrored_left_axes(
    sign_l: float,
) -> None:
    kin = _FakeKin(sign_l=sign_l)
    targets = _targets(18.0, 4.0)
    seeds = address_feet.seed_hip_rotation_deg(
        kin, np.zeros(3), targets, target_axis=np.array([0.0, -1.0, 0.0])
    )
    assert seeds["hip_rotation_r"] == pytest.approx(4.0, abs=0.5)
    assert seeds["hip_rotation_l"] == pytest.approx(sign_l * 18.0, abs=0.5)


def test_seed_document_feet_merges_without_mutating_or_adding_keys() -> None:
    kin = _FakeKin()
    document = {"address_seed_deg": {"LSInputY": 45.0}}
    out = address_feet.seed_document_feet(
        document,
        kin,
        _targets(18.0, 4.0),
        np.zeros(3),
        target_axis=np.array([0.0, -1.0, 0.0]),
    )
    assert document == {"address_seed_deg": {"LSInputY": 45.0}}
    assert out["address_seed_deg"]["LSInputY"] == 45.0
    assert out["address_seed_deg"]["hip_rotation_r"] == pytest.approx(4.0, abs=0.5)
    assert set(out) == set(document)  # the document schema is unchanged


def _spec_kin():
    pytest.importorskip("mujoco")
    import json

    from src.shared.python.motion_matching.pipeline.constants import REPO_ROOT
    from src.shared.python.motion_matching.pipeline.plant import get_plant

    path = (
        REPO_ROOT
        / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    plant = get_plant("mujoco", document)
    return document, plant.create_ik(dict(LEG_SEEDS))


def test_spec_left_hip_rotation_axis_is_mirrored_like_opensim() -> None:
    """+hip_rotation on either side turns that foot in, as in OpenSim (OSV-6 #11737).

    Before the fix +hip_rotation_l turned the left foot OUT, so equal-sign seeds
    splayed the feet and the left hip saturated at its -40 deg limit.
    """
    document, kin = _spec_kin()
    targets = _targets(0.0, 0.0)
    names = list(kin.coordinate_order)
    for side, expected_sign in (("right", -1.0), ("left", -1.0)):
        q = np.zeros(len(names))
        q[names.index(f"hip_rotation_{side[0]}")] = np.radians(10.0)
        angle = model_feet_deg(kin, q, targets, address_feet.MODEL_TARGET_AXIS)[side]
        assert np.sign(angle) == expected_sign
        assert abs(angle) == pytest.approx(10.0, abs=1.0)


def test_shared_seed_puts_both_model_feet_on_target_within_two_degrees() -> None:
    from src.shared.python.motion_matching.pipeline.lane import document_seed

    document, kin = _spec_kin()
    targets = _targets(16.4, 4.0)
    seeded = address_feet.seed_document_feet(
        document, kin, targets, document_seed(document, kin)
    )
    q = document_seed(seeded, kin)
    angles = model_feet_deg(kin, q, targets, address_feet.MODEL_TARGET_AXIS)
    assert angles["left"] == pytest.approx(16.4, abs=2.0)
    assert angles["right"] == pytest.approx(4.0, abs=2.0)
