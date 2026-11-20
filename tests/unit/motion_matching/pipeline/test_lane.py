"""Unit tests for pipeline lane configuration and stance detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_law import GroundPlane

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


def test_stance_spheres_detects_grounded_spheres() -> None:
    from src.shared.python.motion_matching.pipeline.lane import stance_spheres

    labels = ("RAnkleOut", "LAnkleOut", "RToeIn", "RToeOut", "LToeIn", "LToeOut")
    # Frame 0 flat, frame 1 right ankle lifts by 0.05m (> tolerance 0.02m)
    points = np.zeros((2, len(labels), 3))
    points[1, labels.index("RAnkleOut"), 2] = 0.05
    valid = np.ones((2, len(labels)), dtype=bool)

    stance = stance_spheres(points, valid, labels)
    assert len(stance) == 2
    # Frame 0: both feet flat
    assert "heel_r" in stance[0]
    assert "forefoot_r" in stance[0]
    assert "toe_r" in stance[0]
    assert "heel_l" in stance[0]

    # Frame 1: right heel lifted
    assert "heel_r" not in stance[1]
    assert "heel_l" in stance[1]


def test_stance_spheres_validates_shapes() -> None:
    from src.shared.python.motion_matching.pipeline.lane import stance_spheres

    with pytest.raises(ValueError, match="3D array"):
        stance_spheres(np.zeros((2, 3)), np.ones((2, 3), dtype=bool), ("a", "b", "c"))

    with pytest.raises(ValueError, match="match points"):
        stance_spheres(
            np.zeros((2, 3, 3)), np.ones((2, 2), dtype=bool), ("a", "b", "c")
        )


def test_add_toe_spheres_appends_toe_contacts() -> None:
    from src.shared.python.motion_matching.pipeline.lane import add_toe_spheres

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    updated = add_toe_spheres(spec)

    names = {s["name"] for s in updated["contact"]["spheres"]}
    assert "toe_r" in names
    assert "toe_l" in names


def test_document_bounds_and_wrist_bounds() -> None:
    from src.shared.python.motion_matching.pipeline.lane import (
        document_bounds,
        wrist_bounds,
    )

    wb = wrist_bounds()
    assert "LWInputX" in wb
    assert "RWInputX" in wb
    assert wb["LWInputX"][0] < wb["LWInputX"][1]

    doc = {
        "coordinate_ranges_deg": {
            "SpineInputX": [-10.0, 10.0],
            "LWInputX": [-20.0, 20.0],  # in IK_UNBOUNDED, should be skipped
            "hip_flexion_r": [-30.0, 60.0],  # ends with _r, should be skipped
        }
    }
    db = document_bounds(doc)
    assert "SpineInputX" in db
    assert "LWInputX" not in db
    assert "hip_flexion_r" not in db
    assert db["SpineInputX"][0] < db["SpineInputX"][1]

    with pytest.raises(ValueError, match="ranges low < high"):
        document_bounds({"coordinate_ranges_deg": {"BadJoint": [10.0, -10.0]}})


def _synthetic_lane(rate_hz: float = 360.0):  # type: ignore[no-untyped-def]
    from src.shared.python.motion_matching.pipeline.lane import Lane
    from src.shared.python.motion_matching.tour_capture_contract import TourCapture

    labels = ("RAnkleOut", "LAnkleOut", "RToeIn", "RToeOut", "LToeIn", "LToeOut")
    frames = 10
    capture = TourCapture(
        time_s=np.arange(frames) / rate_hz,
        labels=labels,
        points_m=np.zeros((frames, len(labels), 3)),
        valid=np.ones((frames, len(labels)), dtype=bool),
    )
    return Lane(labels=labels, capture=capture)


def test_restart_policy_defaults_to_the_legacy_free_restarts() -> None:
    from src.shared.python.motion_matching.pipeline.constants import (
        TRAJECTORY_RESTARTS,
    )

    lane = _synthetic_lane()
    settings = lane.restart_settings()
    assert lane.restart_policy == "free"
    assert settings["restarts"] == TRAJECTORY_RESTARTS
    assert settings["restart_max_step_rad"] is None


def test_continuous_restart_policy_bounds_the_step_by_the_joint_speed() -> None:
    from src.shared.python.motion_matching.pipeline.constants import (
        RESTART_MAX_JOINT_SPEED_RAD_S,
    )

    lane = _synthetic_lane(rate_hz=360.0)
    lane.set_restart_policy("continuous")
    step = lane.restart_settings()["restart_max_step_rad"]
    assert step == pytest.approx(RESTART_MAX_JOINT_SPEED_RAD_S / 360.0)
    # Strided calibration subsets are not one sample apart: legacy restarts.
    assert lane.restart_settings(consecutive=False)["restart_max_step_rad"] is None
    report = lane.restart_report()
    assert report["policy"] == "continuous"
    assert report["max_step_rad"] == pytest.approx(step)
    assert report["max_joint_speed_rad_s"] == RESTART_MAX_JOINT_SPEED_RAD_S


def test_off_restart_policy_disables_restarts_and_unknown_is_rejected() -> None:
    lane = _synthetic_lane()
    lane.set_restart_policy("off")
    assert lane.restart_settings()["restarts"] == 0
    assert lane.restart_report()["max_step_rad"] is None
    with pytest.raises(ValueError, match="restart policy"):
        lane.set_restart_policy("jitter")


def test_cli_exposes_the_restart_policy() -> None:
    from src.shared.python.motion_matching.pipeline.cli import build_parser

    parser = build_parser()
    assert parser.parse_args([]).ik_restart_policy == "free"
    args = parser.parse_args(["--ik-restart-policy", "continuous"])
    assert args.ik_restart_policy == "continuous"
    with pytest.raises(SystemExit):
        parser.parse_args(["--ik-restart-policy", "jitter"])


@pytest.mark.parametrize("policy", ["free", "continuous", "off"])
def test_restart_report_validates_against_the_receipt_schema(policy: str) -> None:
    from pydantic import ValidationError

    from src.shared.python.motion_matching.pipeline.receipt_components import (
        IkRestartPolicyReceipt,
    )

    lane = _synthetic_lane()
    lane.set_restart_policy(policy)
    block = IkRestartPolicyReceipt.model_validate(lane.restart_report())
    assert block.policy == policy
    with pytest.raises(ValidationError):
        IkRestartPolicyReceipt.model_validate({**lane.restart_report(), "margin_m": -1})


def test_trajectory_convergence_settings_are_opt_in() -> None:
    """Iterations and the spin prior of the full-capture solve (#15296): the
    default leaves the solver's own cap and the document's prior untouched."""
    from src.shared.python.motion_matching.pipeline.cli import build_parser
    from src.shared.python.motion_matching.pipeline.constants import (
        SPIN_COORDINATES,
        SPIN_PRIOR,
    )

    lane = _synthetic_lane()
    lane.prior_weights = dict.fromkeys(SPIN_COORDINATES, SPIN_PRIOR)
    args = build_parser().parse_args([])
    assert (args.ik_iterations, args.ik_spin_prior_weight) == (None, None)
    lane.set_trajectory_convergence(args.ik_iterations, args.ik_spin_prior_weight)
    assert "iterations" not in lane.restart_settings()
    assert lane.convergence_report() == {}  # a default receipt is unchanged
    assert set(lane.prior_weights.values()) == {SPIN_PRIOR}
    args = build_parser().parse_args(
        ["--ik-iterations", "300", "--ik-spin-prior-weight", "0.001"]
    )
    lane.set_trajectory_convergence(args.ik_iterations, args.ik_spin_prior_weight)
    assert lane.restart_settings()["iterations"] == 300
    assert "iterations" not in lane.restart_settings(consecutive=False)
    assert lane.prior_weights == dict.fromkeys(SPIN_COORDINATES, 0.001)
    assert lane.convergence_report() == {"iterations": 300, "spin_prior_weight": 0.001}
    with pytest.raises(ValueError, match="iterations"):
        lane.set_trajectory_convergence(0, None)
    with pytest.raises(ValueError, match="spin prior"):
        lane.set_trajectory_convergence(None, -1.0)


def test_free_stance_slide_is_opt_in_and_keeps_the_feet_on_the_ground() -> None:
    """``--ik-free-stance-slide`` (#15296): the full-capture solve keeps the
    stance spheres on the ground but no longer anchors them where they first
    touched; calibration subsets and the default stay planted."""
    from src.shared.python.motion_matching.pipeline.address_feet import planted_stance
    from src.shared.python.motion_matching.pipeline.cli import build_parser

    lane = _synthetic_lane()
    assert build_parser().parse_args([]).ik_free_stance_slide is False
    assert lane.stance_settings() == planted_stance(lane)
    assert lane.convergence_report() == {}
    args = build_parser().parse_args(["--ik-free-stance-slide"])
    lane.stance_slide = args.ik_free_stance_slide
    free = lane.stance_settings()
    assert free["flat_feet_per_frame"] is lane.stance
    assert (free["plant_stance"], free["anchor_stance_at_init"]) == (False, False)
    assert lane.stance_settings(consecutive=False) == planted_stance(lane)
    assert lane.convergence_report() == {"free_stance_slide": True}


def test_soft_stance_anchor_is_an_opt_in_setting() -> None:
    """``--ik-stance-anchor-weight`` and ``--ik-stance-slide-mm`` (#15296): the
    full-capture solve keeps its plant but lets a planted sphere slide; the
    default, calibration subsets and the receipt of a default run are unchanged."""
    from src.shared.python.motion_matching.pipeline.address_feet import planted_stance
    from src.shared.python.motion_matching.pipeline.cli import build_parser
    from src.shared.python.motion_matching.pipeline.lane import apply_stance_arguments

    lane = _synthetic_lane()
    args = build_parser().parse_args([])
    assert (args.ik_stance_anchor_weight, args.ik_stance_slide_mm) == (None, None)
    apply_stance_arguments(lane, args)
    assert lane.stance_settings() == planted_stance(lane)
    assert lane.convergence_report() == {}
    args = build_parser().parse_args(
        ["--ik-stance-anchor-weight", "10", "--ik-stance-slide-mm", "5"]
    )
    apply_stance_arguments(lane, args)
    soft = lane.stance_settings()
    assert soft == planted_stance(lane) | {
        "anchor_weight": 10.0,
        "anchor_slide_m": pytest.approx(0.005),
    }
    assert lane.stance_settings(consecutive=False) == planted_stance(lane)
    assert lane.convergence_report() == {
        "stance_anchor_weight": 10.0,
        "stance_slide_mm": 5.0,
    }
    only_slide = _synthetic_lane()
    only_slide.set_stance_anchor(None, 8.0)
    assert only_slide.stance_settings() == planted_stance(only_slide) | {
        "anchor_slide_m": pytest.approx(0.008)
    }
    for weight, slide in ((-1.0, None), (None, -2.0), (float("nan"), None)):
        with pytest.raises(ValueError, match="stance anchor"):
            lane.set_stance_anchor(weight, slide)
    both = build_parser().parse_args(
        ["--ik-free-stance-slide", "--ik-stance-slide-mm", "5"]
    )
    slid = _synthetic_lane()
    apply_stance_arguments(slid, both)  # free first solve, soft re-solve
    assert slid.stance_settings()["plant_stance"] is False
    assert "anchor_slide_m" not in slid.stance_settings()
    assert slid.stance_anchor == {"anchor_slide_m": pytest.approx(0.005)}


def test_address_pose_prior_reaches_only_the_first_frame_of_the_swing_solve() -> None:
    """The bundle stage's address pose targets (#15296) enter the full-capture
    solve on its first frame; without them, and in a calibration subset,
    nothing is passed."""
    lane = _synthetic_lane()
    assert lane.address_pose_prior == {}
    assert lane.address_pose_frames(None) is None
    lane.address_pose_prior = {"LEInput": (-0.09, 0.013)}
    per_frame = lane.address_pose_frames(None)
    assert per_frame is not None and len(per_frame) == lane.frames
    assert per_frame[0] == {"LEInput": (-0.09, 0.013)}
    assert set(per_frame[1:]) == {None}
    assert lane.address_pose_frames([0, 5]) is None


def test_girdle_range_is_an_opt_in_tightening() -> None:
    """``--ik-girdle-range-deg`` (#15296): the four shoulder-girdle coordinates
    are bounded about neutral in every solve that reads the lane's bounds; the
    default adds no bound, and a range wider than the document's is refused."""
    from src.shared.python.motion_matching.pipeline.cli import build_parser
    from src.shared.python.motion_matching.pipeline.lane import GIRDLE_COORDINATES

    lane = _synthetic_lane()
    before = dict(lane.bounds)
    assert build_parser().parse_args([]).ik_girdle_range_deg is None
    lane.set_girdle_range(None)
    assert lane.bounds == before
    assert lane.convergence_report() == {}
    args = build_parser().parse_args(["--ik-girdle-range-deg", "12"])
    lane.set_girdle_range(args.ik_girdle_range_deg)
    assert set(GIRDLE_COORDINATES) == {
        "LScapInputX",
        "LScapInputY",
        "RScapInputX",
        "RScapInputY",
    }
    for name in GIRDLE_COORDINATES:
        assert lane.bounds[name] == pytest.approx((-np.radians(12), np.radians(12)))
    assert {k: v for k, v in lane.bounds.items() if k in before} == before
    assert lane.convergence_report() == {"girdle_range_deg": 12.0}
    for bad in (0.0, -3.0, 20.5, float("nan")):
        with pytest.raises(ValueError, match="girdle range"):
            lane.set_girdle_range(bad)
