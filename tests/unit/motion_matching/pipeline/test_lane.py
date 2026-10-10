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
