"""Club sole rests on the ground at the fitted address pose (#12969)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import load_coordinate_swing
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.pipeline.address_club import club_sole_height_m
from src.shared.python.motion_matching.pipeline.constants import (
    ADDRESS_CLUB_SOLE_WEIGHT,
    CLUB_SOLE_GROUND_TOLERANCE_M,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

ROOT = Path(__file__).resolve().parents[4]
MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
EVIDENCE = MODELS / "evidence/ground_support"


def _anthro_kin(club: str):
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_ik import (
        FullBodyMarkerKinematics,
    )
    from src.engines.physics_engines.mujoco.python.overlay_source import (
        MujocoOverlaySource,
    )

    folder = "anthro_driver" if club == "driver" else "anthro_iron"
    receipt = json.loads((EVIDENCE / folder / "receipt.json").read_text())
    spec_path = EVIDENCE / folder / "full_body_spec_hipcal_scaled.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    attachments = {
        label: (entry["body"], tuple(entry["offset_m"]))
        for label, entry in receipt["ik"]["attachments_m"].items()
    }
    adapter = MujocoOverlaySource(spec_path.read_bytes())
    kin = FullBodyMarkerKinematics(adapter, attachments)
    ground_h = float(spec["contact"]["ground"].get("height_m") or 0.0)
    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=ground_h)
    swing = load_coordinate_swing(
        FIXTURES / f"swing_q_{club}.npz",
        FIXTURES / "address_poses.json",
        club,
        spec["coordinate_order"],
    )
    return kin, spec, ground, swing.q[0]


@pytest.mark.parametrize("club", ["driver", "iron7"])
def test_club_sole_weight_pulls_the_sole_to_the_ground(club: str) -> None:
    kin, spec, ground, q0 = _anthro_kin(club)
    kin._set(q0)
    targets = kin.marker_positions(q0)
    valid = np.ones(len(targets), dtype=bool)
    without = kin.solve_pose(
        targets,
        valid,
        q0,
        ground=ground,
        iterations=80,
        club_sole_weight=0.0,
    )
    with_weight = kin.solve_pose(
        targets,
        valid,
        q0,
        ground=ground,
        iterations=80,
        club_sole_weight=ADDRESS_CLUB_SOLE_WEIGHT,
    )
    hover_without = club_sole_height_m(kin, without.q, spec, ground)
    hover_with = club_sole_height_m(kin, with_weight.q, spec, ground)
    assert hover_without is not None and hover_with is not None
    assert hover_with < hover_without
    assert hover_with <= CLUB_SOLE_GROUND_TOLERANCE_M
