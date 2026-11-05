"""Drake address solve rests the club sole on the ground (D20, #12969).

``DrakeFullBodyIK`` inherited the base no-op ``_append_club_sole``, so the
address solve ignored ``club_sole_weight`` and left the sole 126.7 mm above the
ground, tripping ``assert_club_sole_at_ground``.
"""

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

pytestmark = [pytest.mark.unit, pytest.mark.requires_drake]

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / "docs/development/full_body_models/evidence/ground_support"
FIXTURES = ROOT / "tests/fixtures/club_face"


def _drake_kin():
    try:
        import pydrake.all as drake_all
    except ImportError as exc:
        pytest.skip(f"pydrake not importable: {exc}")
    if not hasattr(drake_all, "MultibodyPlant"):
        pytest.skip("real pydrake runtime required")
    from src.engines.physics_engines.drake.python.full_body_ik import DrakeFullBodyIK

    folder = EVIDENCE / "anthro_driver_drake"
    spec = json.loads((folder / "full_body_spec_hipcal_scaled.json").read_text("utf-8"))
    receipt = json.loads((folder / "receipt.json").read_text("utf-8"))
    attachments = {
        k: (v["body"], tuple(v["offset_m"]))
        for k, v in receipt["ik"]["attachments_m"].items()
    }
    ground = GroundPlane(
        normal=(0.0, 0.0, 1.0),
        height_m=float(spec["contact"]["ground"].get("height_m") or 0.0),
    )
    swing = load_coordinate_swing(
        FIXTURES / "swing_q_driver.npz",
        FIXTURES / "address_poses.json",
        "driver",
        spec["coordinate_order"],
    )
    return DrakeFullBodyIK(spec, attachments=attachments), spec, ground, swing.q[0]


def test_drake_club_sole_weight_pulls_the_sole_to_the_ground() -> None:
    kin, spec, ground, q0 = _drake_kin()
    targets = kin.marker_positions(q0)
    valid = np.ones(len(targets), dtype=bool)
    solve = {"ground": ground, "iterations": 80}
    without = kin.solve_pose(targets, valid, q0, club_sole_weight=0.0, **solve)
    weighted = kin.solve_pose(
        targets, valid, q0, club_sole_weight=ADDRESS_CLUB_SOLE_WEIGHT, **solve
    )
    h_without = club_sole_height_m(kin, without.q, spec, ground)
    h_with = club_sole_height_m(kin, weighted.q, spec, ground)
    assert h_without is not None and h_with is not None
    assert h_with < h_without
    assert h_with <= CLUB_SOLE_GROUND_TOLERANCE_M
