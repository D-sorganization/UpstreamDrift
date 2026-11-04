"""SHOW-A4 (#14553): cited neck range of motion, enforced in the MJCF."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np
import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml - test parses our own output
from pathlib import Path

import pytest

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
    export_full_body_mjcf,
)
from src.shared.python.motion_matching import range_of_motion
from src.shared.python.motion_matching.gaze import NECK_COORDINATES
from src.shared.python.motion_matching.neck_rom import NECK_ROM, neck_range_rad

pytestmark = pytest.mark.unit

SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)


def test_every_neck_axis_has_a_cited_range() -> None:
    assert set(NECK_ROM) == set(NECK_COORDINATES)
    for rom in NECK_ROM.values():
        assert rom.lower_deg < 0 < rom.upper_deg
        assert rom.sources and all(len(s) > 20 for s in rom.sources)
        assert rom.motion


def test_range_of_motion_table_reads_the_cited_neck_ranges() -> None:
    for name, rom in NECK_ROM.items():
        assert range_of_motion.HUMAN_RANGES_DEG[name] == (
            rom.lower_deg,
            rom.upper_deg,
        )


def test_spec_coordinate_ranges_match_cited_table() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for name, rom in NECK_ROM.items():
        assert tuple(spec["coordinate_ranges_deg"][name]) == (
            rom.lower_deg,
            rom.upper_deg,
        )


def test_neck_range_rad_is_a_precondition_checked_lookup() -> None:
    lo, hi = neck_range_rad("NeckInputZ")
    assert hi == pytest.approx(math.radians(NECK_ROM["NeckInputZ"].upper_deg))
    with pytest.raises(ValueError):
        neck_range_rad("LEInput")


def test_neck_joints_are_limited_in_the_mjcf() -> None:
    xml, _ = export_full_body_mjcf(SPEC.read_bytes())
    joints = {j.get("name"): j for j in ET.fromstring(xml).iter("joint")}  # nosec B314
    for name in NECK_COORDINATES:
        joint = joints[name]
        assert joint.get("limited") == "true"
        lo, hi = (float(v) for v in joint.get("range", "").split())
        assert (lo, hi) == pytest.approx(neck_range_rad(name))
    # Only the neck is newly limited by A4 (other joints are untouched).
    others = [j for n, j in joints.items() if n not in NECK_COORDINATES]
    assert all(j.get("limited") == "false" for j in others)


def test_neck_joint_tree_contract() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    neck = [
        j
        for j in spec["joints"]
        if [p["coordinate"] for p in j["primitives"]] == list(NECK_COORDINATES)
    ]
    assert len(neck) == 1
    assert [p["primitive"] for p in neck[0]["primitives"]] == ["Rx", "Ry", "Rz"]
    assert "Head" in neck[0]["child"]
    order = spec["coordinate_order"]
    idx = [order.index(c) for c in NECK_COORDINATES]
    assert idx == sorted(idx)


def _neck_model(limited: bool) -> mujoco.MjModel:
    xml, _ = export_full_body_mjcf(SPEC.read_bytes())
    model = mujoco.MjModel.from_xml_string(xml)
    if not limited:
        model.jnt_limited[:] = 0
    return model


def _settled_axial_deg(model: mujoco.MjModel, start_deg: float) -> float:
    data = mujoco.MjData(model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "NeckInputZ")
    adr = model.jnt_qposadr[jid]
    data.qpos[adr] = math.radians(start_deg)
    for _ in range(300):
        mujoco.mj_step(model, data)
    return math.degrees(float(data.qpos[adr]))


def test_mujoco_dynamics_enforce_the_neck_limit() -> None:
    """mj_forward ignores limits; mj_step must pull an over-range neck back."""
    limit = NECK_ROM["NeckInputZ"].upper_deg
    limited = _settled_axial_deg(_neck_model(True), limit + 25.0)
    free = _settled_axial_deg(_neck_model(False), limit + 25.0)
    assert free > limit + 20.0  # control: with no limit the neck stays put
    assert limited < limit + 5.0  # soft constraint: pushed back to the limit


def test_matched_driver_swing_neck_excursion_is_recorded() -> None:
    root = SPEC.parents[3]
    run = root / "docs/development/full_body_models/evidence/ground_support"
    run = run / "anthro_driver_drake"
    order = json.loads(
        (run / "full_body_spec_hipcal_scaled.json").read_text(encoding="utf-8")
    )["coordinate_order"]
    q = np.load(run / "dynamics_record.npz")["q_track"]
    recorded = json.loads(
        (
            root
            / "docs/development/full_body_models/evidence/neck_rom"
            / "driver_swing_neck_excursion.json"
        ).read_text(encoding="utf-8")
    )
    for name, rom in NECK_ROM.items():
        deg = np.degrees(q[:, order.index(name)])
        row = recorded[name]
        assert row["min_deg"] == pytest.approx(float(deg.min()), abs=0.05)
        assert row["max_deg"] == pytest.approx(float(deg.max()), abs=0.05)
        assert row["limit_deg"] == [rom.lower_deg, rom.upper_deg]
        # The fit sits at (not past) the axial limit: it is the binding joint.
        assert deg.min() >= rom.lower_deg - 0.5 and deg.max() <= rom.upper_deg + 0.5
