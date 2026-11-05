"""Face convention guard at the A2 neutral address (Part of #14561, D44/D45).

The showpiece driver spec with its committed neutral-grip address frame must
read a near-square face against the default ``TargetFrame``.  A reference swing
fitted on a different grip can read far from square on this spec (about -90 deg
for the capture-grip bundle); that is a grip issue, not a frame one, and this
test keeps the two apart.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.impact_parameters import TargetFrame
from src.shared.python.impact_parameters.adapters import club_face_spec_from_spec
from src.shared.python.motion_matching import club_face_target as cft

mujoco = pytest.importorskip("mujoco")
pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
SPEC = MODELS / "full_body_spec_anthro_driver.json"
ADDRESS = MODELS / "evidence/grip_roll/address_frame_neutral.npz"
SQUARE_TOL_DEG = 10.0


def _heading_deg(normal: np.ndarray, frame: TargetFrame) -> float:
    """Face heading with GCV-15 signs (+ = open) in the target frame."""
    return math.degrees(math.atan2(-(normal @ frame.y_t), normal @ frame.x_t))


@pytest.fixture(scope="module")
def posed():
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    raw = SPEC.read_bytes()
    xml, meta = export_full_body_mjcf(raw, visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    q0 = np.load(ADDRESS)["q"][0]
    for value, name in zip(q0, meta["coordinate_order"], strict=True):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, meta["frame_sites"]["Clubhead"]
    )
    return json.loads(raw), model, data, site


def test_neutral_address_reads_square_with_both_face_sources(posed) -> None:
    spec, model, data, site = posed
    frame = TargetFrame()
    rot = data.site_xmat[site].reshape(3, 3)
    via_target = rot @ cft.face_axis_in_frame(spec)
    body = int(model.site_bodyid[site])
    via_spec = data.xmat[body].reshape(3, 3) @ np.array(
        club_face_spec_from_spec(spec).normal_body
    )
    assert via_spec == pytest.approx(via_target, abs=1e-9)
    assert abs(_heading_deg(via_target, frame)) <= SQUARE_TOL_DEG
    loft_deg = math.degrees(math.asin(via_target[2]))
    assert 8.0 <= loft_deg <= 20.0  # a driver face looks up as well as at the target


def test_guard_detects_a_rolled_face(posed) -> None:
    """Control: a 60 deg roll about the shaft leaves the +-10 deg band.

    The shaft is the club-body grip axis (``ClubFaceSpec.grip_body``); the
    full-body ``Clubhead`` site is a face frame whose y is not the shaft (#14674).
    """
    spec, model, data, site = posed
    body = int(model.site_bodyid[site])
    face = club_face_spec_from_spec(spec)
    shaft = data.xmat[body].reshape(3, 3) @ np.array(face.grip_body)
    normal = data.xmat[body].reshape(3, 3) @ np.array(face.normal_body)
    assert abs(float(shaft @ normal)) < 1e-9  # the roll axis is in the face plane
    assert shaft[2] > 0.5  # toward the grip: up and away from the head
    a = math.radians(60.0)
    rolled = (
        normal * math.cos(a)
        + np.cross(shaft, normal) * math.sin(a)
        + shaft * (shaft @ normal) * (1 - math.cos(a))
    )
    assert abs(_heading_deg(normal, TargetFrame())) <= SQUARE_TOL_DEG
    assert abs(_heading_deg(rolled, TargetFrame())) > SQUARE_TOL_DEG
