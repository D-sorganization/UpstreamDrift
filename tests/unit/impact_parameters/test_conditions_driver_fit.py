"""C2 (#14558) acceptance on the committed best-fit driver rollout.

Reference: exact directional-derivative velocity and hand-written GCV-15
definitions, on the full-body tree, at the ball-passage detector's impact.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.impact_parameters import TargetFrame
from src.shared.python.impact_parameters.adapters import club_face_spec_from_spec
from src.shared.python.impact_parameters.conditions import Rollout, evaluate_rollout

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.unit

SEEDS = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)
MPS_TO_MPH = 2.2369362920544
FRAME = TargetFrame()  # ADR-0041: Z-up, target along -Y, right-handed


@pytest.fixture(scope="module")
def fit():
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    raw = (SEEDS / "full_body_spec_hipcal_scaled.json").read_bytes()
    spec = json.loads(raw)
    xml, meta = export_full_body_mjcf(raw)
    model = mujoco.MjModel.from_xml_string(xml)
    rec = np.load(SEEDS / "dynamics_record.npz")
    joints = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        for n in meta["coordinate_order"]
    ]
    t = rec["time_s"]
    q = np.zeros((len(t), model.nq))
    v = np.zeros((len(t), model.nv))
    q[:, model.jnt_qposadr[joints]] = rec["q"]
    v[:, model.jnt_dofadr[joints]] = rec["v"]
    site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, meta["frame_sites"]["Clubhead"]
    )
    body = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[site])
    )
    return model, spec, site, body, t, q, v


def _reference(model, site, t, q, v, spec):
    """Face-centre series from the ``Clubhead`` site and ``club_face_target``;
    velocity is the exact directional derivative (h = 1e-7), not the Jacobian."""
    from src.shared.python.motion_matching import club_face_target as cft

    centre = cft.face_centre_in_frame(spec)
    axis = cft.face_axis_in_frame(spec)
    data = mujoco.MjData(model)

    def face(qrow):
        data.qpos[:] = qrow
        mujoco.mj_kinematics(model, data)
        rot = data.site_xmat[site].reshape(3, 3)
        return data.site_xpos[site] + rot @ centre, rot @ axis

    pos = np.array([face(row)[0] for row in q])
    h = 1e-7
    vel = np.array(
        [
            (face(q[i] + h * v[i])[0] - face(q[i] - h * v[i])[0]) / (2 * h)
            for i in range(len(t))
        ]
    )
    normal = np.array([face(row)[1] for row in q])
    return pos, normal, vel


def _hand_conditions(vel, normal):
    """GCV-15 definitions written out directly (RH, target -Y, up +Z)."""
    x_t, up = FRAME.x_t, FRAME.z_t
    y_t = np.cross(up, x_t)
    vh = math.hypot(vel @ x_t, vel @ y_t)

    def heading(w):
        return math.degrees(math.atan2(-(w @ y_t), w @ x_t))

    return {
        "speed_mph": float(np.linalg.norm(vel)) * MPS_TO_MPH,
        "aoa_deg": math.degrees(math.atan2(vel @ up, vh)),
        "path_deg": heading(vel),
        "face_deg": heading(normal),
    }


def test_matches_independent_reference_on_best_fit_driver(fit):
    model, spec, site, body, t, q, v = fit
    face = club_face_spec_from_spec(spec)
    pos, normal_all, vel = _reference(model, site, t, q, v, spec)
    detector = json.loads((SEEDS / "replay_post_impact_capture_a.json").read_text())
    idx = int(np.searchsorted(t, detector["impact_time_s"]))
    normal = normal_all[idx]
    data = mujoco.MjData(model)
    data.qpos[:] = q[idx]
    mujoco.mj_kinematics(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    rot = data.xmat[bid].reshape(3, 3)
    toe, high = rot @ face.axes()[1], rot @ np.cross(face.axes()[1], face.axes()[0])
    # synthetic round trip (not a fit measurement): 8 mm toe, 5 mm crown
    contact = pos[idx] + 0.008 * toe + 0.005 * high

    start = time.perf_counter()
    got = evaluate_rollout(
        model,
        Rollout(t, q, v),
        body,
        FRAME,
        spec=face,
        impact_index=idx,
        ball_contact_m=contact,
    )
    elapsed = time.perf_counter() - start

    want = _hand_conditions(vel[idx], normal)
    # frame check without the face spec: level motion down -Y at driver speed
    v_h = np.hypot(vel[idx][0], vel[idx][1])
    assert math.degrees(math.atan2(abs(vel[idx][0]), -vel[idx][1])) < 30.0
    assert vel[idx][1] < 0.0 and abs(vel[idx][2]) < 0.2 * v_h
    assert 90.0 < got.speed_mph < 130.0
    assert got.speed_mph == pytest.approx(want["speed_mph"], rel=1e-4)
    assert got.aoa_deg == pytest.approx(want["aoa_deg"], abs=0.01)
    assert got.path_deg == pytest.approx(want["path_deg"], abs=0.01)
    assert got.face_deg == pytest.approx(want["face_deg"], abs=0.01)
    assert got.strike_offset_x_m == pytest.approx(0.008, abs=1e-6)
    assert got.strike_offset_y_m == pytest.approx(0.005, abs=1e-6)
    assert elapsed < 1.0


def test_face_spec_rejects_unknown_frame(fit):
    _, spec, *_ = fit
    with pytest.raises(ValueError, match="no frame named"):
        club_face_spec_from_spec(spec, "NoSuchFrame")
