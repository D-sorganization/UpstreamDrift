"""Derive the sign of the NeckInputY -> MyoSuite neck_flexion retarget by FK.

Issue #11729 (OSV-3).  The anthro document neck joint is
``Rx(NeckInputX) Ry(NeckInputY) Rz(NeckInputZ)`` in the COMRod frame with the
head forward axis (towards ``HeadFront``) on +x, so ``NeckInputY`` pitches the
head.  The pinned myo_sim head chain has ``neck_flexion`` (hinge about head z)
and ``neck_rotation`` only.  This script perturbs each coordinate by +/-delta
from the neutral pose in MuJoCo, measures the change of head forward-axis
pitch (angle above the horizontal, positive = looking up) and reports the
sign that makes ``sign * NeckInputY`` produce the same pitch change in
MyoSuite as ``NeckInputY`` in the native model.

Run headless::

    MUJOCO_GL=egl python3 -m scripts.myosuite_neck_flexion_sign \
        --myo-sim shared/models/myosuite/myo_sim
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.shared.python.motion_matching.pipeline.constants import REPO_ROOT
from src.shared.python.motion_matching.pipeline.plant import get_plant

logger = logging.getLogger(__name__)

SPEC = REPO_ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
OUT = REPO_ROOT / "docs/development/full_body_models/evidence/myosuite_neck"
DELTA_RAD = 0.2
# Two head-fixed points 0.1 m apart along the head forward (+x, HeadFront) axis,
# at HeadFront height (spec marker HeadFront = Head + [0.1, 0, 0.16]).
HEAD_AXIS_POINTS = {
    "back": ("Head", (0.0, 0.0, 0.16)),
    "front": ("Head", (0.1, 0.0, 0.16)),
}


def _pitch_deg(forward: np.ndarray, up: np.ndarray) -> float:
    """Angle of ``forward`` above the plane normal to ``up`` (degrees)."""
    f = forward / np.linalg.norm(forward)
    return float(np.degrees(np.arcsin(np.clip(f @ up, -1.0, 1.0))))


def native_pitch(coordinate: str, value: float) -> float:
    """Head forward-axis pitch in the native MuJoCo plant, neutral elsewhere."""
    document = json.loads(SPEC.read_text(encoding="utf-8"))
    plant = get_plant("mujoco", document)
    q = np.zeros(len(plant.coordinate_order))
    q[plant.coordinate_order.index(coordinate)] = value
    pts = plant.marker_positions(q, HEAD_AXIS_POINTS)
    return _pitch_deg(pts[1] - pts[0], np.array([0.0, 0.0, 1.0]))


def myosuite_pitch(model_xml: Path, joint: str, value: float) -> float:
    """Head forward-axis (body +x, OpenSim anterior) pitch in MyoSuite."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[model.jnt_qposadr[model.joint(joint).id]] = value
    mujoco.mj_kinematics(model, data)
    up = -np.asarray(model.opt.gravity) / np.linalg.norm(model.opt.gravity)
    forward = data.body("head").xmat.reshape(3, 3)[:, 0]
    return _pitch_deg(forward, up)


def myosuite_head_forward_alignment(model_xml: Path) -> float:
    """Cosine between head +x and the right foot (calcn -> toes) at neutral."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    foot = data.body("toes_r").xpos - data.body("calcn_r").xpos
    head_x = data.body("head").xmat.reshape(3, 3)[:, 0]
    return float(head_x @ foot / np.linalg.norm(foot))


def derive(myo_sim: Path) -> dict[str, object]:
    """Pitch sensitivities and the resulting retarget sign."""
    xml = myo_sim / "body" / "myobody_simpleupper.xml"
    native = {
        c: (native_pitch(c, DELTA_RAD) - native_pitch(c, -DELTA_RAD)) / 2.0
        for c in ("NeckInputX", "NeckInputY", "NeckInputZ")
    }
    myo = {
        j: (myosuite_pitch(xml, j, DELTA_RAD) - myosuite_pitch(xml, j, -DELTA_RAD))
        / 2.0
        for j in ("neck_flexion", "neck_rotation")
    }
    sign = float(np.sign(native["NeckInputY"] * myo["neck_flexion"]))
    if sign == 0.0:
        raise ValueError("Degenerate pitch sensitivity; sign undetermined")
    if myosuite_head_forward_alignment(xml) < 0.9:
        raise ValueError("MyoSuite head +x is not anterior; pitch is ill-defined")
    return {
        "issue": 11729,
        "method": "central difference of head forward-axis pitch, +/-delta rad",
        "delta_rad": DELTA_RAD,
        "native_spec": SPEC.relative_to(REPO_ROOT).as_posix(),
        "myosuite_model": "myo_sim body/myobody_simpleupper.xml",
        "native_pitch_change_deg": native,
        "myosuite_pitch_change_deg": myo,
        "myosuite_head_x_dot_foot_forward": myosuite_head_forward_alignment(xml),
        "neck_flexion_sign_for_NeckInputY": sign,
        "pitch_convention": "positive = head forward axis above horizontal",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--myo-sim", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT / "neck_flexion_sign.json")
    args = parser.parse_args()
    result = derive(args.myo_sim)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logging.basicConfig(level=logging.INFO)
    sign = result["neck_flexion_sign_for_NeckInputY"]
    logger.info("Wrote %s: sign %+.0f", args.out, sign)


if __name__ == "__main__":
    main()
