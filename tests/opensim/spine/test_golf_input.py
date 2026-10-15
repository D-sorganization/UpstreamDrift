"""Golf humanoid input reader (SPN-9a): native pelvis->thorax rotation + SLERP.

Synthetic ``.sto`` files carry invented coordinate values; the model is the
in-repo ``golf_humanoid.osim``.  The evidence-file test records the SHA-256 of
the repository evidence inputs and skips with a reason when they are absent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.spine.golf_input import (
    GolfTrunkInput,
    back_joint_rotation,
    read_golf_humanoid_motion,
    resample,
)
from tests.helpers.opensim_support import require_opensim

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
EVID = REPO / "docs/development/opensim_tour_matching/evidence/os7_moco_g1/inputs"
EVID_STO = EVID / "ik_states_full.sto"
EVID_OSIM = EVID / "golf_humanoid_scaled_tour_markers.osim"
EVID_STO_SHA = "325796072973aad8b007c5f975253eaf62057b91924cada85e2dae381cd1c2cc"
EVID_OSIM_SHA = "7dd1da1764bd8569d9e5de14eae6a249f8845e9fddf0f343796bef2ae8137381"

LUMBAR = ("lumbar_extension", "lumbar_bending", "lumbar_rotation")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sto(path: Path, t: np.ndarray, cols: dict[str, np.ndarray], deg: bool):
    lines = [
        "synthetic",
        "version=1",
        f"nRows={len(t)}",
        f"nColumns={len(cols) + 1}",
        f"inDegrees={'yes' if deg else 'no'}",
        "endheader",
        "time\t" + "\t".join(cols),
    ]
    for i, ti in enumerate(t):
        row = [f"{ti:.12f}"] + [f"{cols[k][i]:.15g}" for k in cols]
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n")


def _synthetic(tmp_path: Path, deg: bool = False) -> tuple[Path, np.ndarray]:
    t = np.linspace(0.0, 0.5, 6)
    q = np.column_stack(
        [
            0.05 + 0.3 * np.sin(3 * t),
            0.1 - 0.4 * t,
            0.2 + 1.2 * t,
        ]
    )
    scale = 180.0 / np.pi if deg else 1.0
    cols = {n: q[:, i] * scale for i, n in enumerate(LUMBAR)}
    cols["hip_flexion_r"] = 0.2 * t * scale
    cols["pelvis_tx"] = 0.1 * t
    path = tmp_path / "synthetic_states.sto"
    _write_sto(path, t, cols, deg)
    return path, q


def test_back_joint_rotation_is_zxy_and_proper():
    rot = back_joint_rotation(0.2, -0.1, 0.4)
    ref = Rotation.from_euler("ZXY", [0.2, -0.1, 0.4]).as_matrix()
    assert np.allclose(rot, ref, atol=1e-12)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    with pytest.raises(ValueError):
        back_joint_rotation(float("nan"), 0.0, 0.0)


@pytest.mark.requires_opensim
@pytest.mark.parametrize("deg", [False, True])
def test_pelvis_to_thorax_matches_native_and_adr_item_4(tmp_path, deg):
    osim = require_opensim()
    path, q = _synthetic(tmp_path, deg)
    data = read_golf_humanoid_motion(path, MODEL)
    assert isinstance(data, GolfTrunkInput)
    assert data.pelvis_to_thorax.shape == (6, 3, 3)
    assert data.pelvis_pose_ground.shape == (6, 4, 4)
    assert not data.pelvis_to_thorax.flags.writeable
    assert set(data.other_coordinates) >= {"hip_flexion_r"}
    assert not (set(LUMBAR) & set(data.other_coordinates))
    assert data.events == {}
    assert data.source["file"] == path.name
    assert data.source["sha256"] == _sha(path)
    assert "/" not in data.source["file"]

    model = osim.Model(str(MODEL))
    state = model.initSystem()
    po = model.getJointSet().get("back").get_frames(0)
    to = model.getJointSet().get("back").get_frames(1)
    assert po.getName() == "pelvis_offset" and to.getName() == "torso_offset"

    def offset_rot(frame):
        e = frame.get_orientation()
        return Rotation.from_euler("XYZ", [e.get(0), e.get(1), e.get(2)]).as_matrix()

    r_po, r_to = offset_rot(po), offset_rot(to)
    for i in range(q.shape[0]):
        native = None
        for j, n in enumerate(LUMBAR):
            model.updCoordinateSet().get(n).setValue(state, float(q[i, j]), False)
        model.realizePosition(state)
        rp = model.getBodySet().get("pelvis").getTransformInGround(state).R()
        rt = model.getBodySet().get("torso").getTransformInGround(state).R()
        mp = np.array([[rp.get(a, b) for b in range(3)] for a in range(3)])
        mt = np.array([[rt.get(a, b) for b in range(3)] for a in range(3)])
        native = mp.T @ mt
        assert np.allclose(data.pelvis_to_thorax[i], native, atol=1e-9)
        composed = r_po @ back_joint_rotation(*q[i]) @ r_to.T
        assert np.allclose(data.pelvis_to_thorax[i], composed, atol=1e-9)


def test_slerp_resample_endpoints_and_no_wrap_artifacts():
    # Rotation through ~2*pi total about Z, sampled coarsely: per-angle linear
    # interpolation of wrapped angles would jump; SLERP must stay smooth.
    t = np.linspace(0.0, 1.0, 5)
    ang = np.linspace(0.0, 2.0 * np.pi - 0.2, 5)
    rots = Rotation.from_rotvec(np.outer(ang, [0.0, 0.0, 1.0])).as_matrix()
    poses = np.tile(np.eye(4), (5, 1, 1))
    poses[:, 0, 3] = t
    data = GolfTrunkInput(
        time_s=t,
        pelvis_to_thorax=rots,
        pelvis_pose_ground=poses,
        other_coordinates={"hip_flexion_r": 2.0 * t},
        events={"impact": 0.75},
        source={"file": "x.sto", "sha256": "0" * 64},
    )
    out = resample(data, 100.0)
    assert out.time_s[0] == pytest.approx(0.0)
    assert out.time_s[-1] == pytest.approx(1.0)
    assert np.allclose(out.pelvis_to_thorax[0], rots[0], atol=1e-12)
    assert np.allclose(out.pelvis_to_thorax[-1], rots[-1], atol=1e-12)
    assert np.allclose(out.pelvis_pose_ground[-1], poses[-1], atol=1e-12)
    assert np.allclose(out.other_coordinates["hip_flexion_r"], 2.0 * out.time_s)
    steps = [
        np.linalg.norm(Rotation.from_matrix(a.T @ b).as_rotvec())
        for a, b in zip(
            out.pelvis_to_thorax[:-1], out.pelvis_to_thorax[1:], strict=True
        )
    ]
    assert max(steps) < 0.1  # no 2*pi jump; ~0.065 rad/step expected
    assert out.events == data.events
    assert not out.pelvis_to_thorax.flags.writeable
    for r in out.pelvis_to_thorax:
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    with pytest.raises(ValueError):
        resample(data, 0.0)


@pytest.mark.requires_opensim
def test_evidence_file_loads_with_finite_rotations():
    require_opensim()
    if not (EVID_STO.exists() and EVID_OSIM.exists()):
        pytest.skip("os7_moco_g1 evidence inputs are not present")
    assert _sha(EVID_STO) == EVID_STO_SHA
    assert _sha(EVID_OSIM) == EVID_OSIM_SHA
    data = read_golf_humanoid_motion(EVID_STO, EVID_OSIM)
    assert data.pelvis_to_thorax.shape[0] > 100
    assert np.all(np.isfinite(data.pelvis_to_thorax))
    assert np.all(np.isfinite(data.pelvis_pose_ground))
    assert data.source["sha256"] == EVID_STO_SHA
    for r in data.pelvis_to_thorax[::50]:
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-9)
