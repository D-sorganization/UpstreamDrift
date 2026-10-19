"""Cross-engine forward kinematics of the landmark full-body-v2 trunk (HRG-D2).

Every exporter is compared with the specification's own forward kinematics
(``spec_body_poses``) on every named frame over 120 seeded configurations.
The tolerance is the measured exporter precision with a small margin: the
URDF, MJCF and OpenSim writers print 17 significant digits and every engine
composes the same transforms in double precision, so the worst case seen is
of order 1e-15 m and 1e-15 rad. ``FK_TOL`` (1e-12) is tighter than the
issue's 1e-6 by six orders and is never loosened.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching import full_body_v2 as v2
from src.shared.python.motion_matching import full_body_v2_geometry as geo
from src.shared.python.motion_matching.full_body_spec import upper_body_slice
from tests.helpers.opensim_support import require_opensim

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
N_CONFIGS = 120
SEED = 1317313
FK_TOL = 1e-12


@pytest.fixture(scope="module")
def doc() -> dict:
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    return geo.derive_landmark_v2_spec(v1, provenance="HRG-D2 cross-engine fixture")


@pytest.fixture(scope="module")
def doc_bytes(doc) -> bytes:
    return json.dumps(doc, sort_keys=True).encode("utf-8")


@pytest.fixture(scope="module")
def configs(doc) -> list[dict[str, float]]:
    """Seeded configurations; trunk coordinates span their soft limits +-20 %."""
    rng = np.random.default_rng(SEED)
    soft = doc["soft_limits_deg"]
    out = []
    for _ in range(N_CONFIGS):
        q = {}
        for name in doc["coordinate_order"]:
            if name.startswith("Translation"):
                q[name] = float(rng.uniform(-0.5, 0.5))
            elif name in soft:
                lo, hi = np.radians(soft[name])
                pad = 0.2 * (hi - lo)
                q[name] = float(rng.uniform(lo - pad, hi + pad))
            else:
                q[name] = float(rng.uniform(-0.6, 0.6))
        out.append(q)
    return out


def _spec_frames(doc: dict, q: dict[str, float]) -> dict[str, np.ndarray]:
    poses = spec_body_poses(doc, q)
    return {
        f["name"]: poses[f["body"]] @ np.asarray(f["placement"], dtype=float)
        for f in doc["frames"]
    }


def _errors(expected: np.ndarray, got: np.ndarray) -> tuple[float, float]:
    pos = float(np.linalg.norm(got[:3, 3] - expected[:3, 3]))
    rel = expected[:3, :3].T @ got[:3, :3]
    # Geodesic angle via the chord ||R - I||_F = 2 sqrt(2) sin(theta / 2);
    # unlike arccos of the trace this keeps full precision near zero.
    chord = float(np.linalg.norm(rel - np.eye(3)))
    return pos, 2.0 * float(np.arcsin(min(1.0, chord / (2.0 * np.sqrt(2.0)))))


def _check(name: str, doc: dict, configs, poses_of) -> None:
    worst_p, worst_r = 0.0, 0.0
    for q in configs:
        expected = _spec_frames(doc, q)
        got = poses_of(q)
        for frame, pose in expected.items():
            p, r = _errors(pose, np.asarray(got[frame], dtype=float))
            worst_p, worst_r = max(worst_p, p), max(worst_r, r)
    assert len(configs) >= 100
    assert worst_p <= FK_TOL, f"{name} position error {worst_p:.3e} m"
    assert worst_r <= FK_TOL, f"{name} rotation error {worst_r:.3e} rad"


@pytest.mark.requires_mujoco
def test_mujoco_fk_matches_spec(doc, doc_bytes, configs) -> None:
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, meta = export_full_body_mjcf(doc_bytes)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    adr = {n: model.joint(n).qposadr[0] for n in doc["coordinate_order"]}
    sites = {
        f: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
        for f, s in meta["frame_sites"].items()
    }

    def poses(q):
        for name, value in q.items():
            data.qpos[adr[name]] = value
        mujoco.mj_kinematics(model, data)
        out = {}
        for frame, sid in sites.items():
            pose = np.eye(4)
            pose[:3, :3] = data.site_xmat[sid].reshape(3, 3)
            pose[:3, 3] = data.site_xpos[sid]
            out[frame] = pose
        return out

    assert set(sites) == {f["name"] for f in doc["frames"]}
    _check("MuJoCo", doc, configs, poses)


@pytest.mark.requires_drake
def test_drake_urdf_fk_matches_spec(doc, configs) -> None:
    pytest.importorskip("pydrake.all")
    from src.engines.physics_engines.drake.python.full_body_model import (
        FullBodyDrakeModel,
    )

    model = FullBodyDrakeModel(doc)
    _check("Drake/URDF", doc, configs, model.frame_poses)


@pytest.mark.requires_pinocchio
def test_pinocchio_fk_and_v2_upper_body_model(doc, configs) -> None:
    pytest.importorskip("pinocchio")
    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )

    model = FullBodyPinocchioModel(doc)
    _check("Pinocchio", doc, configs, model.frame_poses)
    upper = model.upper_body_model()
    names = list(upper_body_slice(doc)["coordinate_order"])
    assert set(v2.V2_TRUNK.coordinates) <= set(names)
    for q in configs[:10]:
        sub = {k: q.get(k, 0.0) for k in names}
        full = model.frame_poses(q)
        for frame, pose in upper.frame_poses(sub).items():
            np.testing.assert_allclose(pose, full[frame], atol=FK_TOL)


@pytest.mark.requires_opensim
def test_opensim_fk_matches_spec(doc, doc_bytes, configs) -> None:
    require_opensim()
    from src.engines.physics_engines.opensim.python.full_body_parity import (
        OpenSimFullBodyParityAdapter,
    )

    adapter = OpenSimFullBodyParityAdapter(doc_bytes)
    _check("OpenSim", doc, configs, adapter.frame_poses)


def test_myosuite_mapping_reads_the_v2_thorax(doc) -> None:
    """MyoSuite is reached by segment-orientation IK on the spec skeleton.

    The thorax segment keeps the v1 body name, so the mapping consumes the v2
    pelvis-to-thorax rotation unchanged; the lumbar body has no MyoFullBody
    counterpart (its regional split is exempt, see ``EXPORT_STATUS``).
    """
    from src.shared.python.myofullbody import anatomy

    bodies = [b["name"] for b in doc["bodies"]]
    for frame in anatomy.spec_frame_defs().values():
        assert any(name.endswith(frame.body) for name in bodies), frame.body
    status = geo.EXPORT_STATUS["myosuite"]
    assert status["status"] == "mapped_thorax_only"
    assert "lumbar" in status["reason"]
    for engine in ("mujoco", "drake_urdf", "pinocchio", "opensim"):
        assert geo.EXPORT_STATUS[engine]["status"] == "exported"
