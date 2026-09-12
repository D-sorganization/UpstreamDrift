"""Native MuJoCo tree and explicit rigid closure contracts."""

import copy
import json

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.native_model import NativeMujocoModel

pytestmark = [pytest.mark.live_simulation, pytest.mark.requires_mujoco]


@pytest.fixture
def specification() -> dict:
    pytest.importorskip("mujoco")
    eye = np.eye(4).tolist()
    bodies = [{"name": "world", "solids": []}]
    joints, frames, names = [], [], []
    for name, mass in (("a", 2.0), ("b", 3.0)):
        bodies.append(
            {
                "name": name,
                "solids": [
                    {
                        "name": name,
                        "mass_kg": mass,
                        "com_m": [0, 0, 0],
                        "inertia_com_kg_m2": np.eye(3).tolist(),
                        "placement": eye,
                    }
                ],
            }
        )
        primitives = [
            {"primitive": p, "coordinate": name + p}
            for p in ("Px", "Py", "Pz", "Rx", "Ry", "Rz")
        ]
        names.extend(p["coordinate"] for p in primitives)
        joints.append(
            {
                "parent": "world",
                "child": name,
                "parent_to_base": eye,
                "child_to_follower": eye,
                "primitives": primitives,
            }
        )
        frames.append({"name": name, "body": name, "placement": eye})
    return {
        "schema_version": 1,
        "coordinate_order": names,
        "bodies": bodies,
        "joints": joints,
        "frames": frames,
        "gravity_m_s2": [0, 0, -9.81],
        "closure": {
            "body_a": "a",
            "body_b": "b",
            "placement_a": eye,
            "placement_b": eye,
        },
    }


def test_rigid_weld_shares_force_without_synthetic_mass(specification: dict) -> None:
    model = NativeMujocoModel(json.dumps(specification).encode())
    zero = dict.fromkeys(specification["coordinate_order"], 0.0)
    effort = dict(zero, aPx=10.0)
    result = model.accelerations(zero, zero, effort)
    assert result["aPx"] == pytest.approx(2.0, abs=1e-12)
    assert result["bPx"] == pytest.approx(2.0, abs=1e-12)
    assert result["aPz"] == pytest.approx(-9.81, abs=1e-12)
    assert sum(model.model.body_mass) == pytest.approx(5)
    assert not model.model.jnt_limited.any()
    assert not model.model.dof_damping.any()
    assert all(np.linalg.norm(v) < 1e-12 for v in model.closure_errors())


def test_composed_rotation_order(specification: dict) -> None:
    from scipy.spatial.transform import Rotation

    model = NativeMujocoModel(json.dumps(specification).encode())
    q = dict.fromkeys(specification["coordinate_order"], 0.0)
    q.update(aRx=0.2, aRy=-0.3, aRz=0.4, aPx=0.1)
    pose = model.frame_poses(q)["a"]
    np.testing.assert_allclose(
        pose[:3, :3],
        Rotation.from_euler("XYZ", [0.2, -0.3, 0.4]).as_matrix(),
        atol=1e-14,
    )
    np.testing.assert_allclose(pose[:3, 3], [0.1, 0, 0], atol=1e-14)


def test_reject_invalid_state_and_geometry(specification: dict) -> None:
    model = NativeMujocoModel(json.dumps(specification).encode())
    with pytest.raises(ValueError, match="inventory"):
        model.frame_poses({})
    bad = copy.deepcopy(specification)
    bad["joints"][0]["parent_to_base"][0][0] = 2
    with pytest.raises(ValueError, match="transform"):
        NativeMujocoModel(json.dumps(bad).encode())


def test_identity_is_bound(specification: dict) -> None:
    import hashlib

    raw = json.dumps(specification).encode()
    model = NativeMujocoModel(raw)
    assert model.model_sha256 == hashlib.sha256(raw).hexdigest()


def test_zero_mass_reference_solids_do_not_add_mass(specification: dict) -> None:
    ghost = copy.deepcopy(specification["bodies"][1]["solids"][0])
    ghost.update(name="ghost", mass_kg=0, inertia_com_kg_m2=np.zeros((3, 3)).tolist())
    specification["bodies"][1]["solids"].append(ghost)
    model = NativeMujocoModel(json.dumps(specification).encode())
    assert sum(model.model.body_mass) == pytest.approx(5)


def test_rotating_rigid_pair_has_centripetal_acceleration(specification: dict) -> None:
    specification["closure"]["placement_a"] = np.eye(4).tolist()
    specification["closure"]["placement_b"] = np.eye(4).tolist()
    specification["closure"]["placement_a"][0][3] = 0.6
    specification["closure"]["placement_b"][0][3] = -0.4
    model = NativeMujocoModel(json.dumps(specification).encode())
    zero = dict.fromkeys(specification["coordinate_order"], 0.0)
    q = dict(zero, bPx=1.0)
    v = dict(zero, aPy=-1.2, bPy=0.8, aRz=2.0, bRz=2.0)
    acceleration = model.accelerations(q, v, zero)
    assert acceleration["aPx"] == pytest.approx(2.4, abs=1e-12)
    assert acceleration["bPx"] == pytest.approx(-1.6, abs=1e-12)
    assert all(np.linalg.norm(value) < 1e-12 for value in model.closure_errors())
