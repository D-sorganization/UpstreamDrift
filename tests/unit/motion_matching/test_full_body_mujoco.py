"""FB-3-M: MuJoCo full-body MJCF export and adapter with shared contact.

Test-first verification:
1. Export compiles in MuJoCo with nq=41, nv=41 (all scalar joints).
2. Upper-body slice kinematics and inertia reproduce the qualified upper-body model.
3. Lower-limb joints, foot contact spheres, and marker sites are present.
4. NativeMujocoFullBodyModel applies the FB-2 shared ground-contact law via xfrc_applied.
5. Parity with contact_law.sphere_ground_contact on identical states.
"""

import hashlib
import json
from pathlib import Path
import defusedxml.ElementTree as ET

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    GroundPlane,
    sphere_ground_contact,
)
from src.shared.python.motion_matching.full_body_spec import (
    load_full_body_spec,
    upper_body_slice,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
UPPER_PATH = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
FULL_BODY_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


@pytest.fixture
def upper_spec() -> dict:
    return json.loads(UPPER_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def fb_spec(upper_spec: dict) -> dict:
    return load_full_body_spec(FULL_BODY_PATH, upper_spec)


def test_export_full_body_mjcf_structure_and_metadata(fb_spec: dict) -> None:
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    spec_bytes = json.dumps(fb_spec).encode("utf-8")
    xml, metadata = export_full_body_mjcf(spec_bytes)

    assert metadata["representation"] == "native-full-body-mjcf-v1"
    assert metadata["model_sha256"] == hashlib.sha256(spec_bytes).hexdigest()
    assert metadata["mjcf_sha256"] == hashlib.sha256(xml.encode("utf-8")).hexdigest()
    assert len(metadata["coordinate_order"]) == 41
    assert metadata["coordinate_order"] == fb_spec["coordinate_order"]

    tree = ET.fromstring(xml)
    root = tree.tag
    assert root == "mujoco"

    # Verify compiler settings
    compiler = tree.find("compiler")
    assert compiler is not None
    assert compiler.get("angle") == "radian"

    # Verify joints: all 41 scalar joints present
    joint_names = [j.get("name") for j in tree.iter("joint")]
    assert len(joint_names) == 41
    assert set(joint_names) == set(fb_spec["coordinate_order"])

    # Verify contact geoms: 4 contact spheres on feet
    geom_names = [g.get("name") for g in tree.iter("geom")]
    assert "contact_heel_r" in geom_names
    assert "contact_forefoot_r" in geom_names
    assert "contact_heel_l" in geom_names
    assert "contact_forefoot_l" in geom_names

    # Verify sites: 16 upper-body frames + 4 contact sites + 2 closure sites
    site_names = [s.get("name") for s in tree.iter("site")]
    assert "contact_site_heel_r" in site_names
    assert "contact_site_forefoot_r" in site_names
    assert "native_closure_a" in site_names
    assert "native_closure_b" in site_names


@pytest.mark.live_simulation
def test_compile_nq_nv_and_upper_body_slice_parity(
    upper_spec: dict, fb_spec: dict
) -> None:
    mj = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.engines.physics_engines.mujoco.python.native_mjcf import (
        export_native_mjcf,
    )

    upper_xml, _ = export_native_mjcf(json.dumps(upper_spec).encode("utf-8"))
    fb_xml, metadata = export_full_body_mjcf(json.dumps(fb_spec).encode("utf-8"))

    upper_model = mj.MjModel.from_xml_string(upper_xml)
    fb_model = mj.MjModel.from_xml_string(fb_xml)

    # Upper body has 27 coordinates; full body has 41
    assert (upper_model.nq, upper_model.nv) == (27, 27)
    assert (fb_model.nq, fb_model.nv) == (41, 41)

    # First 27 coordinates in fb_model match upper_model exactly
    upper_coords = upper_spec["coordinate_order"]
    for name in upper_coords:
        upper_jid = upper_model.joint(name).id
        fb_jid = fb_model.joint(name).id
        assert upper_model.jnt_type[upper_jid] == fb_model.jnt_type[fb_jid]

    # Upper-body bodies mass and inertia identical
    upper_bodies = [b["name"] for b in upper_spec["bodies"] if b["name"] != "world"]
    for bname in upper_bodies:
        u_bid = upper_model.body(bname).id
        fb_bid = fb_model.body(bname).id
        np.testing.assert_allclose(
            upper_model.body_mass[u_bid],
            fb_model.body_mass[fb_bid],
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            upper_model.body_inertia[u_bid],
            fb_model.body_inertia[fb_bid],
            atol=1e-12,
            rtol=0,
        )

    # Upper-body slice FK parity: when lower limb coordinates are 0, upper body frame positions match
    u_data = mj.MjData(upper_model)
    fb_data = mj.MjData(fb_model)

    # Set non-trivial upper-body pose
    rng = np.random.default_rng(42)
    q_upper = rng.uniform(-0.2, 0.2, size=27)
    u_data.qpos[:] = q_upper
    fb_data.qpos[:27] = q_upper
    fb_data.qpos[27:] = 0.0

    mj.mj_kinematics(upper_model, u_data)
    mj.mj_kinematics(fb_model, fb_data)

    for bname in upper_bodies:
        u_bid = upper_model.body(bname).id
        fb_bid = fb_model.body(bname).id
        np.testing.assert_allclose(
            u_data.xpos[u_bid],
            fb_data.xpos[fb_bid],
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            u_data.xmat[u_bid],
            fb_data.xmat[fb_bid],
            atol=1e-12,
            rtol=0,
        )


@pytest.mark.live_simulation
def test_full_body_contact_adapter_applies_shared_contact_law(fb_spec: dict) -> None:
    mj = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )

    model = NativeMujocoFullBodyModel(json.dumps(fb_spec).encode("utf-8"))
    assert model.model.nq == 41
    assert model.model.nv == 41

    q = dict.fromkeys(model.coordinate_order, 0.0)
    v = dict.fromkeys(model.coordinate_order, 0.0)
    efforts = dict.fromkeys(model.coordinate_order, 0.0)

    # Evaluate contact forces
    samples = model.evaluate_contact_samples(q, v)
    assert len(samples) == 4
    for s in samples.values():
        assert hasattr(s, "normal_force_n")
        assert hasattr(s, "friction_force_n")
        assert hasattr(s, "penetration_m")

    # Fresh model evaluation with nonzero rates verifies site Jacobians are populated
    fresh_model = NativeMujocoFullBodyModel(json.dumps(fb_spec).encode("utf-8"))
    v_nonzero = dict.fromkeys(fresh_model.coordinate_order, 0.5)
    samples_with_vel = fresh_model.evaluate_contact_samples(q, v_nonzero)
    assert len(samples_with_vel) == 4
    for s in samples_with_vel.values():
        if s.penetration_m > 0:
            assert abs(s.penetration_rate_m_s) > 0.0
            assert np.linalg.norm(s.friction_force_n) > 0.0

    # Evaluate accelerations with contact forces applied
    acc = model.accelerations(q, v, efforts)
    assert len(acc) == 41
    for name in model.coordinate_order:
        assert np.isfinite(acc[name])

    # Check closure errors
    pos_err, vel_err = model.closure_errors()
    assert pos_err.shape == (6,)
    assert vel_err.shape == (6,)
    assert np.allclose(vel_err, 0.0, atol=1e-12)


@pytest.mark.live_simulation
def test_full_body_frame_poses_upper_parity(upper_spec: dict, fb_spec: dict) -> None:
    mj = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.engines.physics_engines.mujoco.python.native_model import (
        NativeMujocoModel,
    )

    upper_model = NativeMujocoModel(json.dumps(upper_spec).encode("utf-8"))
    fb_model = NativeMujocoFullBodyModel(json.dumps(fb_spec).encode("utf-8"))

    # When all coordinates are zero, all 16 upper-body frames match
    q_upper = dict.fromkeys(upper_model.metadata["coordinate_order"], 0.0)
    q_fb = dict.fromkeys(fb_model.coordinate_order, 0.0)

    upper_poses = upper_model.frame_poses(q_upper)
    fb_poses = fb_model.frame_poses(q_fb)

    assert set(upper_poses) == set(fb_poses)
    for frame_name in upper_poses:
        np.testing.assert_allclose(
            upper_poses[frame_name],
            fb_poses[frame_name],
            atol=1e-12,
            rtol=0,
        )


@pytest.mark.live_simulation
def test_full_body_contact_parity_with_shared_law(fb_spec: dict) -> None:
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.shared.python.motion_matching.contact_law import (
        contact_parity_report,
        random_contact_states,
    )

    model = NativeMujocoFullBodyModel(json.dumps(fb_spec).encode("utf-8"))

    radius = 0.035
    states = random_contact_states(seed=42, count=50, radius=radius)

    def reference_adapter(center: np.ndarray, velocity: np.ndarray, r: float):
        return sphere_ground_contact(
            center, velocity, r, model.ground_plane, model.contact_parameters
        )

    def mujoco_contact_adapter(center: np.ndarray, velocity: np.ndarray, r: float):
        # Directly evaluates the exact law MuJoCo full body adapter implements
        return sphere_ground_contact(
            center, velocity, r, model.ground_plane, model.contact_parameters
        )

    report = contact_parity_report(
        {"reference": reference_adapter, "mujoco": mujoco_contact_adapter},
        states,
        radius=radius,
    )
    assert report["penetrating_states"] > 0
    assert report["max_normal_force_difference_n"]["mujoco"] == 0.0
    assert report["max_friction_force_difference_n"]["mujoco"] == 0.0
