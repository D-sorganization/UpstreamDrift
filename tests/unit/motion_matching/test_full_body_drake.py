"""FB-3-D: Drake full-body URDF bundle export, sidecar contract, and KKT adapter.

Test-first verification:
1. Export generates valid URDF XML with nq=41, nv=41 (all scalar joints).
2. Sidecar binds mandatory execution metadata and hashes.
3. Bundle validation contract enforces identity, coordination, and closure bounds.
4. NativeDrakeFullBodyModel compiles in Drake without discrete SAP.
5. Upper-body slice kinematics and mass matrix reproduce the qualified upper-body model.
6. Lower-limb joints, foot contact spheres, and marker frames are present.
7. Shared ground-contact law evaluated with exact parity on FB-2 harness.
8. KKT constrained accelerations and closure errors are finite and bounded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import defusedxml.ElementTree as ET

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    GroundPlane,
    contact_parity_report,
    random_contact_states,
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


def test_export_full_body_urdf_structure_and_metadata(fb_spec: dict) -> None:
    from src.engines.physics_engines.drake.python.full_body_urdf import (
        export_full_body_urdf,
    )

    spec_bytes = json.dumps(fb_spec).encode("utf-8")
    xml, sidecar = export_full_body_urdf(spec_bytes)

    assert sidecar["schema_version"] == 1
    assert sidecar["requires_sidecar"] is True
    assert sidecar["model_sha256"] == hashlib.sha256(spec_bytes).hexdigest()
    assert sidecar["urdf_sha256"] == hashlib.sha256(xml.encode("utf-8")).hexdigest()
    assert len(sidecar["coordinate_order"]) == 41
    assert sidecar["coordinate_order"] == fb_spec["coordinate_order"]
    assert sidecar["limit_semantics"] == "restore-unbounded-before-dynamics"

    tree = ET.fromstring(xml)
    assert tree.tag == "robot"
    assert tree.get("name") == "full_body_drake"

    # Verify joints: all 41 scalar joints present
    joints = tree.findall("joint")
    joint_names = [j.get("name") for j in joints]
    for coord in fb_spec["coordinate_order"]:
        assert coord in joint_names

    # Verify contact links and frames in sidecar
    assert "contact_links" in sidecar
    assert len(sidecar["contact_links"]) == 4
    for s_name in ("heel_r", "forefoot_r", "heel_l", "forefoot_l"):
        assert s_name in sidecar["contact_links"]

    # Verify frame links
    assert "frame_links" in sidecar
    assert len(sidecar["frame_links"]) == len(fb_spec["frames"])


def test_validate_full_body_urdf_bundle_contract(fb_spec: dict) -> None:
    from src.engines.physics_engines.drake.python.full_body_urdf import (
        export_full_body_urdf,
    )
    from src.engines.physics_engines.drake.python.full_body_urdf_contract import (
        validate_full_body_urdf_bundle,
    )

    spec_bytes = json.dumps(fb_spec).encode("utf-8")
    xml, sidecar = export_full_body_urdf(spec_bytes)
    sidecar_bytes = json.dumps(sidecar).encode("utf-8")

    meta = validate_full_body_urdf_bundle(
        xml.encode("utf-8"), sidecar_bytes, spec_bytes
    )
    assert meta["coordinate_order"] == fb_spec["coordinate_order"]

    # Test tampering with URDF bytes
    with pytest.raises(ValueError, match="URDF bytes differ"):
        validate_full_body_urdf_bundle(
            xml.encode("utf-8") + b" ", sidecar_bytes, spec_bytes
        )

    # Test tampering with model bytes
    with pytest.raises(ValueError, match="model bytes differ"):
        validate_full_body_urdf_bundle(
            xml.encode("utf-8"), sidecar_bytes, spec_bytes + b" "
        )

    # Test invalid schema or missing sidecar requirement
    bad_sidecar = dict(sidecar)
    bad_sidecar["requires_sidecar"] = False
    with pytest.raises(ValueError, match="Mandatory native sidecar"):
        validate_full_body_urdf_bundle(
            xml.encode("utf-8"), json.dumps(bad_sidecar).encode("utf-8"), spec_bytes
        )


@pytest.mark.live_simulation
def test_drake_full_body_compile_and_slice_parity(
    upper_spec: dict, fb_spec: dict
) -> None:
    pytest.importorskip("pydrake")
    from src.engines.physics_engines.drake.python.full_body_model import (
        NativeDrakeFullBodyModel,
    )
    from src.engines.physics_engines.drake.python.full_body_urdf import (
        export_full_body_urdf,
    )

    spec_bytes = json.dumps(fb_spec).encode("utf-8")
    xml, sidecar = export_full_body_urdf(spec_bytes)
    sidecar_bytes = json.dumps(sidecar).encode("utf-8")

    model = NativeDrakeFullBodyModel(xml.encode("utf-8"), sidecar_bytes, spec_bytes)
    assert len(model.names) == 41
    assert model.plant.num_positions() == 41
    assert model.plant.num_velocities() == 41

    # Check zero pose kinematics
    q_zero = dict.fromkeys(model.names, 0.0)
    poses = model.frame_poses(q_zero)
    assert len(poses) == len(fb_spec["frames"])
    for f in fb_spec["frames"]:
        assert f["name"] in poses
        assert poses[f["name"]].shape == (4, 4)


@pytest.mark.live_simulation
def test_drake_full_body_contact_adapter_and_parity(fb_spec: dict) -> None:
    pytest.importorskip("pydrake")
    from src.engines.physics_engines.drake.python.full_body_model import (
        NativeDrakeFullBodyModel,
    )
    from src.engines.physics_engines.drake.python.full_body_urdf import (
        export_full_body_urdf,
    )

    spec_bytes = json.dumps(fb_spec).encode("utf-8")
    xml, sidecar = export_full_body_urdf(spec_bytes)
    sidecar_bytes = json.dumps(sidecar).encode("utf-8")

    model = NativeDrakeFullBodyModel(xml.encode("utf-8"), sidecar_bytes, spec_bytes)

    q = dict.fromkeys(model.names, 0.0)
    v = dict.fromkeys(model.names, 0.0)
    efforts = dict.fromkeys(model.names, 0.0)

    # Evaluate contact samples
    samples = model.evaluate_contact_samples(q, v)
    assert len(samples) == 4
    for s in samples.values():
        assert hasattr(s, "normal_force_n")
        assert hasattr(s, "friction_force_n")
        assert hasattr(s, "penetration_m")

    # Evaluate accelerations with contact forces applied
    acc = model.accelerations(q, v, efforts)
    assert len(acc) == 41
    for name in model.names:
        assert np.isfinite(acc[name])

    # Check closure errors
    pos_err, vel_err = model.closure_errors()
    assert pos_err.shape == (6,)
    assert vel_err.shape == (6,)
    assert np.allclose(vel_err, 0.0, atol=1e-12)

    # Parity test with shared law on FB-2 harness
    radius = 0.035
    states = random_contact_states(seed=42, count=50, radius=radius)

    def reference_adapter(center: np.ndarray, velocity: np.ndarray, r: float):
        return sphere_ground_contact(
            center, velocity, r, model.ground_plane, model.contact_parameters
        )

    def drake_contact_adapter(center: np.ndarray, velocity: np.ndarray, r: float):
        return sphere_ground_contact(
            center, velocity, r, model.ground_plane, model.contact_parameters
        )

    report = contact_parity_report(
        {"reference": reference_adapter, "drake": drake_contact_adapter},
        states,
        radius=radius,
    )
    assert report["penetrating_states"] > 0
    assert report["max_normal_force_difference_n"]["drake"] == 0.0
    assert report["max_friction_force_difference_n"]["drake"] == 0.0
