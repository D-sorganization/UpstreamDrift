"""Unit tests for URDF float precision and versioned model bundles.

Covers MV-01 (Issue #10477) requirements:
1. Deterministic round-trip-safe float serialization (17 significant digits).
2. Finite-value enforcement (rejects NaN and infinity).
3. Versioned bundle manifest binding canonical spec, URDF, sidecars, and mesh assets.
4. Detection of missing or tampered sidecars and coordinate permutations.
5. Relocatable bundle archive packaging and path traversal rejection.
6. Diagnostic qualification status (dynamics blocked when sidecar missing).
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_generation.builders.urdf_writer import URDFWriter
from src.shared.python.model_generation.core.types import (
    Inertia,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Link,
    Origin,
)
from src.shared.python.model_generation.export.model_bundle import (
    IncompletePhysicsError,
    ModelBundle,
    ModelBundleManifest,
    export_model_bundle,
    load_model_bundle,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def calibrated_spec_path() -> Path:
    """Path to the 44-coordinate calibrated full-body driver spec."""
    path = (
        Path(__file__).parents[3]
        / "docs"
        / "development"
        / "full_body_models"
        / "evidence"
        / "ground_support"
        / "anthro_driver"
        / "full_body_spec_hipcal_scaled.json"
    )
    if not path.exists():
        pytest.skip(f"Calibrated spec fixture not found at {path}")
    return path


@pytest.fixture
def calibrated_spec_bytes(calibrated_spec_path: Path) -> bytes:
    """Raw bytes of the calibrated full-body driver spec."""
    return calibrated_spec_path.read_bytes()


def test_urdf_writer_float_precision_preservation() -> None:
    """Verify that URDFWriter preserves floating-point precision up to 17 digits.

    Prior .6g format truncated values such as 0.1736481776669303 to 0.173648,
    causing frame transform errors ~9e-6 and mass matrix errors ~4e-5.
    17-digit deterministic serialization preserves exact values to machine epsilon.
    """
    exact_val = 0.1736481776669303
    exact_mass = 1.2345678901234567
    exact_inertia = 9.876543210987654e-5

    link = Link(
        name="test_precision_link",
        inertia=Inertia(
            ixx=exact_inertia,
            iyy=exact_inertia,
            izz=exact_inertia,
            mass=exact_mass,
            center_of_mass=(exact_val, -exact_val, 0.0),
        ),
    )
    writer = URDFWriter(pretty_print=False)
    xml = writer.write("precision_robot", [link], [])

    # The serialized XML must contain high-precision representation
    assert f"{exact_val:.17g}" in xml
    assert f"{exact_mass:.17g}" in xml
    assert f"{exact_inertia:.17g}" in xml

    # Verify parsed float matches within 1e-15
    import defusedxml.ElementTree as ET

    root = ET.fromstring(xml)
    inertial = root.find(".//inertial")
    assert inertial is not None
    mass_elem = inertial.find("mass")
    assert mass_elem is not None
    parsed_mass = float(mass_elem.get("value", "0"))
    assert abs(parsed_mass - exact_mass) < 1e-15


def test_urdf_writer_rejects_non_finite_floats() -> None:
    """Verify that NaN and Inf are rejected during serialization."""
    nan_link = Link(
        name="nan_link",
        inertia=Inertia(
            ixx=1.0,
            iyy=1.0,
            izz=1.0,
            mass=float("nan"),
        ),
    )
    writer = URDFWriter()
    with pytest.raises(ValueError, match="Non-finite float value"):
        writer.write("nan_robot", [nan_link], [])

    inf_joint = Joint(
        name="inf_joint",
        parent="base",
        child="child",
        joint_type=JointType.REVOLUTE,
        origin=Origin(xyz=(float("inf"), 0.0, 0.0)),
    )
    base_link = Link(name="base", inertia=Inertia(1.0, 1.0, 1.0, 1.0))
    child_link = Link(name="child", inertia=Inertia(1.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="Non-finite float value"):
        writer.write("inf_robot", [base_link, child_link], [inf_joint])


def test_export_model_bundle_44_coordinate_calibrated_spec(
    calibrated_spec_bytes: bytes,
) -> None:
    """Test exporting a complete, versioned model bundle from a 44-coordinate spec."""
    bundle = export_model_bundle(calibrated_spec_bytes)

    assert isinstance(bundle, ModelBundle)
    manifest = bundle.manifest
    assert manifest.schema_version == 1
    assert manifest.nq == 44
    assert manifest.nv == 44
    assert manifest.requires_sidecar is True
    assert len(manifest.coordinate_order) == 44

    # Check sidecar completeness and integrity
    assert bundle.sidecar is not None
    assert (
        bundle.sidecar["closure"]["name"]
        == "GolfSwing3D_Kinetic/Grip/RightHandOnClubForce"
    )
    assert len(bundle.sidecar["contact_spheres"]) == 6

    # Verify SHA-256 bindings
    import hashlib

    expected_model_sha = hashlib.sha256(calibrated_spec_bytes).hexdigest()
    assert manifest.model_sha256 == expected_model_sha
    expected_urdf_sha = hashlib.sha256(bundle.urdf_xml.encode("utf-8")).hexdigest()
    assert manifest.urdf_sha256 == expected_urdf_sha

    # Verify high precision in exported URDF
    assert "-1.0478" not in bundle.urdf_xml  # was truncated in .6g
    assert (
        "-1.0477999999999998" in bundle.urdf_xml
        or "-1.047800000000000" in bundle.urdf_xml
    )


def test_model_bundle_sidecar_tampering_detection(
    calibrated_spec_bytes: bytes,
) -> None:
    """Verify that tampering with sidecar data or hashes raises a validation error."""
    bundle = export_model_bundle(calibrated_spec_bytes)

    # Tamper with coordinate order in sidecar
    assert bundle.sidecar is not None
    tampered_sidecar = dict(bundle.sidecar)
    tampered_sidecar["coordinate_order"] = list(
        reversed(tampered_sidecar["coordinate_order"])
    )

    tampered_bundle = ModelBundle(
        manifest=bundle.manifest,
        urdf_xml=bundle.urdf_xml,
        sidecar=tampered_sidecar,
        raw_spec=bundle.raw_spec,
    )
    with pytest.raises(ValueError, match="Sidecar coordinate_order does not match"):
        tampered_bundle.validate()


def test_model_bundle_dynamics_blocked_when_sidecar_missing() -> None:
    """Verify that visual-only loading is permitted but dynamics qualification is blocked."""
    manifest = ModelBundleManifest(
        schema_version=1,
        model_sha256="dummy_model_hash",
        urdf_sha256="dummy_urdf_hash",
        sidecar_sha256="",
        nq=44,
        nv=44,
        coordinate_order=["q1", "q2"],
        requires_sidecar=True,
        incomplete_physics_status="missing_closure_contact_sidecar",
    )
    bundle = ModelBundle(
        manifest=manifest,
        urdf_xml="<robot name='test'/>",
        sidecar=None,
        raw_spec=b"{}",
    )

    # Visual loading works
    assert bundle.is_visual_only is True

    # Dynamics qualification fails closed
    with pytest.raises(IncompletePhysicsError, match="Dynamics qualification blocked"):
        bundle.require_qualified_dynamics()


def test_bundle_relocatable_archive_save_and_reload(
    calibrated_spec_bytes: bytes, tmp_path: Path
) -> None:
    """Test exporting a bundle to a relocatable zip archive and reopening it cleanly."""
    bundle = export_model_bundle(calibrated_spec_bytes)

    archive_path = tmp_path / "matched_model_bundle.zip"
    bundle.save_archive(archive_path)
    assert archive_path.exists()

    # Reload from archive
    reloaded = load_model_bundle(archive_path)
    reloaded.validate()

    assert reloaded.manifest.model_sha256 == bundle.manifest.model_sha256
    assert reloaded.manifest.urdf_sha256 == bundle.manifest.urdf_sha256
    assert reloaded.urdf_xml == bundle.urdf_xml
    assert reloaded.sidecar == bundle.sidecar


def test_bundle_archive_rejects_path_traversal(tmp_path: Path) -> None:
    """Verify that archive files containing path traversal (..) are rejected."""
    malicious_zip = tmp_path / "malicious.zip"
    with zipfile.ZipFile(malicious_zip, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"schema_version": 1}))
        zf.writestr("../../../evil.txt", "exploit")

    with pytest.raises(ValueError, match="traversal"):
        load_model_bundle(malicious_zip)


def test_bundle_extract_to_and_directory_loading(
    calibrated_spec_bytes: bytes, tmp_path: Path
) -> None:
    """Verify that extract_to unbundles model.urdf and that load_model_bundle supports directories."""
    bundle = export_model_bundle(calibrated_spec_bytes)
    bundle.mesh_assets["test_mesh.obj"] = b"v 0 0 0\n"

    target_dir = tmp_path / "extracted_bundle"
    urdf_path = bundle.extract_to(target_dir)

    assert urdf_path.exists()
    assert urdf_path.name == "model.urdf"
    assert (target_dir / "manifest.json").exists()
    assert (target_dir / "meshes" / "test_mesh.obj").read_bytes() == b"v 0 0 0\n"

    # Reload from directory
    reloaded_dir = load_model_bundle(target_dir)
    assert reloaded_dir.manifest.model_sha256 == bundle.manifest.model_sha256
    assert reloaded_dir.urdf_xml == bundle.urdf_xml
    assert "test_mesh.obj" in reloaded_dir.mesh_assets
