"""Source and bilateral registration contracts for a generic shoulder hypothesis."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from dataclasses import replace

import numpy as np
import pytest
from tests.helpers.opensim_support import require_opensim
from defusedxml import ElementTree

from src.engines.physics_engines.opensim.python.tour_matching.generic_shoulder_lamina import (
    ShoulderLaminaManifest,
    _assert_allowed_xml_changes,
    admit_reviewed_557_manifest,
    audit_reviewed_557_shoulder_candidate,
    build_generic_shoulder_lamina,
    build_reviewed_557_shoulder_lamina,
    REGISTRATION,
    REVIEWED_557_SOURCE_SHA256,
    REVIEWED_RIGHT_MESH_SHA256,
    _call_provider_worker,
)

pytestmark = pytest.mark.integration


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pinned_provider_worker_does_not_mutate_parent_import_state() -> None:
    before_path = tuple(sys.path)
    package_prefix = "shared.python.humanoid_character_builder.mesh"
    before_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == package_prefix or name.startswith(package_prefix + ".")
    }
    identity = _call_provider_worker()
    after_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == package_prefix or name.startswith(package_prefix + ".")
    }
    assert tuple(sys.path) == before_path
    assert after_modules == before_modules
    assert len(identity["provider_sha256"]) == 64
    assert len(identity["representation_sha256"]) == 64


def _mesh(path: Path) -> None:
    path.write_text(
        '<VTKFile type="PolyData"><PolyData><Piece NumberOfPoints="4" '
        'NumberOfPolys="4"><PointData><DataArray Name="Normals" format="ascii">'
        "0 0 1 0 0 1 0 0 1 0 0 1</DataArray></PointData>"
        '<Points><DataArray NumberOfComponents="3" '
        'format="ascii">0 0 0 0.1 0 0 0 0.2 0 0 0 0.3</DataArray>'
        '</Points><Polys><DataArray Name="connectivity" format="ascii">'
        '0 1 2 0 1 3 0 2 3 1 2 3</DataArray><DataArray Name="offsets" '
        'format="ascii">3 6 9 12</DataArray></Polys></Piece></PolyData></VTKFile>',
        encoding="utf-8",
    )


def _source(path: Path) -> None:
    bodies = []
    for name, filename in (
        ("clavicle_r", "clavicle.vtp"),
        ("scapula_r", "scapula.vtp"),
        ("clavicle_l", "clavicle_l.vtp"),
        ("scapula_l", "scapula_l.vtp"),
    ):
        bodies.append(
            f'<Body name="{name}"><mass>1</mass><mass_center>0 0 0</mass_center>'
            "<inertia>0.02 0.02 0.02 0 0 0</inertia><attached_geometry>"
            f"<Mesh><socket_frame>..</socket_frame><scale_factors>1 1 1</scale_factors>"
            f"<mesh_file>{filename}</mesh_file></Mesh></attached_geometry></Body>"
        )
    path.write_text(
        "<OpenSimDocument><Model><length_units>meters</length_units><BodySet>"
        f"<objects>{''.join(bodies)}</objects></BodySet></Model></OpenSimDocument>",
        encoding="utf-8",
    )


@pytest.fixture
def manifest(tmp_path: Path) -> ShoulderLaminaManifest:
    source = tmp_path / "source.osim"
    clavicle, scapula = tmp_path / "clavicle.vtp", tmp_path / "scapula.vtp"
    _source(source)
    _mesh(clavicle)
    _mesh(scapula)
    return ShoulderLaminaManifest(
        source,
        _sha(source),
        clavicle,
        _sha(clavicle),
        scapula,
        _sha(scapula),
        "right-identity-left-local-z-reflection/1.0.0",
    )


def test_candidate_preserves_only_declared_body_fields(
    manifest: ShoulderLaminaManifest, tmp_path: Path
) -> None:
    result = build_generic_shoulder_lamina(manifest, tmp_path / "candidate")
    source = ElementTree.parse(manifest.source_path)
    candidate = ElementTree.parse(result.model_path)
    source_bodies = source.findall("./Model/BodySet/objects/Body")
    candidate_bodies = candidate.findall("./Model/BodySet/objects/Body")
    assert [b.get("name") for b in source_bodies] == [
        b.get("name") for b in candidate_bodies
    ]
    for original, derived in zip(source_bodies, candidate_bodies, strict=True):
        assert original.findtext("mass") == derived.findtext("mass")
        assert original.findtext("mass_center") != derived.findtext("mass_center")
        assert original.findtext("inertia") != derived.findtext("inertia")
        assert original.findtext("attached_geometry/Mesh/socket_frame") == ".."
        assert derived.findtext("attached_geometry/Mesh/scale_factors") == "1 1 1"
    assert result.source_sha256 == manifest.source_sha256
    assert result.model_sha256 == _sha(result.model_path)
    vendor_src = Path(__file__).resolve().parents[2] / "vendor/ud-tools/src"
    assert result.lamina_provider_sha256 == _sha(
        vendor_src / "shared/python/humanoid_character_builder/mesh/triangle_lamina.py"
    )
    assert result.inertia_representation_sha256 == _sha(
        vendor_src
        / "shared/python/humanoid_character_builder/mesh/inertia_calculator.py"
    )
    assert result.provider_worker_sha256 == _sha(
        Path(__file__).resolve().parents[2]
        / "scripts/opensim/triangle_lamina_provider_worker.py"
    )


def test_left_is_exact_local_reflection_of_right_mass_moments(
    manifest: ShoulderLaminaManifest, tmp_path: Path
) -> None:
    result = build_generic_shoulder_lamina(manifest, tmp_path / "candidate")
    reflection = np.diag([1.0, 1.0, -1.0])
    for part in ("clavicle", "scapula"):
        right = result.bodies[f"{part}_r"]
        left = result.bodies[f"{part}_l"]
        np.testing.assert_allclose(
            left.center_m, reflection @ right.center_m, atol=1e-14
        )
        np.testing.assert_allclose(
            left.inertia_kg_m2,
            reflection @ right.inertia_kg_m2 @ reflection.T,
            atol=1e-14,
        )


def test_source_and_mesh_mutations_reject_before_candidate_write(
    manifest: ShoulderLaminaManifest, tmp_path: Path
) -> None:
    manifest.clavicle_mesh_path.write_bytes(
        manifest.clavicle_mesh_path.read_bytes() + b" "
    )
    with pytest.raises(ValueError, match="mesh.*hash"):
        build_generic_shoulder_lamina(manifest, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def test_undeclared_frame_or_registration_rejects(
    manifest: ShoulderLaminaManifest, tmp_path: Path
) -> None:
    xml = manifest.source_path.read_text(encoding="utf-8")
    manifest.source_path.write_text(xml.replace("1 1 1", "1 1 2", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="source.*hash"):
        build_generic_shoulder_lamina(manifest, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def test_undeclared_path_or_mass_edit_is_not_hidden_by_shoulder_edit(
    manifest: ShoulderLaminaManifest, tmp_path: Path
) -> None:
    candidate = build_generic_shoulder_lamina(manifest, tmp_path / "candidate")
    raw = candidate.model_path.read_bytes()
    assert b"<mass>1</mass>" in raw
    with pytest.raises(ValueError, match="undeclared"):
        _assert_allowed_xml_changes(
            manifest.source_path.read_bytes(),
            raw.replace(b"<mass>1</mass>", b"<mass>2</mass>", 1),
        )


def test_left_visual_vertices_winding_and_normals_are_reflected(
    manifest: ShoulderLaminaManifest, tmp_path: Path
) -> None:
    result = build_generic_shoulder_lamina(manifest, tmp_path / "candidate")
    geometry = result.model_path.parent / "Geometry"
    right = ElementTree.parse(geometry / "generic_clavicle_r.vtp")
    left = ElementTree.parse(geometry / "generic_clavicle_l.vtp")
    right_points = np.fromstring(
        right.findtext(".//Points/DataArray"), sep=" "
    ).reshape(-1, 3)
    left_points = np.fromstring(left.findtext(".//Points/DataArray"), sep=" ").reshape(
        -1, 3
    )
    right_faces = np.fromstring(
        right.findtext(".//Polys/DataArray[@Name='connectivity']"), sep=" ", dtype=int
    ).reshape(-1, 3)
    left_faces = np.fromstring(
        left.findtext(".//Polys/DataArray[@Name='connectivity']"), sep=" ", dtype=int
    ).reshape(-1, 3)
    left_normals = np.fromstring(
        left.findtext(".//PointData/DataArray[@Name='Normals']"), sep=" "
    ).reshape(-1, 3)
    np.testing.assert_array_equal(left_points, right_points * [1, 1, -1])
    np.testing.assert_array_equal(left_faces, right_faces[:, [0, 2, 1]])
    np.testing.assert_array_equal(left_normals[:, 2], [-1, -1, -1, -1])


def test_reviewed_profile_rejects_a_different_exact_source(
    manifest: ShoulderLaminaManifest,
) -> None:
    with pytest.raises(ValueError, match="reviewed 557 source"):
        admit_reviewed_557_manifest(manifest)


def test_nontriangle_polygons_reject_even_with_updated_mesh_digest(
    manifest: ShoulderLaminaManifest, tmp_path: Path
) -> None:
    mesh = manifest.clavicle_mesh_path
    raw = mesh.read_text(encoding="utf-8").replace(
        'Name="offsets" format="ascii">3 6 9 12',
        'Name="offsets" format="ascii">3 7 10 13',
    )
    mesh.write_text(raw, encoding="utf-8")
    changed = replace(manifest, clavicle_mesh_sha256=_sha(mesh))
    with pytest.raises(ValueError, match="complete triangles"):
        build_generic_shoulder_lamina(changed, tmp_path / "candidate")


def test_exact_reviewed_557_native_physical_and_kinematic_boundary(
    tmp_path: Path,
) -> None:
    require_opensim()
    root = os.environ.get("GENERIC_SHOULDER_FIXTURE_ROOT")
    if root is None:
        pytest.skip("exact source/mesh fixture is outside this repository")
    planning = Path(root)
    manifest = ShoulderLaminaManifest(
        planning / "composed_reduction_12161_v2/03-custom-mtp.osim",
        REVIEWED_557_SOURCE_SHA256,
        planning / "generic_shoulder_mesh_triage_v1/clavicle.vtp",
        REVIEWED_RIGHT_MESH_SHA256["clavicle"],
        planning / "generic_shoulder_mesh_triage_v1/scapula.vtp",
        REVIEWED_RIGHT_MESH_SHA256["scapula"],
        REGISTRATION,
    )
    candidate = build_reviewed_557_shoulder_lamina(manifest, tmp_path / "candidate")
    audit = audit_reviewed_557_shoulder_candidate(
        manifest, candidate, (("pelvis_tilt", 0.0), ("SC_x", 0.01), ("SC_x", -0.01))
    )
    assert audit.body_count == 40
    assert audit.pose_count == 3
    assert audit.max_transform_error == 0
    assert audit.max_constraint_error_difference == 0
    left = candidate.model_path.parent / "Geometry/generic_scapula_l.vtp"
    left.write_bytes(left.read_bytes() + b" ")
    with pytest.raises(ValueError, match="derived geometry hash"):
        audit_reviewed_557_shoulder_candidate(
            manifest, candidate, (("pelvis_tilt", 0.0),)
        )
