"""Unit tests for anatomical visual assets and skin resolution (MV-02, #10478)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.body_part_viz.anatomical_visuals import (
    VisualAssetBinding,
    VisualSkinMode,
    get_diagnostic_fallback_binding,
    resolve_anatomical_visual,
)

pytestmark = [pytest.mark.unit]


def test_visual_skin_mode_values() -> None:
    """Supported skin modes must match the MV-02 contract."""
    assert VisualSkinMode.INERTIA_ELLIPSOIDS.value == "inertia_ellipsoids"
    assert VisualSkinMode.ANATOMICAL_MESH.value == "anatomical_mesh"
    assert VisualSkinMode.ANATOMICAL_CAPSULE.value == "anatomical_capsule"


def test_resolve_anatomical_visual_head() -> None:
    """Head body must resolve to bundled CC0-1.0 head mesh with valid scale."""
    binding = resolve_anatomical_visual(
        "GolfSwing3D_Kinetic/Head", "GolfSwing3D_Kinetic/Head/head"
    )
    assert binding is not None
    assert binding.semantic_body == "head"
    assert binding.mesh_relative_path.endswith("head.stl")
    assert not Path(binding.mesh_relative_path).is_absolute()
    assert binding.license_name == "CC0-1.0"
    assert binding.units == "meters"
    assert len(binding.scale) == 3
    assert all(math.isfinite(s) and s > 0 for s in binding.scale)
    assert binding.is_fallback is False


def test_resolve_anatomical_visual_legs_and_feet() -> None:
    """Thigh, shin, and foot bodies must resolve to appropriate anatomical shapes."""
    thigh = resolve_anatomical_visual("femur_r", "femur_r")
    assert thigh is not None
    assert thigh.semantic_body == "thigh"
    assert thigh.mesh_relative_path.endswith("thigh.stl")

    shin = resolve_anatomical_visual("tibia_l", "tibia_l")
    assert shin is not None
    assert shin.semantic_body == "shin"
    assert shin.mesh_relative_path.endswith("shin.stl")

    foot = resolve_anatomical_visual("calcn_r", "calcn_r")
    assert foot is not None
    assert foot.semantic_body == "foot"
    assert foot.mesh_relative_path.endswith("foot.stl")


def test_resolve_anatomical_visual_club() -> None:
    """Club components must resolve to recognizable club visual primitives."""
    clubhead = resolve_anatomical_visual(
        "solid_reference:GolfSwing3D_Kinetic/Club/Clubface Vector",
        "solid_reference:GolfSwing3D_Kinetic/Club/Clubface Vector/Clubhead",
    )
    assert clubhead is not None
    assert clubhead.semantic_body == "clubhead"
    assert clubhead.license_name == "CC0-1.0"

    shaft = resolve_anatomical_visual(
        "solid_reference:GolfSwing3D_Kinetic/Club/Clubface Vector",
        "solid_reference:GolfSwing3D_Kinetic/Club/Clubface Vector/Rigid Shaft",
    )
    assert shaft is not None
    assert shaft.semantic_body == "shaft"


def test_missing_mesh_diagnostic_fallback(tmp_path: Path) -> None:
    """When a mesh file is missing, diagnostic fallback geometry must be returned."""
    fallback = get_diagnostic_fallback_binding("missing_link", "missing.stl")
    assert fallback.is_fallback is True
    assert fallback.semantic_body == "missing_link"
    # Diagnostic fallback uses high-visibility magenta color
    assert fallback.color == (1.0, 0.0, 1.0, 1.0)
    assert fallback.fallback_geometry_type in ("box", "capsule", "sphere")


def test_all_bindings_use_relative_paths() -> None:
    """Visual asset bindings must never contain original local machine absolute paths."""
    bodies = [
        ("GolfSwing3D_Kinetic/Head", "GolfSwing3D_Kinetic/Head/head"),
        (
            "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/COMRod",
            "middle_trunk",
        ),
        ("solid_reference:GolfSwing3D_Kinetic/LUpperArm", "upper_arm"),
        (
            "solid_reference:GolfSwing3D_Kinetic/Left Forearm/LLowerForearm",
            "forearm_distal",
        ),
        ("femur_r", "femur_r"),
        ("tibia_r", "tibia_r"),
        ("calcn_r", "calcn_r"),
    ]
    for b_name, s_name in bodies:
        binding = resolve_anatomical_visual(b_name, s_name)
        assert ":" not in binding.mesh_relative_path


def test_export_model_bundle_with_anatomical_visuals(tmp_path: Path) -> None:
    """Exporting model bundle with visuals must produce relative URDF mesh elements and bundle assets."""
    import defusedxml.ElementTree as ET
    from src.shared.python.model_generation.export.model_bundle import (
        export_model_bundle,
    )

    repo_root = Path(__file__).resolve().parents[3]
    spec_path = (
        repo_root
        / "docs/development/full_body_models/evidence/ground_support/anthro_driver/full_body_spec_hipcal_scaled.json"
    )
    if not spec_path.is_file():
        pytest.skip("Full body spec not found")

    spec_bytes = spec_path.read_bytes()
    bundle = export_model_bundle(spec_bytes, include_visuals=True)

    # URDF must parse securely with defusedxml and contain visual elements
    root = ET.fromstring(bundle.urdf_xml)
    visuals = root.findall(".//visual")
    assert len(visuals) > 0

    # Every visual mesh must use relative paths only
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        assert filename.startswith("meshes/")
        assert ".." not in filename
        assert ":" not in filename
        assert not filename.startswith("/")

    # Bundle archive must save and contain mesh files
    archive_path = tmp_path / "visual_bundle.zip"
    bundle.save_archive(archive_path)
    assert archive_path.is_file()
    assert archive_path.stat().st_size > 0
