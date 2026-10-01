"""Unit tests for anatomical mesh substitution without changing qualified physics (MMR-05 #11089).

Verifies:
1. One-segment File Solid / mesh substitution behind optional visual preset preserves
   exact joint frames, mass, COM, inertia, and same-state forward kinematics (marker
   positions invariant to visual skin changes).
2. Wrong mesh units, non-finite mesh bounds, or missing asset rejects or uses documented graceful fallback.
3. Redistribution terms and asset provenance tracked.
4. Retains compiled count within MMR-04 budget.
5. Reusable segment adapter for pelvis/trunk/head/hand/shoe prototypes with handedness,
   key swing poses (address, top, impact, follow-through) marker overlay, and clipping checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.anatomical_meshes import (
    AnatomicalMeshError,
    AnatomicalSegmentAdapter,
    BoundingBox3D,
    BudgetCheckResult,
    InvalidMeshUnitsError,
    MeshAssetProvenance,
    MeshMetadata,
    MissingMeshAssetError,
    NonFiniteMeshBoundsError,
    SegmentMeshConfig,
    SimscapeVisualStrategy,
    VisualFallbackStatus,
    VisualPreset,
    apply_mesh_substitution,
    check_simscape_mesh_block_budget,
    load_bundled_human_mesh_metadata,
)
from src.tools.tour_matching_viewer.core import body_poses_from_state

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
BUNDLED_MESHES_DIR = (
    ROOT
    / "src/tools/model_explorer/bundled_assets/human_models/human_subject_with_meshes/meshes"
)
BUNDLED_METADATA_PATH = (
    ROOT
    / "src/tools/model_explorer/bundled_assets/human_models/human_subject_with_meshes/metadata.json"
)


def _t(translation: list[float] | tuple[float, float, float]) -> list[list[float]]:
    m = np.eye(4)
    m[:3, 3] = translation
    return m.tolist()


def _make_test_humanoid_spec() -> dict[str, Any]:
    """Create a minimal humanoid kinematic tree containing pelvis, trunk, head, hand, and foot."""
    return {
        "schema_version": "full-body-v1",
        "gravity_m_s2": [0.0, 0.0, -9.81],
        "coordinate_order": [
            "root_px",
            "root_py",
            "root_pz",
            "root_rx",
            "root_ry",
            "root_rz",
            "trunk_rx",
            "trunk_ry",
            "trunk_rz",
            "neck_rx",
            "neck_ry",
            "neck_rz",
            "shoulder_l_rx",
            "shoulder_l_ry",
            "shoulder_l_rz",
            "hip_l_rx",
            "hip_l_ry",
            "hip_l_rz",
        ],
        "bodies": [
            {"name": "world", "solids": []},
            {
                "name": "pelvis",
                "solids": [
                    {
                        "name": "pelvis/solid",
                        "mass_kg": 11.5,
                        "com_m": [0.0, 0.0, 0.05],
                        "inertia_com_kg_m2": (np.diag([0.15, 0.12, 0.14])).tolist(),
                        "placement": np.eye(4).tolist(),
                    }
                ],
            },
            {
                "name": "trunk",
                "solids": [
                    {
                        "name": "trunk/solid",
                        "mass_kg": 28.0,
                        "com_m": [0.0, 0.0, 0.25],
                        "inertia_com_kg_m2": (np.diag([0.85, 0.70, 0.60])).tolist(),
                        "placement": np.eye(4).tolist(),
                    }
                ],
            },
            {
                "name": "head",
                "solids": [
                    {
                        "name": "head/solid",
                        "mass_kg": 4.8,
                        "com_m": [0.0, 0.0, 0.10],
                        "inertia_com_kg_m2": (np.diag([0.026, 0.026, 0.019])).tolist(),
                        "placement": np.eye(4).tolist(),
                    }
                ],
            },
            {
                "name": "hand_l",
                "solids": [
                    {
                        "name": "hand_l/solid",
                        "mass_kg": 0.5,
                        "com_m": [0.0, 0.0, -0.08],
                        "inertia_com_kg_m2": (
                            np.diag([0.0012, 0.0012, 0.0006])
                        ).tolist(),
                        "placement": np.eye(4).tolist(),
                    }
                ],
            },
            {
                "name": "foot_l",
                "solids": [
                    {
                        "name": "foot_l/solid",
                        "mass_kg": 1.2,
                        "com_m": [0.06, 0.0, -0.04],
                        "inertia_com_kg_m2": (np.diag([0.003, 0.007, 0.008])).tolist(),
                        "placement": np.eye(4).tolist(),
                    }
                ],
            },
        ],
        "joints": [
            {
                "name": "root_joint",
                "parent": "world",
                "child": "pelvis",
                "parent_to_base": _t([0.0, 0.0, 0.95]),
                "child_to_follower": np.eye(4).tolist(),
                "primitives": [
                    {"primitive": "Px", "coordinate": "root_px"},
                    {"primitive": "Py", "coordinate": "root_py"},
                    {"primitive": "Pz", "coordinate": "root_pz"},
                    {"primitive": "Rx", "coordinate": "root_rx"},
                    {"primitive": "Ry", "coordinate": "root_ry"},
                    {"primitive": "Rz", "coordinate": "root_rz"},
                ],
            },
            {
                "name": "lumbar_joint",
                "parent": "pelvis",
                "child": "trunk",
                "parent_to_base": _t([0.0, 0.0, 0.12]),
                "child_to_follower": np.eye(4).tolist(),
                "primitives": [
                    {"primitive": "Rx", "coordinate": "trunk_rx"},
                    {"primitive": "Ry", "coordinate": "trunk_ry"},
                    {"primitive": "Rz", "coordinate": "trunk_rz"},
                ],
            },
            {
                "name": "cervical_joint",
                "parent": "trunk",
                "child": "head",
                "parent_to_base": _t([0.0, 0.0, 0.48]),
                "child_to_follower": np.eye(4).tolist(),
                "primitives": [
                    {"primitive": "Rx", "coordinate": "neck_rx"},
                    {"primitive": "Ry", "coordinate": "neck_ry"},
                    {"primitive": "Rz", "coordinate": "neck_rz"},
                ],
            },
            {
                "name": "shoulder_l_joint",
                "parent": "trunk",
                "child": "hand_l",
                "parent_to_base": _t([0.0, 0.18, 0.42]),
                "child_to_follower": np.eye(4).tolist(),
                "primitives": [
                    {"primitive": "Rx", "coordinate": "shoulder_l_rx"},
                    {"primitive": "Ry", "coordinate": "shoulder_l_ry"},
                    {"primitive": "Rz", "coordinate": "shoulder_l_rz"},
                ],
            },
            {
                "name": "hip_l_joint",
                "parent": "pelvis",
                "child": "foot_l",
                "parent_to_base": _t([0.0, 0.09, -0.05]),
                "child_to_follower": np.eye(4).tolist(),
                "primitives": [
                    {"primitive": "Rx", "coordinate": "hip_l_rx"},
                    {"primitive": "Ry", "coordinate": "hip_l_ry"},
                    {"primitive": "Rz", "coordinate": "hip_l_rz"},
                ],
            },
        ],
        "frames": [
            {
                "name": "PELVIS_MARKER",
                "body": "pelvis",
                "placement": _t([0.08, 0.0, 0.05]),
            },
            {"name": "C7_MARKER", "body": "trunk", "placement": _t([-0.05, 0.0, 0.45])},
            {
                "name": "HEAD_TOP_MARKER",
                "body": "head",
                "placement": _t([0.0, 0.0, 0.22]),
            },
            {
                "name": "HAND_L_MARKER",
                "body": "hand_l",
                "placement": _t([0.02, 0.0, -0.15]),
            },
            {
                "name": "TOE_L_MARKER",
                "body": "foot_l",
                "placement": _t([0.16, 0.0, -0.05]),
            },
        ],
        "closure": {},
        "upper_body_counts": {
            "bodies": 6,
            "joints": 5,
            "coordinates": 18,
        },
        "upper_body_schema_version": "native-geometry-v1",
        "upper_body_qualification": "test",
        "visual_hints": {
            "shapes": {
                "pelvis/solid": {
                    "shape": "ellipsoid",
                    "half_size_m": [0.12, 0.17, 0.10],
                    "center_m": [0.0, 0.0, 0.05],
                },
                "head/solid": {
                    "shape": "ellipsoid",
                    "half_size_m": [0.09, 0.08, 0.11],
                    "center_m": [0.0, 0.0, 0.10],
                },
            }
        },
    }


def _create_sample_metadata(
    mesh_path: Path,
    mesh_name: str = "Head.stl",
    units: str = "m",
    bbox_min: tuple[float, float, float] = (-0.1, -0.1, -0.1),
    bbox_max: tuple[float, float, float] = (0.1, 0.1, 0.1),
) -> MeshMetadata:
    provenance = MeshAssetProvenance(
        asset_name=mesh_name,
        source_repository="https://github.com/gbionics/human-gazebo",
        source_commit="master",
        license_type="CC-BY-SA 2.0",
        license_url="https://creativecommons.org/licenses/by-sa/2.0/",
        attribution="gbionics/human-gazebo",
        redistribution_allowed=True,
    )
    return MeshMetadata(
        file_path=mesh_path,
        file_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        format="STL",
        units=units,
        scale_to_meters=1.0 if units == "m" else 0.001,
        bounding_box=BoundingBox3D(min_point=bbox_min, max_point=bbox_max),
        vertex_count=1000,
        face_count=350,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Test 1: Invariance of Mass, COM, Inertia, Joint Frames, and FK Markers
# ---------------------------------------------------------------------------


def test_one_segment_file_solid_preserves_exact_physics_and_fk_markers(
    tmp_path: Path,
) -> None:
    """Replacing a segment's visual with a File Solid must not alter physics or marker FK."""
    spec = _make_test_humanoid_spec()
    dummy_mesh = tmp_path / "Head.stl"
    dummy_mesh.write_bytes(b"solid Head\nendsolid Head\n")

    meta = _create_sample_metadata(dummy_mesh, "Head.stl")
    config = SegmentMeshConfig(
        segment_name="head",
        mesh_metadata=meta,
        visual_strategy=SimscapeVisualStrategy.FILE_SOLID_SUBSTITUTION,
    )

    # 1. Apply under visual preset ANATOMICAL_MESH
    substituted_spec = apply_mesh_substitution(
        spec, [config], preset=VisualPreset.ANATOMICAL_MESH
    )

    # Physical properties check
    original_head_body = [b for b in spec["bodies"] if b["name"] == "head"][0]
    sub_head_body = [b for b in substituted_spec["bodies"] if b["name"] == "head"][0]

    assert (
        original_head_body["solids"][0]["mass_kg"]
        == sub_head_body["solids"][0]["mass_kg"]
    )
    np.testing.assert_allclose(
        original_head_body["solids"][0]["com_m"],
        sub_head_body["solids"][0]["com_m"],
        atol=1e-15,
    )
    np.testing.assert_allclose(
        original_head_body["solids"][0]["inertia_com_kg_m2"],
        sub_head_body["solids"][0]["inertia_com_kg_m2"],
        atol=1e-15,
    )
    np.testing.assert_allclose(
        original_head_body["solids"][0]["placement"],
        sub_head_body["solids"][0]["placement"],
        atol=1e-15,
    )

    # Joint frames invariant
    orig_joints = {j["name"]: j for j in spec["joints"]}
    sub_joints = {j["name"]: j for j in substituted_spec["joints"]}
    for j_name, orig_j in orig_joints.items():
        np.testing.assert_allclose(
            orig_j["parent_to_base"], sub_joints[j_name]["parent_to_base"], atol=1e-15
        )
        np.testing.assert_allclose(
            orig_j["child_to_follower"],
            sub_joints[j_name]["child_to_follower"],
            atol=1e-15,
        )

    # Forward Kinematics markers invariant across diverse swing poses
    swing_poses = {
        "address": np.zeros(len(spec["coordinate_order"])),
        "top": np.array(
            [
                0.0,
                0.0,
                0.95,
                0.1,
                0.2,
                0.3,
                0.2,
                -0.4,
                0.5,
                -0.1,
                0.2,
                -0.3,
                0.5,
                0.6,
                0.7,
                -0.2,
                0.1,
                0.0,
            ]
        ),
        "impact": np.array(
            [
                0.05,
                0.0,
                0.93,
                0.05,
                0.1,
                -0.2,
                0.4,
                0.1,
                -0.6,
                0.1,
                -0.1,
                0.4,
                0.2,
                0.1,
                -0.3,
                0.1,
                -0.2,
                0.3,
            ]
        ),
        "follow_through": np.array(
            [
                0.1,
                0.05,
                0.96,
                -0.1,
                -0.2,
                -0.7,
                -0.3,
                0.5,
                -0.8,
                0.2,
                -0.3,
                0.6,
                -0.4,
                -0.5,
                -0.6,
                0.4,
                0.2,
                -0.4,
            ]
        ),
    }

    for pose_name, q in swing_poses.items():
        poses_orig = body_poses_from_state(spec, q)
        poses_sub = body_poses_from_state(substituted_spec, q)

        for frame in spec["frames"]:
            b_name = frame["body"]
            f_placement = np.array(frame["placement"])
            # Compute actual 3D marker position
            world_pos_orig = (poses_orig[b_name] @ f_placement)[:3, 3]
            world_pos_sub = (poses_sub[b_name] @ f_placement)[:3, 3]

            np.testing.assert_allclose(
                world_pos_orig,
                world_pos_sub,
                atol=1e-12,
                err_msg=f"Marker {frame['name']} shifted at pose {pose_name}",
            )


# ---------------------------------------------------------------------------
# Test 2: Validation of Mesh Units & Graceful Fallback
# ---------------------------------------------------------------------------


def test_wrong_mesh_units_rejects_or_falls_back(tmp_path: Path) -> None:
    """Invalid mesh units must raise InvalidMeshUnitsError or gracefully fall back to primitive."""
    dummy_mesh = tmp_path / "Pelvis.stl"
    dummy_mesh.write_bytes(b"solid Pelvis\nendsolid Pelvis\n")

    # Fail closed on unknown/non-SI units when fallback is disabled
    with pytest.raises(InvalidMeshUnitsError, match="Unsupported mesh units"):
        _create_sample_metadata(dummy_mesh, units="inches")

    # With documented graceful fallback
    meta_wrong_units = MeshMetadata(
        file_path=dummy_mesh,
        file_sha256="test_sha",
        format="STL",
        units="cubits",  # invalid
        scale_to_meters=-1.0,  # invalid negative
        bounding_box=BoundingBox3D((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)),
        vertex_count=500,
        face_count=200,
        provenance=MeshAssetProvenance(
            "Pelvis.stl", "repo", "main", "CC-BY-SA 2.0", "url", "attr", True
        ),
        validate=False,
    )
    config = SegmentMeshConfig(
        segment_name="pelvis",
        mesh_metadata=meta_wrong_units,
        fallback_shape={"shape": "ellipsoid", "half_size_m": [0.15, 0.15, 0.1]},
    )

    spec = _make_test_humanoid_spec()
    # Reject when fallback_on_error=False
    with pytest.raises(InvalidMeshUnitsError):
        apply_mesh_substitution(spec, [config], allow_fallback=False)

    # Graceful fallback when allow_fallback=True
    fallback_spec = apply_mesh_substitution(spec, [config], allow_fallback=True)
    hints = fallback_spec.get("visual_hints", {}).get("shapes", {})
    assert "pelvis/solid" in hints
    assert hints["pelvis/solid"]["shape"] == "ellipsoid"
    assert fallback_spec.get("visual_hints", {}).get("mesh_fallbacks", {}).get(
        "pelvis"
    ) == (VisualFallbackStatus.FALLBACK_TO_PRIMITIVE.value)


# ---------------------------------------------------------------------------
# Test 3: Validation of Non-Finite Mesh Bounds & Graceful Fallback
# ---------------------------------------------------------------------------


def test_non_finite_mesh_bounds_rejects_or_falls_back(tmp_path: Path) -> None:
    """Mesh with NaN, Inf, or inverted bounds must raise NonFiniteMeshBoundsError."""
    dummy_mesh = tmp_path / "trunk.stl"
    dummy_mesh.write_bytes(b"solid Trunk\nendsolid Trunk\n")

    # NaN bounds
    with pytest.raises(NonFiniteMeshBoundsError, match="finite"):
        BoundingBox3D(min_point=(0.0, np.nan, -0.2), max_point=(0.2, 0.2, 0.2))

    # Inverted bounds (min >= max)
    with pytest.raises(
        NonFiniteMeshBoundsError, match="min_point must be strictly less than max_point"
    ):
        BoundingBox3D(min_point=(0.5, 0.5, 0.5), max_point=(0.2, 0.2, 0.2))


# ---------------------------------------------------------------------------
# Test 4: Missing Asset Rejection and Graceful Fallback
# ---------------------------------------------------------------------------


def test_missing_asset_rejects_or_falls_back() -> None:
    """Missing file at path must raise MissingMeshAssetError or apply fallback."""
    missing_file = Path("non/existent/path/Foot.stl")
    meta = _create_sample_metadata(missing_file)
    config = SegmentMeshConfig(
        segment_name="foot_l",
        mesh_metadata=meta,
        fallback_shape={"shape": "box", "half_size_m": [0.08, 0.04, 0.03]},
    )
    spec = _make_test_humanoid_spec()

    with pytest.raises(MissingMeshAssetError, match="Mesh asset file not found"):
        apply_mesh_substitution(spec, [config], allow_fallback=False)

    # Documented fallback
    fallback_spec = apply_mesh_substitution(spec, [config], allow_fallback=True)
    assert fallback_spec["visual_hints"]["mesh_fallbacks"]["foot_l"] == (
        VisualFallbackStatus.FALLBACK_TO_PRIMITIVE.value
    )


# ---------------------------------------------------------------------------
# Test 5: Redistribution Terms and Asset Provenance Tracking
# ---------------------------------------------------------------------------


def test_redistribution_terms_and_asset_provenance() -> None:
    """Asset provenance must track source repository, commit, license, and redistribution rights."""
    prov = MeshAssetProvenance(
        asset_name="Pelvis.stl",
        source_repository="https://github.com/gbionics/human-gazebo",
        source_commit="master",
        license_type="CC-BY-SA 2.0",
        license_url="https://creativecommons.org/licenses/by-sa/2.0/",
        attribution="gbionics/human-gazebo",
        redistribution_allowed=True,
    )
    assert prov.is_valid_for_redistribution()
    assert prov.attribution == "gbionics/human-gazebo"

    # Non-redistributable or unfree license rejected
    unfree_prov = MeshAssetProvenance(
        asset_name="Proprietary.stl",
        source_repository="internal",
        source_commit="1",
        license_type="Proprietary-All-Rights-Reserved",
        license_url="",
        attribution="Acme Corp",
        redistribution_allowed=False,
    )
    assert not unfree_prov.is_valid_for_redistribution()

    # Load actual bundled human mesh metadata from repository
    if BUNDLED_METADATA_PATH.exists():
        bundled_meta = load_bundled_human_mesh_metadata(BUNDLED_METADATA_PATH)
        assert bundled_meta.provenance.license_type == "CC-BY-SA 2.0"
        assert bundled_meta.provenance.redistribution_allowed is True
        assert bundled_meta.mesh_count == 51


# ---------------------------------------------------------------------------
# Test 6: Retains Compiled Block Count Within MMR-04 Budget
# ---------------------------------------------------------------------------


def test_compiled_count_retained_within_mmr04_budget() -> None:
    """File Solid substitution preserves compiled count (delta=0), while extra solids check <=975 ceiling."""
    # Production ceiling: <= 975 compiled nonvirtual blocks (with 25 reserve to 1000 license limit)
    # File Solid substitution swaps visual geometry inside existing solid: delta = 0
    res_sub = check_simscape_mesh_block_budget(
        current_compiled_blocks=965,  # GS3DX_Human baseline
        strategy=SimscapeVisualStrategy.FILE_SOLID_SUBSTITUTION,
        num_segments=5,
    )
    assert res_sub.passed
    assert res_sub.compiled_delta == 0
    assert res_sub.new_compiled_total == 965
    assert res_sub.headroom == 975 - 965  # 10 blocks headroom to production ceiling

    # Adding external visual solids compiles to 7 blocks per solid (measured in SHAPE.md: 973 -> 980)
    # Adding 1 external solid to 970 blocks: 970 + 7 = 977 > 975 -> Exceeds production ceiling!
    res_ext = check_simscape_mesh_block_budget(
        current_compiled_blocks=970,
        strategy=SimscapeVisualStrategy.EXTERNAL_VISUAL_SOLID,
        num_segments=1,
    )
    assert not res_ext.passed
    assert res_ext.new_compiled_total == 977
    assert "exceeds production budget ceiling of 975" in res_ext.diagnostic

    # External render skin has 0 Simulink blocks added
    res_skin = check_simscape_mesh_block_budget(
        current_compiled_blocks=965,
        strategy=SimscapeVisualStrategy.EXTERNAL_RENDER_SKIN,
        num_segments=5,
    )
    assert res_skin.passed
    assert res_skin.new_compiled_total == 965


# ---------------------------------------------------------------------------
# Test 7: Reusable Segment Adapter for Pelvis, Trunk, Head, Hand, Shoe Prototypes
# ---------------------------------------------------------------------------


def test_reusable_segment_adapter_prototypes(tmp_path: Path) -> None:
    """Segment adapter handles pelvis, trunk, head, hand, shoe with handedness and swing pose verification."""
    spec = _make_test_humanoid_spec()
    adapter = AnatomicalSegmentAdapter(spec)

    # Verify prototype catalog covers the required 5 anatomical regions
    assert set(adapter.supported_prototypes()) == {
        "pelvis",
        "trunk",
        "head",
        "hand",
        "shoe",
    }

    # Attach prototypes with synthetic STLs
    prototypes = ["pelvis", "trunk", "head", "hand_l", "shoe_l"]
    for proto in prototypes:
        mesh_file = tmp_path / f"{proto}.stl"
        mesh_file.write_bytes(f"solid {proto}\nendsolid {proto}\n".encode())
        meta = _create_sample_metadata(mesh_file, f"{proto}.stl")
        adapter.configure_segment(proto, meta)

    # Test left/right handedness mirroring on hand and shoe
    hand_r_meta = adapter.mirror_bilateral_metadata(
        adapter.get_segment_config("hand_l").mesh_metadata, lateral_axis="y"
    )
    assert hand_r_meta.bounding_box.min_point[1] == pytest.approx(
        -adapter.get_segment_config("hand_l").mesh_metadata.bounding_box.max_point[1]
    )

    # Verify marker overlay invariance at address, top, impact, follow-through
    poses = {
        "address": np.zeros(len(spec["coordinate_order"])),
        "top": np.ones(len(spec["coordinate_order"])) * 0.15,
        "impact": np.ones(len(spec["coordinate_order"])) * -0.1,
        "follow_through": np.ones(len(spec["coordinate_order"])) * 0.25,
    }
    for pose_name, q in poses.items():
        residuals = adapter.verify_marker_overlay_invariance(q)
        for marker_name, err in residuals.items():
            assert err < 1e-12, f"Residual {err} on {marker_name} at {pose_name}"

    # Verify UI clipping / self-intersection check runs without crashing and reports clearance
    clipping_report = adapter.check_segment_clearance(poses["address"])
    assert "trunk_vs_pelvis" in clipping_report
    assert clipping_report["trunk_vs_pelvis"].clearance_m is not None
