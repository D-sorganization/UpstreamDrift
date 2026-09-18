"""Unit tests and RED failure fixtures for OpenSim anatomical baseline audit (OG-01, #10395).

Asserts:
1. Frozen baseline model hash and structural counts (23 bodies, 39 coordinates, 0 muscles).
2. Detection of empty Club attached_geometry (reproducing the missing club defect).
3. Detection of inconsistent segment scaling where joint frames are scaled but arm meshes remain unit scale.
4. Typed failure reasons for missing files, invalid models, or broken geometry contracts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.tour_matching.model_audit import (
    BASELINE_MODEL_SHA256,
    BASELINE_OS3B_INPUT_SHA256,
    audit_model_geometry,
    verify_model_qualification,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_MODEL_PATH = (
    REPO_ROOT
    / "docs"
    / "development"
    / "opensim_tour_matching"
    / "evidence"
    / "os7_moco_g1"
    / "golf_humanoid_scaled_tour_markers_moco.osim"
)
INPUT_MODEL_PATH = (
    REPO_ROOT
    / "docs"
    / "development"
    / "opensim_tour_matching"
    / "evidence"
    / "os7_moco_g1"
    / "inputs"
    / "golf_humanoid_scaled_tour_markers.osim"
)


def test_baseline_model_files_exist_and_match_hashes() -> None:
    """Verify that both pinned baseline models resolve and match their declared SHA-256 hashes."""
    assert BASELINE_MODEL_PATH.is_file(), (
        f"Missing baseline model at {BASELINE_MODEL_PATH}"
    )
    assert INPUT_MODEL_PATH.is_file(), f"Missing input model at {INPUT_MODEL_PATH}"

    res_baseline = audit_model_geometry(BASELINE_MODEL_PATH)
    assert res_baseline.sha256 == BASELINE_MODEL_SHA256

    res_input = audit_model_geometry(INPUT_MODEL_PATH)
    assert res_input.sha256 == BASELINE_OS3B_INPUT_SHA256


def test_audit_baseline_model_structure() -> None:
    """Audit baseline model structural counts: 23 bodies, 39 coordinates, 39 actuators, 0 muscles."""
    res = audit_model_geometry(BASELINE_MODEL_PATH)
    assert res.num_bodies == 23
    assert res.num_coordinates == 39
    assert res.num_actuators == 39
    assert res.num_muscles == 0
    assert "Club" in res.body_names


def test_audit_reproduces_empty_club_attached_geometry() -> None:
    """RED defect reproduction: baseline model has 0 attached geometry on Club body."""
    res = audit_model_geometry(BASELINE_MODEL_PATH)
    assert res.club_attached_geometry_count == 0
    assert not res.has_visible_club


def test_audit_reproduces_unscaled_arm_meshes_defect() -> None:
    """RED defect reproduction: humerus_r and humerus_l arm meshes remain at scale 1 1 1."""
    res = audit_model_geometry(BASELINE_MODEL_PATH)
    assert "humerus_r" in res.arm_mesh_scales
    assert "humerus_l" in res.arm_mesh_scales

    # Both humerus bone meshes are unscaled despite non-unit joint frame scaling
    assert res.arm_mesh_scales["humerus_r"] == [("humerus_rv.vtp", (1.0, 1.0, 1.0))]
    assert res.arm_mesh_scales["humerus_l"] == [("humerus_lv.vtp", (1.0, 1.0, 1.0))]
    assert res.has_unscaled_arm_mesh_defect


def test_qualification_fails_on_historical_baseline_fixture() -> None:
    """Qualification gate MUST fail on historical baseline because of missing club and unscaled meshes."""
    with pytest.raises(ValueError, match="Club body has no attached visual geometry"):
        verify_model_qualification(BASELINE_MODEL_PATH, require_visible_club=True)


def test_audit_fails_closed_on_missing_file() -> None:
    """Audit fails with FileNotFoundError when model path does not exist."""
    missing_path = REPO_ROOT / "docs" / "nonexistent_model.osim"
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        audit_model_geometry(missing_path)


def test_audit_synthetic_valid_model_passes_qualification(tmp_path: Path) -> None:
    """Synthetic minimal valid OpenSim XML with attached club geometry passes qualification."""
    model_xml = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40600">
    <Model name="synthetic_qualified_golf_model">
        <BodySet name="bodyset">
            <objects>
                <Body name="pelvis">
                    <attached_geometry>
                        <Mesh name="pelvis_geom">
                            <scale_factors>1 1 1</scale_factors>
                            <mesh_file>pelvis.vtp</mesh_file>
                        </Mesh>
                    </attached_geometry>
                </Body>
                <Body name="humerus_r">
                    <attached_geometry>
                        <Mesh name="humerus_r_geom">
                            <scale_factors>1.4567 1.4567 1.4567</scale_factors>
                            <mesh_file>humerus_rv.vtp</mesh_file>
                        </Mesh>
                    </attached_geometry>
                </Body>
                <Body name="Club">
                    <mass>0.32</mass>
                    <attached_geometry>
                        <Cylinder name="shaft_geom">
                            <radius>0.006</radius>
                            <length>1.0</length>
                        </Cylinder>
                    </attached_geometry>
                </Body>
            </objects>
        </BodySet>
        <JointSet name="jointset">
            <objects />
        </JointSet>
        <ForceSet name="forceset">
            <objects />
        </ForceSet>
    </Model>
</OpenSimDocument>
"""
    osim_file = tmp_path / "valid_model.osim"
    osim_file.write_text(model_xml, encoding="utf-8")

    res = audit_model_geometry(osim_file)
    assert res.num_bodies == 3
    assert res.club_attached_geometry_count == 1
    assert res.has_visible_club
    assert not res.has_unscaled_arm_mesh_defect

    # Verification should succeed with require_visible_club=True
    verify_model_qualification(osim_file, require_visible_club=True)
