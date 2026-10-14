"""Unit tests for unified specification-exported URDF contract (#11570).

Validates:
1. Canonical golfer URDF XML and sidecar metadata bundle generation from full-body spec.
2. Cross-engine parity across URDF, MJCF, and Pinocchio depth-first joint ordering:
   - Identical total mass conservation (< 1e-9 kg).
   - Coordinate names matching specification exactly.
   - Neutral-pose frame origins within 1 mm (< 1e-3 m).
3. Ledger of MyoSuite topology divergence (MyoHub 11/44 coordinates vs 41/44 URDF).
4. Retired legacy URDF paths tracking divergent golfer.urdf copies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_matching.validators import (
    MYOSUITE_TOPOLOGY_DIVERGENCE_LEDGER,
    RETIRED_LEGACY_URDF_PATHS,
    get_canonical_golfer_urdf_bundle,
    verify_spec_exported_urdf_engine_parity,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO_ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


def test_myosuite_topology_divergence_ledger() -> None:
    """Validate MyoSuite topology divergence ledger contents and semantics."""
    ledger = MYOSUITE_TOPOLOGY_DIVERGENCE_LEDGER
    assert isinstance(ledger, dict)
    assert ledger["engine"] == "myosuite"
    assert ledger["tier_musculoskeletal"] == "tier_3_musculoskeletal"
    assert ledger["tier_rigid_body"] == "tier_1_rigid_body"
    assert ledger["divergence_type"] == "topology_divergence"
    assert ledger["myohub_coordinates_mapped"] == 11
    assert ledger["total_spec_coordinates"] == 44
    assert ledger["spec_exported_urdf_coordinates"] == 41
    assert "11 of 44 coordinates" in ledger["description"]
    assert "tier_3_musculoskeletal" in ledger["description"]
    assert "tier_1_rigid_body" in ledger["description"]
    assert "ledger_reference" in ledger


def test_retired_legacy_urdf_paths() -> None:
    """Validate tuple of retired divergent golfer.urdf paths."""
    expected = (
        "src/engines/physics_engines/pinocchio/models/generated/golfer.urdf",
        "src/engines/physics_engines/pinocchio/models/generated/golfer_ik.urdf",
        "src/engines/physics_engines/drake/models/generated/golfer.urdf",
    )
    assert isinstance(RETIRED_LEGACY_URDF_PATHS, tuple)
    assert expected == RETIRED_LEGACY_URDF_PATHS


def test_get_canonical_golfer_urdf_bundle_default() -> None:
    """Validate export of canonical URDF XML and metadata sidecar with default spec."""
    xml, sidecar = get_canonical_golfer_urdf_bundle()

    assert isinstance(xml, str)
    assert "<robot" in xml
    assert "</robot>" in xml

    assert isinstance(sidecar, dict)
    assert "coordinate_order" in sidecar
    assert len(sidecar["coordinate_order"]) == 41
    assert "body_links" in sidecar
    assert "frame_links" in sidecar
    assert "schema_version" in sidecar
    assert sidecar[
        "representation"
    ] == "canonical-full-body-urdf-bundle-v1" or "urdf" in sidecar.get(
        "representation", ""
    )


def test_get_canonical_golfer_urdf_bundle_explicit_doc() -> None:
    """Validate bundle export when passing an explicit spec document."""
    spec_doc = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    xml, sidecar = get_canonical_golfer_urdf_bundle(spec_doc)

    assert isinstance(xml, str)
    assert "<robot" in xml
    assert sidecar["coordinate_order"] == spec_doc["coordinate_order"]


def test_verify_spec_exported_urdf_engine_parity_success() -> None:
    """Verify cross-engine parity within strict tolerances (1 mm and 1e-9 kg)."""
    result = verify_spec_exported_urdf_engine_parity(
        mass_tol_kg=1e-9,
        frame_tol_m=1e-3,
    )

    assert result["passed"] is True
    assert result["coordinate_names_match"] is True
    assert len(result["coordinate_order"]) == 41
    assert result["max_mass_error_kg"] <= 1e-9
    assert result["max_frame_error_m"] <= 1e-3

    # Engine-specific checks
    assert result["max_urdf_frame_error_m"] <= 1e-3
    assert result["max_mjcf_frame_error_m"] <= 1e-3
    assert result["max_pinocchio_frame_error_m"] <= 1e-3
    assert result["pinocchio_joint_count"] == 23
    assert result["urdf_mass_kg"] == pytest.approx(result["spec_mass_kg"], abs=1e-9)
    assert result["mjcf_mass_kg"] == pytest.approx(result["spec_mass_kg"], abs=1e-9)


def test_verify_spec_exported_urdf_engine_parity_tolerance_enforcement() -> None:
    """Verify validation errors when tolerances are invalid or violated."""
    with pytest.raises(ValueError, match="mass_tol_kg must be > 0"):
        verify_spec_exported_urdf_engine_parity(mass_tol_kg=0.0)

    with pytest.raises(ValueError, match="frame_tol_m must be > 0"):
        verify_spec_exported_urdf_engine_parity(frame_tol_m=-0.001)

    with pytest.raises(ValueError, match="Frame error .* exceeds tolerance"):
        # Impossibly tight frame tolerance to verify failure path
        verify_spec_exported_urdf_engine_parity(frame_tol_m=1e-25)
