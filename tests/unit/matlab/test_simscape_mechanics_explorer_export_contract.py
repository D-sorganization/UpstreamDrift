"""Contract verification for Simscape Mechanics Explorer video export (NV-8, #11681).

Verifies:
1. `scripts/matlab/export_simscape_mechanics_explorer_views.m` exists, enforces
   the MATLAB R2025b release gate, defines all four NV-1 camera view presets
   (face_on, down_the_line, overhead, perspective), invokes smwritevideo, and
   produces receipts with SHA-256 digests.
2. `docs/development/deferred-catalog-bundle.json` contains the NV-8 / #11681
   deferred record and satisfies bundle schema validation.
3. Camera geometry (view directions, camera positions, up vectors, and field
   of view) strictly matches `golf_view_presets` specification constants.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.golf_view_presets.framing import VIEWER_FOV_Y_RAD
from src.shared.python.golf_view_presets.presets import (
    VIEW_PRESETS,
    get_view_preset,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "matlab" / "export_simscape_mechanics_explorer_views.m"
)
CATALOG_PATH = REPO_ROOT / "docs" / "development" / "deferred-catalog-bundle.json"


def test_export_script_exists_and_enforces_r2025b_gate() -> None:
    """The export script must exist and fail closed if not running on R2025b."""
    assert SCRIPT_PATH.is_file(), f"Missing export script at {SCRIPT_PATH}"
    content = SCRIPT_PATH.read_text(encoding="utf-8")

    # Verify release check
    assert "version('-release')" in content
    assert "'2025b'" in content
    # Release gate assertion check
    release_assertion = re.search(
        r"assert\s*\(\s*strcmp\s*\(\s*rel\s*,\s*'2025b'\s*\)",
        content,
    )
    assert release_assertion is not None, (
        "Script must assert version('-release') == '2025b'"
    )


def test_export_script_defines_all_four_nv1_views() -> None:
    """The script must define face_on, down_the_line, overhead, and perspective."""
    content = SCRIPT_PATH.read_text(encoding="utf-8")

    expected_views = ["face_on", "down_the_line", "overhead", "perspective"]
    for view in expected_views:
        assert f'"{view}"' in content or f"'{view}'" in content, (
            f"View preset '{view}' must be defined in the export script"
        )

    # Check FOV specification matches 0.7 rad (VIEWER_FOV_Y_RAD)
    assert "0.7" in content
    assert "VIEWER_FOV_Y_RAD" in content or "fov_y_rad" in content


def test_export_script_invokes_smwritevideo_and_receipt() -> None:
    """The script must invoke smwritevideo and construct a JSON receipt with SHA-256."""
    content = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "smwritevideo" in content
    assert "CameraPosition" in content
    assert "CameraTarget" in content
    assert "CameraUpVector" in content
    assert "CameraViewAngle" in content

    # Check MP4 naming pattern
    for view in ["face_on", "down_the_line", "overhead", "perspective"]:
        assert f"%s_{view}.mp4" in content or f"_{view}.mp4" in content

    # Check receipt schema and sha256 calculation
    assert "simscape-mechanics-explorer-export/v1" in content
    assert "11681" in content
    assert "11673" in content
    assert "sha256" in content
    assert "jsonencode" in content


def test_deferred_catalog_bundle_contains_nv8_and_validates() -> None:
    """`deferred-catalog-bundle.json` must record NV-8 (#11681) and preserve schema."""
    assert CATALOG_PATH.is_file(), f"Missing catalog at {CATALOG_PATH}"
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    # Preserved top-level fields
    assert data.get("source_repository") == "D-sorganization/Repository_Management"
    assert "source_commit" in data
    assert "files" in data
    assert len(data["files"]) == 5
    for rel_path in [
        "shared_scripts/deferred_validation.py",
        "shared_scripts/deferred_planning.py",
        "shared_scripts/handoff_validator.py",
        "shared_scripts/handoff_schema.py",
        "shared_scripts/archive_policy.py",
    ]:
        assert rel_path in data["files"]

    # Deferred items section
    assert "deferred_items" in data
    items = data["deferred_items"]
    assert isinstance(items, list)

    nv8_matches = [item for item in items if item.get("id") == "NV-8"]
    assert len(nv8_matches) == 1, "Exactly one NV-8 entry expected in deferred items"

    nv8 = nv8_matches[0]
    assert nv8["issue"] == 11681
    assert nv8["epic_issue"] == 11673
    assert nv8["status"] == "deferred"
    assert "Simscape Mechanics Explorer" in nv8["title"]
    assert "R2025b" in nv8["host_requirement"]
    assert (
        nv8["export_script"]
        == "scripts/matlab/export_simscape_mechanics_explorer_views.m"
    )

    expected_views = ["face_on", "down_the_line", "overhead", "perspective"]
    assert nv8["view_presets"] == expected_views
    assert len(nv8["rationale"]) > 20


def test_camera_geometry_consistency_with_shared_presets() -> None:
    """Check mathematical agreement between MATLAB camera formulas and Python presets."""
    lookat = np.array([0.0, 0.0, 0.85])
    fov_rad = VIEWER_FOV_Y_RAD

    # Test mapping between MATLAB presets and Python presets
    preset_map = {
        "face_on": "face_on",
        "down_the_line": "down_the_line",
        "overhead": "overhead",
        "perspective": "oblique",  # oblique is perspective in specification
    }

    matlab_defs = {
        "face_on": {"az": 0.0, "el": -6.0, "dist": 3.4},
        "down_the_line": {"az": -90.0, "el": -8.0, "dist": 3.75},
        "overhead": {"az": 0.0, "el": -89.0, "dist": 2.28},
        "perspective": {"az": 135.0, "el": -14.0, "dist": 3.95},
    }

    world_up = np.array([0.0, 0.0, 1.0])

    for matlab_name, python_name in preset_map.items():
        py_preset = get_view_preset(python_name)
        m_def = matlab_defs[matlab_name]

        # Preset values must match
        assert math.isclose(py_preset.azimuth_deg, m_def["az"], abs_tol=1e-6)
        assert math.isclose(py_preset.elevation_deg, m_def["el"], abs_tol=1e-6)
        assert math.isclose(py_preset.default_distance_m, m_def["dist"], abs_tol=1e-6)

        # Compute MATLAB formula
        a = math.radians(m_def["az"])
        e = math.radians(m_def["el"])
        view_dir = np.array(
            [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]
        )
        pos = lookat - m_def["dist"] * view_dir
        right = np.cross(view_dir, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, view_dir)
        up = up / np.linalg.norm(up)

        # Compare with Python ViewPreset methods
        py_view_dir = py_preset.view_direction()
        py_pos = py_preset.camera_position(lookat, m_def["dist"])
        py_up = py_preset.image_up()

        np.testing.assert_allclose(view_dir, py_view_dir, atol=1e-7)
        np.testing.assert_allclose(pos, py_pos, atol=1e-7)
        np.testing.assert_allclose(up, py_up, atol=1e-7)

    # Verify FOV conversion in degrees
    fov_deg = fov_rad * 180.0 / math.pi
    assert math.isclose(fov_deg, 40.107045659, rel_tol=1e-5)
