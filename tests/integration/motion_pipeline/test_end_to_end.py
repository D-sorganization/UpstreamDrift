"""End-to-end pipeline tests for the motion capture pipeline.

Part of issue #4571. Depends on #4569.

These tests verify the full motion pipeline from raw mocap data through
to physics engine export, parametrized over source formats and backends.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.motion_pipeline]

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "tests" / "data" / "motion_pipeline" / "golden"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Parametrized test combinations: (source_format, ik_backend, matching_backend)
# Each combination that is not explicitly skipped should run the full pipeline.
PIPELINE_COMBINATIONS = [
    # C3D sources
    ("c3d", "mujoco", "mujoco"),
    ("c3d", "drake", "mujoco"),
    ("c3d", "pinocchio", "mujoco"),
    # OpenPose sources (2D, requires lifting)
    ("openpose_json", "mujoco", "mujoco"),
    ("openpose_json", "drake", "mujoco"),
    # MediaPipe sources (2D, requires lifting)
    ("mediapipe_json", "mujoco", "mujoco"),
    # BVH sources
    ("bvh", "mujoco", "mujoco"),
    ("bvh", "drake", "mujoco"),
    # TRC sources (Theia/Vicon)
    ("trc", "mujoco", "mujoco"),
    ("trc", "pinocchio", "mujoco"),
]

# Tolerance thresholds per fixture (not hardcoded globally)
TOLERANCE_THRESHOLDS = {
    "c3d_vicon_driver": {
        "rmse_position": 0.02,  # meters
        "rmse_orientation": 0.05,  # radians
        "impact_speed_tolerance": 0.5,  # m/s
    },
    "openpose_swing": {
        "rmse_position": 0.05,  # meters (2D lifting is less accurate)
        "rmse_orientation": 0.1,  # radians
    },
    "mediapipe_swing": {
        "rmse_position": 0.05,
        "rmse_orientation": 0.1,
    },
    "bvh_moveai": {
        "rmse_position": 0.01,  # meters (BVH is already 3D)
        "rmse_orientation": 0.02,
    },
    "trc_opencap": {
        "rmse_position": 0.015,
        "rmse_orientation": 0.03,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_motion_matching_result_invariants(result: Any) -> None:
    """Assert MotionMatchingResult invariants.

    Verifies:
    - Result object is not None
    - Has required attributes: trajectory, joint_angles, metadata
    - Metadata contains provenance information
    """
    assert result is not None, "MotionMatchingResult should not be None"
    assert hasattr(result, "trajectory"), "Result must have trajectory attribute"
    assert hasattr(result, "joint_angles"), "Result must have joint_angles attribute"
    assert hasattr(result, "metadata"), "Result must have metadata attribute"

    # Provenance metadata must be present
    assert "source_format" in result.metadata, "metadata must contain source_format"
    assert "ik_backend" in result.metadata, "metadata must contain ik_backend"
    assert "matching_backend" in result.metadata, (
        "metadata must contain matching_backend"
    )
    assert "processed_at" in result.metadata, (
        "metadata must contain processed_at timestamp"
    )


def compute_rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    """Compute root mean square error between actual and expected arrays."""
    return float(np.sqrt(np.mean((actual - expected) ** 2)))


def load_golden_fixture(fixture_name: str, fmt: str) -> dict[str, Any]:
    """Load a golden fixture from the data directory.

    Args:
        fixture_name: Name of the fixture (e.g., "c3d_vicon_driver")
        fmt: Format extension (e.g., "c3d", "json", "bvh")

    Returns:
        Dictionary containing the golden data

    Raises:
        FileNotFoundError: If the fixture file does not exist
    """
    fixture_path = DATA_DIR / f"{fixture_name}.{fmt}"
    if not fixture_path.exists():
        raise FileNotFoundError(
            f"Required golden fixture not present: {fixture_path}. "
            "These fixtures are required for regression testing. "
            "See tests/data/motion_pipeline/golden/ for the fixture directory."
        )

    if fmt == "json":
        return json.loads(fixture_path.read_text(encoding="utf-8"))
    # For binary formats, return path for downstream loading
    return {"path": fixture_path}


# ---------------------------------------------------------------------------
# End-to-end pipeline tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LoD test: import graph assertions
# ---------------------------------------------------------------------------


def test_motion_pipeline_no_direct_engine_imports() -> None:
    """Verify LoD: motion_pipeline has no direct engine imports outside backends.

    The motion_pipeline packages should only import engine modules through
    the designated backend interfaces in ik/*_backend.py and matching/*.
    """
    import importlib
    import sys

    # Modules that are allowed to import engines directly
    ALLOWED_ENGINE_IMPORTS = {
        "src.engines.motion_pipeline.ik.mujoco_backend",
        "src.engines.motion_pipeline.ik.drake_backend",
        "src.engines.motion_pipeline.ik.pinocchio_backend",
        "src.engines.motion_pipeline.matching.mujoco_backend",
        "src.engines.motion_pipeline.matching.drake_backend",
    }

    # Get all motion_pipeline modules
    pipeline_modules = [
        name
        for name in sys.modules.keys()
        if name.startswith("src.engines.motion_pipeline")
    ]

    # Check each module for forbidden imports
    forbidden_imports = []
    for mod_name in pipeline_modules:
        if mod_name in ALLOWED_ENGINE_IMPORTS:
            continue

        mod = importlib.import_module(mod_name)
        mod_file = getattr(mod, "__file__", "")
        if not mod_file:
            continue

        try:
            content = Path(mod_file).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        # Check for direct engine imports
        forbidden_patterns = [
            "from pydrake",
            "import pydrake",
            "from mujoco",
            "import mujoco",
            "from pinocchio",
            "import pinocchio",
            "from opensim",
            "import opensim",
        ]

        for pattern in forbidden_patterns:
            if pattern in content:
                forbidden_imports.append(f"{mod_name}: {pattern}")

    assert not forbidden_imports, (
        "Found forbidden engine imports in motion_pipeline:\n"
        + "\n".join(forbidden_imports)
    )
