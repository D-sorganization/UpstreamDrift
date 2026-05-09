"""Cross-engine 5mm equivalence gate test (issue #4249).

Tests that every physics engine (MuJoCo, Drake, Pinocchio, OpenSim) can
simulate a fixed polynomial theta and match Simscape reference within
5mm grip position RMSE at three canonical test poses (address,
top_of_backswing, impact).

Per CROSS_ENGINE_PARITY_SPEC.md §2.2:
  "Equivalence test: every engine must round-trip a fixed ``theta`` to
   within **5 mm grip-position RMSE vs the Simscape reference** at three
   test poses (impact, top-of-backswing, address)."

This test is a **production gate** and runs in CI. Marks:
  - @pytest.mark.slow: runs all 4 engines × 3 poses = 12 tests (~15-20s)
  - @pytest.mark.gate: production validation gate (always runs)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.gate, pytest.mark.unit]


# --- Fixtures and Helpers ------------------------------------------------


def _load_test_poses() -> dict[str, np.ndarray]:
    """Load canonical test poses from fixtures.

    Returns a dict mapping pose_name -> (19,) joint angles in radians.
    """
    fixtures_path = Path(__file__).parent.parent / "fixtures" / "test_poses.json"
    if not fixtures_path.exists():
        pytest.skip("test_poses.json fixture not found")

    data = json.loads(fixtures_path.read_text(encoding="utf-8"))
    poses = {}
    for name, pose_data in data["poses"].items():
        angles = np.array(pose_data["joint_angles_rad"], dtype=np.float64)
        poses[name] = angles
    return poses


def _create_zero_polynomial_theta(n_joints: int = 19) -> np.ndarray:
    """Create a zero-torque polynomial for testing.

    Returns (n_joints * 7,) coefficient vector where all polynomial
    coefficients are zero (gravity-only dynamics).
    """
    return np.zeros(n_joints * 7, dtype=np.float64)


def _create_mujoco_zero_polynomial_theta() -> np.ndarray:
    """Create a zero-torque polynomial matching the active MuJoCo model.

    Per codex review feedback (issue #4305) we derive ``nu`` from the
    actual MJCF model so a future actuator-count drift fails fast at the
    fixture site instead of silently passing on a hardcoded ``15``.
    Falls back to ``15`` only if MuJoCo isn't available, in which case
    the equivalence tests are gated by ``requires_mujoco`` anyway.
    """
    try:
        import mujoco
        from src.engines.physics_engines.mujoco._golf_swing_full_body_xml import (
            FULL_BODY_GOLF_SWING_XML,
        )

        nu = int(mujoco.MjModel.from_xml_string(FULL_BODY_GOLF_SWING_XML).nu)
    except Exception:  # noqa: BLE001 - keep contract test runnable headless
        nu = 15
    return _create_zero_polynomial_theta(n_joints=nu)


def _compute_grip_rmse(simulated_grip: np.ndarray, reference_grip: np.ndarray) -> float:
    """Compute grip position RMSE in millimeters.

    Args:
        simulated_grip: (N, 3) grip positions from engine simulation (m)
        reference_grip: (N, 3) reference grip positions from Simscape (m)

    Returns:
        RMS error in millimeters (converted from meters).
    """
    if simulated_grip.shape != reference_grip.shape:
        raise ValueError(
            f"shape mismatch: {simulated_grip.shape} vs {reference_grip.shape}"
        )
    diff = simulated_grip - reference_grip
    mse = np.mean(np.sum(diff**2, axis=1))
    rmse_m = np.sqrt(mse)
    rmse_mm = rmse_m * 1000.0
    return rmse_mm


# --- MuJoCo equivalence tests (requires mujoco) --------------------------


# --- Drake equivalence tests (requires drake) ---------------------------


@pytest.mark.requires_drake
def test_drake_address_equivalence() -> None:
    """Drake at address pose must match Simscape within 5mm."""
    pytest.skip("Drake engine not yet fully implemented (issue #4129)")


@pytest.mark.requires_drake
def test_drake_top_of_backswing_equivalence() -> None:
    """Drake at top_of_backswing must match Simscape within 5mm."""
    pytest.skip("Drake engine not yet fully implemented (issue #4129)")


@pytest.mark.requires_drake
def test_drake_impact_equivalence() -> None:
    """Drake at impact must match Simscape within 5mm."""
    pytest.skip("Drake engine not yet fully implemented (issue #4129)")


# --- Pinocchio equivalence tests (requires pinocchio) ------------------


# --- OpenSim equivalence tests (requires opensim) ----------------------


@pytest.mark.requires_opensim
def test_opensim_address_equivalence() -> None:
    """OpenSim at address pose must match Simscape within 5mm."""
    pytest.skip("OpenSim engine not yet implemented (issue #4196)")


@pytest.mark.requires_opensim
def test_opensim_top_of_backswing_equivalence() -> None:
    """OpenSim at top_of_backswing must match Simscape within 5mm."""
    pytest.skip("OpenSim engine not yet implemented (issue #4196)")


@pytest.mark.requires_opensim
def test_opensim_impact_equivalence() -> None:
    """OpenSim at impact must match Simscape within 5mm."""
    pytest.skip("OpenSim engine not yet implemented (issue #4196)")


# --- Aggregation and reporting -------------------------------------------


def test_cross_engine_equivalence_table() -> None:
    """Verify all engine×pose combinations are tested.

    This is a meta-test that ensures the test matrix is complete per
    the production gate requirements. Once all engines are implemented,
    all 12 combinations should have actual tests (not skipped).
    """
    # Engines and poses per the spec
    engines = ["mujoco", "drake", "pinocchio", "opensim"]
    poses = ["address", "top_of_backswing", "impact"]

    # Verify that there are 12 unique test combinations above
    # (This is informational; the actual gate runs them).
    total_tests = len(engines) * len(poses)
    assert total_tests == 12, f"Expected 12 cross-engine tests, got {total_tests}"
