"""Tests for OpenSim predictive Moco deadlift simulation (OSL-7, #13044).

Follows TDD red-first against pure-Python fake specifications and OpenSim 4.6.
Verifies:
1. PredictiveDeadliftConfig validation and bounds checking (Design by Contract).
2. Pure-Python problem specification builder: goals, state bounds, exploratory label.
3. Unavailable tracked solution reporting: explicit reason, never zero or invented.
4. Comparison table formatting between predictive solution and bundle kinematics.
5. Integration test (requires_opensim, slow): builds MocoStudy and checks solver init.
"""

from __future__ import annotations

import pytest

from src.engines.physics_engines.opensim.python.msk_lift_predictive import (
    TRACKED_SOLUTION_UNAVAILABLE_REASON,
    PredictiveDeadliftConfig,
    build_comparison_table,
    build_predictive_deadlift_study,
    create_predictive_deadlift_spec,
    format_predictive_receipt,
)

pytestmark = pytest.mark.unit


# -------------------------------------------------------------------------
# Unit Tests (Pure Python / Without OpenSim)
# -------------------------------------------------------------------------


def test_config_defaults() -> None:
    """Default configuration provides valid parameters and exploratory flag."""
    cfg = PredictiveDeadliftConfig()
    assert cfg.mesh_intervals >= 10
    assert cfg.bar_mass_kg == 100.0
    assert cfg.body_mass_kg == 80.0
    assert cfg.height_m == 1.78
    assert cfg.max_iterations > 0
    assert cfg.convergence_tolerance > 0.0
    assert cfg.constraint_tolerance > 0.0
    assert cfg.duration_bounds == (0.5, 3.0)
    assert cfg.exploratory is True


def test_config_validation_invalid_mesh() -> None:
    """Non-positive mesh intervals raise ValueError."""
    with pytest.raises(ValueError, match="mesh_intervals"):
        PredictiveDeadliftConfig(mesh_intervals=0)


def test_config_validation_invalid_mass() -> None:
    """Non-positive mass raises ValueError."""
    with pytest.raises(ValueError, match="bar_mass_kg"):
        PredictiveDeadliftConfig(bar_mass_kg=-10.0)

    with pytest.raises(ValueError, match="body_mass_kg"):
        PredictiveDeadliftConfig(body_mass_kg=0.0)


def test_config_validation_invalid_duration_bounds() -> None:
    """Invalid duration bounds raise ValueError."""
    with pytest.raises(ValueError, match="duration_bounds"):
        PredictiveDeadliftConfig(duration_bounds=(2.0, 1.0))


def test_spec_goals_and_bounds() -> None:
    """Pure-Python spec defines effort goal, terminal condition, and setup/lockout bounds."""
    cfg = PredictiveDeadliftConfig(mesh_intervals=25)
    spec = create_predictive_deadlift_spec(cfg)

    assert spec["exploratory"] is True
    assert "effort" in spec["goals"]
    assert "terminal_lockout" in spec["goals"]
    assert spec["mesh_intervals"] == 25

    # Check setup and lockout coordinate targets/bounds
    bounds = spec["coordinate_bounds"]
    assert "pelvis_ty" in bounds
    assert "hip_flexion_r" in bounds
    assert "knee_angle_r" in bounds
    assert "lumbar_extension" in bounds

    # Check that welded joints and active actuators are recorded
    assert "welded_joints" in spec
    assert "acromial_r" in spec["welded_joints"]
    assert "elbow_r" in spec["welded_joints"]
    assert "subtalar_r" in spec["welded_joints"]
    assert "mtp_r" in spec["welded_joints"]


def test_comparison_table_reports_unavailable_for_tracked_opensim() -> None:
    """Comparison table reports explicit reason for tracked OpenSim solution."""
    cfg = PredictiveDeadliftConfig()
    predictive_results = {
        "bar_start_height_m": 0.456,
        "bar_final_height_m": 0.935,
        "bar_vertical_rise_m": 0.479,
        "peak_vertical_velocity_m_s": 0.42,
        "duration_s": 1.5,
        "converged": False,
        "solver_status": "Maximum_Iterations_Exceeded",
    }
    table = build_comparison_table(predictive_results, cfg)

    assert "quantities" in table
    assert "tracked_opensim" in table
    assert table["tracked_opensim"]["status"] == "unavailable"
    assert table["tracked_opensim"]["reason"] == TRACKED_SOLUTION_UNAVAILABLE_REASON
    assert "OSL-3 #13041 is still open" in table["tracked_opensim"]["reason"]

    # Bundle kinematics row must exist
    assert "bundle_kinematics" in table
    assert table["bundle_kinematics"]["label"] == "bundle kinematics"


def test_format_predictive_receipt() -> None:
    """Receipt contains all mandatory fields from RULES.md."""
    cfg = PredictiveDeadliftConfig()
    solve_summary = {
        "solver_status": "Solve_Succeeded",
        "converged": True,
        "iterations": 45,
        "objective": 123.45,
        "constraint_violation": 0.005,
        "wall_clock_s": 120.5,
    }
    receipt = format_predictive_receipt(cfg, solve_summary)

    assert receipt["exploratory"] is True
    assert receipt["converged"] is True
    assert receipt["solver_status"] == "Solve_Succeeded"
    assert receipt["iterations"] == 45
    assert receipt["objective"] == 123.45
    assert receipt["wall_clock_s"] == 120.5
    assert "welded_joints" in receipt
    assert "actuators" in receipt
    assert "reproduction_command" in receipt
    assert "comparison_table" in receipt


# -------------------------------------------------------------------------
# Integration Test (Real OpenSim 4.6)
# -------------------------------------------------------------------------


@pytest.mark.requires_opensim
@pytest.mark.slow
def test_build_predictive_deadlift_study_initializes() -> None:
    """MocoStudy builds and its CasADi solver initializes cleanly."""
    cfg = PredictiveDeadliftConfig(mesh_intervals=15, max_iterations=2)
    study, meta = build_predictive_deadlift_study(cfg)

    assert study is not None
    assert meta["exploratory"] is True
    assert meta["mesh_intervals"] == 15

    # Check solver initialization
    solver = study.updSolver()
    assert solver is not None
