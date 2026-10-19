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
    compute_bundle_metrics,
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


def test_compute_bundle_metrics_matches_npz() -> None:
    """Bundle metrics match direct numpy calculation from deadlift_bundle.npz."""
    from pathlib import Path
    import numpy as np
    from src.shared.python.motion_matching.same_input import InputBundle

    repo_root = Path(__file__).resolve().parents[3]
    bundle_path = (
        repo_root / "docs/development/lifting/evidence/deadlift/deadlift_bundle.npz"
    )
    assert bundle_path.is_file(), f"missing bundle: {bundle_path}"

    metrics = compute_bundle_metrics(bundle_path)

    # Compute directly from bundle arrays
    bundle = InputBundle.load(bundle_path)
    ik_path = np.asarray(bundle.provenance["ik_bar_path_m"], dtype=float)
    times = np.asarray(bundle.provenance["sagittal_solution"]["time_s"], dtype=float)

    z = ik_path[:, 2]
    expected_start = float(z[0])
    expected_final = float(z[-1])
    expected_rise = float(z[-1] - z[0])
    expected_duration = float(times[-1] - times[0])
    expected_peak_v = float(np.max(np.diff(z) / np.diff(times)))

    assert metrics["bar_start_height_m"] == pytest.approx(expected_start, rel=1e-6)
    assert metrics["bar_final_height_m"] == pytest.approx(expected_final, rel=1e-6)
    assert metrics["bar_vertical_rise_m"] == pytest.approx(expected_rise, rel=1e-6)
    assert metrics["lift_duration_s"] == pytest.approx(expected_duration, rel=1e-6)
    assert metrics["peak_vertical_velocity_m_s"] == pytest.approx(
        expected_peak_v, rel=1e-6
    )


def test_comparison_table_infeasible_solution_has_none_metrics() -> None:
    """Infeasible solve must have None for all predictive metrics plus an explicit reason."""
    cfg = PredictiveDeadliftConfig()
    infeasible_results = {
        "converged": False,
        "solver_status": "Maximum_Iterations_Exceeded",
        "iterations": 50,
        "constraint_violation": 61.27,
    }
    table = build_comparison_table(infeasible_results, cfg)
    q = table["quantities"]

    for key, item in q.items():
        assert item["predictive"] is None, (
            f"infeasible result must be None for {key}, got {item['predictive']}"
        )
        assert "reason" in item, f"missing reason for None predictive metric {key}"
        assert "infeasible iterate" in item["reason"], (
            f"reason must mention infeasible iterate: {item['reason']}"
        )
        assert "61.27" in item["reason"], (
            f"reason must mention constraint violation: {item['reason']}"
        )


def test_comparison_table_missing_field_converged_has_none_not_fallback() -> None:
    """Converged solve with missing field must report None plus reason, NEVER a fallback float."""
    cfg = PredictiveDeadliftConfig()
    partial_results = {
        "converged": True,
        "solver_status": "Solve_Succeeded",
        "iterations": 120,
        "bar_start_height_m": 0.432,
        # Omit all other metrics
    }
    table = build_comparison_table(partial_results, cfg)
    q = table["quantities"]

    assert q["bar_start_height_m"]["predictive"] == 0.432
    assert q["bar_final_height_m"]["predictive"] is None
    assert "missing" in q["bar_final_height_m"]["reason"]
    assert q["bar_vertical_rise_m"]["predictive"] is None
    assert "missing" in q["bar_vertical_rise_m"]["reason"]
    assert q["peak_vertical_velocity_m_s"]["predictive"] is None
    assert "missing" in q["peak_vertical_velocity_m_s"]["reason"]
    assert q["lift_duration_s"]["predictive"] is None
    assert "missing" in q["lift_duration_s"]["reason"]


def test_config_actuation_mode() -> None:
    """Configuration validates torque and muscle actuation modes."""
    cfg_torque = PredictiveDeadliftConfig(
        actuation_mode="torque", torque_optimal_force_nm=500.0
    )
    assert cfg_torque.actuation_mode == "torque"
    assert cfg_torque.torque_optimal_force_nm == 500.0

    cfg_muscle = PredictiveDeadliftConfig(actuation_mode="muscle")
    assert cfg_muscle.actuation_mode == "muscle"

    with pytest.raises(ValueError, match="actuation_mode"):
        PredictiveDeadliftConfig(actuation_mode="pneumatic")

    with pytest.raises(ValueError, match="torque_optimal_force_nm"):
        PredictiveDeadliftConfig(torque_optimal_force_nm=0.0)


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
    assert "total_weight_n" in receipt
    assert "residual_peak_n" in receipt
    assert "residual_rms_n" in receipt
    assert "residual_ratio" in receipt
    assert "residual_assisted" in receipt


@pytest.mark.requires_opensim
@pytest.mark.slow
def test_compute_deadlift_poses() -> None:
    """Consistent setup and lockout poses place spheres near ground and bar near targets."""
    from src.engines.physics_engines.opensim.python.msk_lift_model import (
        build_lift_model,
    )
    from src.engines.physics_engines.opensim.python.msk_lift_predictive import (
        compute_deadlift_poses,
    )

    built = build_lift_model(
        "deadlift", body_mass_kg=80.0, height_m=1.78, bar_mass_kg=100.0
    )
    setup, lockout = compute_deadlift_poses(built.model)

    assert setup["bar_height_m"] == pytest.approx(0.432, abs=0.01)
    assert lockout["bar_height_m"] > setup["bar_height_m"]
    assert lockout["hip_flexion"] == 0.0
    assert lockout["knee_angle"] == 0.0
    assert setup["hip_flexion"] > 1.0
    assert setup["knee_angle"] > 1.0


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


def test_residual_optimal_force_scales_with_total_weight() -> None:
    """Residual optimal force is the configured multiple of total weight."""
    cfg = PredictiveDeadliftConfig(
        body_mass_kg=80.0, bar_mass_kg=100.0, residual_force_scale=2.0
    )
    assert cfg.residual_optimal_force_n() == pytest.approx(2.0 * 180.0 * 9.80665)
    with pytest.raises(ValueError):
        PredictiveDeadliftConfig(residual_force_scale=0.0)
    with pytest.raises(ValueError):
        PredictiveDeadliftConfig(residual_effort_weight=-1.0)


def test_residual_metrics_multiply_controls_by_optimal_force() -> None:
    """Nondimensional residual controls are converted to newtons."""
    import numpy as np

    from src.engines.physics_engines.opensim.python.msk_lift_predictive_report import (
        _extract_residual_metrics,
    )

    controls = np.array([[0.3, 0.4, 0.0], [0.0, 0.0, 0.0]])

    class _Matrix:
        def to_numpy(self) -> np.ndarray:
            return controls

    class _Table:
        labels = [
            "/forceset/residual_pelvis_tx",
            "/forceset/residual_pelvis_ty",
            "/forceset/residual_pelvis_tilt",
        ]

        def getNumColumns(self) -> int:  # noqa: N802 - OpenSim API name
            return len(self.labels)

        def getColumnLabel(self, i: int) -> str:  # noqa: N802 - OpenSim API name
            return self.labels[i]

        def getMatrix(self) -> _Matrix:  # noqa: N802 - OpenSim API name
            return _Matrix()

    class _Solution:
        def exportToControlsTable(self) -> _Table:  # noqa: N802 - OpenSim API name
            return _Table()

    peak, rms, ratio, assisted = _extract_residual_metrics(_Solution(), 1000.0, 200.0)
    assert peak == pytest.approx(0.5 * 200.0)
    assert rms == pytest.approx(np.sqrt(0.5) * 0.5 * 200.0)
    assert ratio == pytest.approx(rms / 1000.0)
    assert assisted is True
