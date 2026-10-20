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

from pathlib import Path

import numpy as np
from typing import Any

import pytest

from src.engines.physics_engines.opensim.python.msk_lift_predictive import (
    PredictiveDeadliftConfig,
    build_comparison_table,
    build_predictive_deadlift_study,
    compute_bundle_metrics,
    create_predictive_deadlift_spec,
    format_predictive_receipt,
)
from src.engines.physics_engines.opensim.python.msk_lift_predictive_report import (
    tracked_opensim_record,
)

pytestmark = pytest.mark.unit

BUNDLE = (
    Path(__file__).resolve().parents[3]
    / "docs/development/lifting/evidence/deadlift/deadlift_bundle.npz"
)


def _fake_solution(labels: list[str], controls: Any) -> Any:
    """A MocoSolution stand-in whose controls table holds ``controls``."""

    class _Matrix:
        def to_numpy(self) -> Any:
            return controls

    class _Table:
        def getNumColumns(self) -> int:  # noqa: N802 - OpenSim API name
            return len(labels)

        def getColumnLabel(self, i: int) -> str:  # noqa: N802 - OpenSim API name
            return labels[i]

        def getMatrix(self) -> _Matrix:  # noqa: N802 - OpenSim API name
            return _Matrix()

    class _Solution:
        def exportToControlsTable(self) -> _Table:  # noqa: N802 - OpenSim API name
            return _Table()

    return _Solution()


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

    # Bounds come from the solved poses, never from constants in the spec.
    assert "compute_deadlift_poses" in spec["coordinate_bounds"]

    # Straight arms that flex at the shoulder, held in the sagittal plane.
    assert "welded_joints" in spec
    assert "acromial_r" not in spec["welded_joints"]
    assert "elbow_r" in spec["welded_joints"]
    for coord in ("arm_add_r", "arm_rot_r", "arm_add_l", "arm_rot_l"):
        assert coord in spec["zero_constrained_coordinates"]
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
    assert table["tracked_opensim"] == tracked_opensim_record()
    assert table["tracked_opensim"]["status"] in {"unavailable", "not compared"}
    assert table["tracked_opensim"]["reason"]

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
    assert receipt["residual_optimal_force_n"] == cfg.residual_optimal_force_n()
    assert receipt["residual_effort_weight"] == cfg.residual_effort_weight
    for key in (
        "residual_moment_peak_nm",
        "residual_moment_rms_nm",
        "residual_moment_limit_nm",
        "residual_moment_within_limit",
    ):
        assert key in receipt


@pytest.mark.requires_opensim
@pytest.mark.slow
def test_compute_deadlift_poses() -> None:
    """Consistent setup and lockout poses place spheres near ground and bar near targets."""
    from src.engines.physics_engines.opensim.python.msk_lift_model import (
        build_lift_model,
    )
    from src.engines.physics_engines.opensim.python.msk_lift_predictive import (
        POSE_FLOOR_PENETRATION_M,
        _clean_model_and_get_welds,
        compute_deadlift_poses,
    )
    from src.engines.physics_engines.opensim.python.msk_lift_predictive_report import (
        bundle_pose_targets,
    )

    built = build_lift_model(
        "deadlift", body_mass_kg=80.0, height_m=1.78, bar_mass_kg=100.0
    )
    _clean_model_and_get_welds(built.model)
    targets = bundle_pose_targets(BUNDLE)
    setup, lockout = compute_deadlift_poses(built.model, targets)

    assert setup["bar_height_m"] == pytest.approx(
        targets["setup_bar_height_m"], abs=0.005
    )
    assert lockout["bar_height_m"] > setup["bar_height_m"]
    assert lockout["hip_flexion"] == 0.0
    assert lockout["knee_angle"] == 0.0
    assert setup["hip_flexion"] > 1.0
    assert setup["knee_angle"] > 1.0
    for pose in (setup, lockout):
        support = _static_support(built.model, pose)
        # Flat-footed: every foot sphere touches the floor, none deeper than
        # the seating penetration; the combined COM lies over the spheres.
        assert support["sphere_y"].max() <= 0.0, pose
        assert support["sphere_y"].min() >= -POSE_FLOOR_PENETRATION_M - 1e-9, pose
        assert (
            support["sphere_x"].min() <= support["com_x"] <= support["sphere_x"].max()
        )
    # The feet stay planted.  Measured: the spheres coincide within 1.9 mm.
    # They cannot coincide exactly: the Rajagopal ankle axis is oblique and
    # the subtalar joint is welded, so changing the ankle angle turns the foot
    # slightly, which no pelvis translation removes.
    np.testing.assert_allclose(
        _static_support(built.model, setup)["sphere_x"],
        _static_support(built.model, lockout)["sphere_x"],
        atol=2.5e-3,
    )
    # A deadlift setup: bar over midfoot, trunk inclined forward of the hips.
    support = _static_support(built.model, setup)
    assert support["bar_x"] == pytest.approx(support["sphere_x"].mean(), abs=0.02)
    assert support["shoulder_x"] > support["hip_x"]
    # Trunk inclinations are the bundle's; the arms hang vertically.
    for pose, key in ((setup, "setup_trunk_rad"), (lockout, "lockout_trunk_rad")):
        support = _static_support(built.model, pose)
        assert support["trunk_rad"] == pytest.approx(targets[key], abs=1e-4)
        assert support["hand_x"] == pytest.approx(support["shoulder_x"], abs=1e-4)


def _static_support(model: Any, pose: dict[str, Any]) -> dict[str, Any]:
    """Recompute, by forward kinematics, where a pose puts COM, bar and feet."""
    from src.engines.physics_engines.opensim.python.msk_lift_model import (
        _foot_spheres,
        _ground_points,
    )

    state = model.initSystem()
    cset = model.getCoordinateSet()
    for key in ("hip_flexion", "knee_angle", "ankle_angle"):
        for side in "rl":
            cset.get(f"{key}_{side}").setValue(state, pose[key], False)
    for side in "rl":
        cset.get(f"arm_flex_{side}").setValue(state, pose["arm_flex"], False)
    for key in ("pelvis_tilt", "lumbar_extension", "pelvis_tx", "pelvis_ty"):
        cset.get(key).setValue(state, pose[key], False)
    model.realizePosition(state)
    points = _ground_points(model, state, _foot_spheres(1.78 / 1.70))
    bodies = model.getBodySet()

    def _xy(body: str) -> np.ndarray:
        pos = bodies.get(body).getPositionInGround(state)
        return np.array([pos.get(0), pos.get(1)])

    trunk = _xy("humerus_r") - _xy("femur_r")
    return {
        "trunk_rad": float(np.arctan2(trunk[0], trunk[1])),
        "hand_x": _xy("hand_r")[0],
        "com_x": model.calcMassCenterPosition(state).get(0),
        "sphere_x": np.sort(points[:, 0]),
        "sphere_y": points[:, 1],
        "bar_x": bodies.get("barbell").getPositionInGround(state).get(0),
        "shoulder_x": bodies.get("humerus_r").getPositionInGround(state).get(0),
        "hip_x": bodies.get("femur_r").getPositionInGround(state).get(0),
    }


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

    labels = [
        "/forceset/residual_pelvis_tx",
        "/forceset/residual_pelvis_ty",
        "/forceset/residual_pelvis_tilt",
    ]

    peak, rms, ratio, assisted = _extract_residual_metrics(
        _fake_solution(labels, controls), 1000.0, 200.0
    )
    assert peak == pytest.approx(0.5 * 200.0)
    assert rms == pytest.approx(np.sqrt(0.5) * 0.5 * 200.0)
    assert ratio == pytest.approx(rms / 1000.0)
    assert assisted is True


def test_dynamics_mode_validated() -> None:
    """Only explicit and implicit multibody dynamics modes are accepted."""
    assert PredictiveDeadliftConfig(multibody_dynamics_mode="implicit")
    with pytest.raises(ValueError):
        PredictiveDeadliftConfig(multibody_dynamics_mode="semi")


def test_source_commit_falls_back_to_env_without_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A container without usable git reads the commit from the environment."""
    import importlib
    import subprocess

    mod = importlib.import_module("scripts.measure_opensim_predictive_deadlift")

    def _fail(*_a: object, **_k: object) -> None:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(mod.subprocess, "run", _fail)
    monkeypatch.setenv("UPSTREAMDRIFT_SOURCE_COMMIT", "abc123")
    assert mod._git("rev-parse", "HEAD") == "abc123"
    monkeypatch.delenv("UPSTREAMDRIFT_SOURCE_COMMIT")
    assert mod._git("rev-parse", "HEAD") is None


def test_stage1_guess_warm_starts_the_solve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A saved trajectory becomes the stage-1 initial guess and is recorded."""
    import importlib

    mod = importlib.import_module("scripts.measure_opensim_predictive_deadlift")
    sto = tmp_path / "prior.sto"
    sto.write_text("stub", encoding="utf-8")
    seen: dict[str, object] = {}
    monkeypatch.setattr(mod, "_load_guess", lambda path: ("guess", path))
    monkeypatch.setattr(
        mod, "build_predictive_deadlift_study", lambda cfg: ("study", {})
    )

    def _solve(config: object, **kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(mod, "solve_predictive_deadlift", _solve)
    monkeypatch.setattr(
        mod,
        "_parse_solve_log",
        lambda path: dict.fromkeys(
            ("iterations", "objective", "constraint_violation", "last_iterate_inf_pr")
        ),
    )
    summary = mod.run_stage1_solve(
        PredictiveDeadliftConfig(), tmp_path / "s1.log", guess_path=sto
    )
    assert seen["initial_guess"] == ("guess", sto)
    assert summary["warm_start"] == str(sto)


def test_load_guess_rejects_missing_file(tmp_path: Path) -> None:
    """A missing warm-start trajectory is an error, never a silent cold start."""
    import importlib

    mod = importlib.import_module("scripts.measure_opensim_predictive_deadlift")
    with pytest.raises(ValueError, match="not found"):
        mod._load_guess(tmp_path / "absent.sto")


def test_residuals_cover_every_pelvis_coordinate() -> None:
    """Out-of-plane pelvis DOFs held at zero still need an actuator (#13044)."""
    from src.engines.physics_engines.opensim.python.msk_lift_predictive import (
        PELVIS_RESIDUAL_COORDINATES,
        ZERO_CONSTRAINED_COORDS,
        create_predictive_deadlift_spec,
    )

    pelvis = {f"pelvis_{c}" for c in ("tilt", "list", "rotation", "tx", "ty", "tz")}
    assert set(PELVIS_RESIDUAL_COORDINATES) == pelvis
    held = {c for c in ZERO_CONSTRAINED_COORDS if c.startswith("pelvis_")}
    assert held <= set(PELVIS_RESIDUAL_COORDINATES)
    for mode in ("torque", "muscle"):
        spec = create_predictive_deadlift_spec(
            PredictiveDeadliftConfig(actuation_mode=mode)
        )
        for c in pelvis:
            assert f"residual_{c}" in spec["actuators"]


def test_residual_force_norm_includes_tz() -> None:
    """The lateral residual counts toward the reported residual force."""
    import numpy as np

    from src.engines.physics_engines.opensim.python.msk_lift_predictive_report import (
        _extract_residual_metrics,
    )

    controls = np.array([[0.3, 0.0, 0.4]])

    labels = [
        "/forceset/residual_pelvis_tx",
        "/forceset/residual_pelvis_ty",
        "/forceset/residual_pelvis_tz",
    ]

    peak, _rms, _ratio, _assisted = _extract_residual_metrics(
        _fake_solution(labels, controls), 1.0, 100.0
    )
    assert peak == pytest.approx(50.0)


@pytest.mark.requires_opensim
@pytest.mark.slow
def test_bar_kinematics_replays_on_the_processed_model() -> None:
    """FK replay runs on a solved trajectory (it crashed on a MocoProblemRep)."""
    pytest.importorskip("opensim")
    import numpy as np

    from src.engines.physics_engines.opensim.python.msk_lift_predictive_report import (
        _extract_bar_kinematics,
    )

    config = PredictiveDeadliftConfig(mesh_intervals=3, max_iterations=1)
    study, _meta = build_predictive_deadlift_study(config)
    solution = study.solve()
    solution.unseal()
    metrics = _extract_bar_kinematics(study, solution)
    assert set(metrics) == {
        "bar_start_height_m",
        "bar_final_height_m",
        "bar_vertical_rise_m",
        "peak_vertical_velocity_m_s",
        "duration_s",
        "com_height_m",
    }
    assert all(np.isfinite(v) for v in metrics.values())


@pytest.mark.requires_opensim
@pytest.mark.slow
def test_stage2_guess_is_built_from_stage1_states() -> None:
    """The muscle study's guess carries stage 1's grid, states and shared controls."""
    pytest.importorskip("opensim")
    import importlib

    import numpy as np

    mod = importlib.import_module("scripts.measure_opensim_predictive_deadlift")
    study1, _ = build_predictive_deadlift_study(
        PredictiveDeadliftConfig(mesh_intervals=3, max_iterations=1)
    )
    stage1 = study1.solve()
    stage1.unseal()
    study2, _ = build_predictive_deadlift_study(
        PredictiveDeadliftConfig(
            actuation_mode="muscle", mesh_intervals=3, max_iterations=1
        )
    )
    guess = mod.stage2_guess_from_stage1(study2, stage1)
    assert guess.getNumTimes() == stage1.getNumTimes()
    np.testing.assert_allclose(guess.getTime().to_numpy(), stage1.getTime().to_numpy())
    name = next(n for n in stage1.getStateNames() if n.endswith("pelvis_ty/value"))
    np.testing.assert_allclose(
        guess.getState(name).to_numpy(), stage1.getState(name).to_numpy()
    )
    shared = "/forceset/residual_pelvis_ty"
    np.testing.assert_allclose(
        guess.getControl(shared).to_numpy(), stage1.getControl(shared).to_numpy()
    )
    assert any("activation" in n for n in guess.getStateNames())


def test_tracked_row_follows_the_osl3_receipt(tmp_path: Path) -> None:
    """The tracked row reports the receipt's state, never a hard-coded claim."""
    import json

    from src.engines.physics_engines.opensim.python.msk_lift_predictive_report import (
        tracked_opensim_record,
    )

    missing = tracked_opensim_record(tmp_path / "absent.json")
    assert missing["status"] == "unavailable"
    assert "unreadable" in missing["reason"]
    receipt = tmp_path / "receipts.json"
    receipt.write_text(
        json.dumps(
            {"lifts": {"deadlift": {"ik_id": {"status": "unavailable", "reason": "x"}}}}
        ),
        encoding="utf-8",
    )
    assert tracked_opensim_record(receipt)["status"] == "unavailable"
    receipt.write_text(
        json.dumps({"lifts": {"deadlift": {"ik_id": {"status": "complete"}}}}),
        encoding="utf-8",
    )
    assert tracked_opensim_record(receipt)["status"] == "not compared"


def test_residual_moments_are_reported_in_newton_metres() -> None:
    """The rotational pelvis residuals are reported, never dropped."""
    import numpy as np

    from src.engines.physics_engines.opensim.python.msk_lift_predictive_report import (
        residual_moment_metrics,
    )

    labels = [
        "/forceset/residual_pelvis_tilt",
        "/forceset/residual_pelvis_list",
        "/forceset/residual_pelvis_rotation",
        "/forceset/residual_pelvis_tx",
    ]
    controls = np.array([[0.3, 0.4, 0.0, 9.0], [0.0, 0.0, 0.0, 9.0]])
    peak, rms = residual_moment_metrics(_fake_solution(labels, controls), 100.0)
    assert peak == pytest.approx(50.0)
    assert rms == pytest.approx(np.sqrt(0.5) * 50.0)


def test_residual_moments_unavailable_without_rotational_columns() -> None:
    """No rotational residual columns give None, not zero."""
    import numpy as np

    from src.engines.physics_engines.opensim.python.msk_lift_predictive_report import (
        residual_moment_metrics,
    )

    labels = ["/forceset/residual_pelvis_tx"]
    solution = _fake_solution(labels, np.array([[0.1]]))
    assert residual_moment_metrics(solution, 100.0) == (None, None)


def test_residual_moment_gate_uses_the_hicks_fraction() -> None:
    """The moment limit is the OSL-3 Hicks fraction of COM height times weight."""
    from src.engines.physics_engines.opensim.python.msk_lift_ik_id import (
        RESIDUAL_MOMENT_FRACTION,
    )
    from src.engines.physics_engines.opensim.python.msk_lift_predictive_report import (
        residual_moment_gate,
    )

    gate = residual_moment_gate(30.0, 0.9, 1000.0)
    assert gate["residual_moment_limit_nm"] == pytest.approx(
        RESIDUAL_MOMENT_FRACTION * 0.9 * 1000.0
    )
    assert gate["residual_moment_within_limit"] is False
    assert residual_moment_gate(5.0, 0.9, 1000.0)["residual_moment_within_limit"]
    unknown = residual_moment_gate(5.0, None, 1000.0)
    assert unknown["residual_moment_limit_nm"] is None
    assert unknown["residual_moment_within_limit"] is None
    with pytest.raises(ValueError):
        residual_moment_gate(5.0, 0.9, 0.0)


def test_fixed_duration_is_a_valid_bound() -> None:
    """Equal bounds pin the final time (the effort goal otherwise hits t_min)."""
    cfg = PredictiveDeadliftConfig(duration_bounds=(2.0, 2.0))
    assert cfg.duration_bounds == (2.0, 2.0)
    with pytest.raises(ValueError, match="duration_bounds"):
        PredictiveDeadliftConfig(duration_bounds=(0.0, 0.0))


def test_cli_duration_bounds_reach_the_stage1_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--duration-bounds`` sets the stage-1 final-time bounds."""
    import importlib

    mod = importlib.import_module("scripts.measure_opensim_predictive_deadlift")
    seen: dict[str, object] = {}

    def _stop(config: PredictiveDeadliftConfig, *args: object) -> dict[str, object]:
        seen["bounds"] = config.duration_bounds
        raise SystemExit(0)

    monkeypatch.setattr(mod, "run_stage1_solve", _stop)
    with pytest.raises(SystemExit):
        mod.main(
            [
                "--out-dir",
                str(tmp_path),
                "--log-dir",
                str(tmp_path),
                "--duration-bounds",
                "2.0",
                "2.0",
            ]
        )
    assert seen["bounds"] == (2.0, 2.0)


def test_bundle_pose_targets_read_the_optimizer_solution() -> None:
    """Setup bar height and trunk inclinations are the bundle's, not constants."""
    from src.engines.physics_engines.opensim.python.msk_lift_predictive_report import (
        bundle_pose_targets,
    )
    from src.shared.python.motion_matching.same_input import InputBundle

    targets = bundle_pose_targets(BUNDLE)
    sagittal = InputBundle.load(BUNDLE).provenance["sagittal_solution"]
    trunk = np.asarray(sagittal["segment_angles_rad"], dtype=float)[:, 2]
    assert targets["setup_trunk_rad"] == trunk[0]
    assert targets["lockout_trunk_rad"] == trunk[-1]
    metrics = compute_bundle_metrics(BUNDLE)
    assert targets["setup_bar_height_m"] == metrics["bar_start_height_m"]
