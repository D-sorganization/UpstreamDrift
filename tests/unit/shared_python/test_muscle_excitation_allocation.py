from __future__ import annotations

from dataclasses import replace
import math
import pytest

from src.shared.python.biomechanics.activation_dynamics import ActivationDynamics
from src.shared.python.biomechanics.hill_muscle import HillMuscleModel, MuscleParameters
from src.shared.python.biomechanics.muscle_allocation import (
    MuscleRecruitmentPath,
    allocate_muscle_excitations,
)
from src.shared.python.biomechanics.muscle_equilibrium import EquilibriumSolver
from src.shared.python.biomechanics import muscle_allocation

pytestmark = pytest.mark.unit


def _path(
    name: str,
    arms: dict[str, float],
    *,
    prior: float = 0.2,
    rate: float = 1000.0,
    pennation: float = 0.0,
) -> MuscleRecruitmentPath:
    model = HillMuscleModel(
        MuscleParameters(
            F_max=100.0,
            l_opt=0.1,
            l_slack=0.2,
            damping=0.0,
            pennation_angle=pennation,
        )
    )
    return MuscleRecruitmentPath(
        name=name,
        muscle=model,
        activation_dynamics=ActivationDynamics(),
        muscle_tendon_length_m=0.31,
        moment_arms_m_or_rad=arms,
        activation=prior,
        previous_excitation=prior,
        excitation_bounds=(0.0, 1.0),
        max_excitation_rate_per_s=rate,
    )


def test_baseline_candidate_is_retained_when_budget_is_one() -> None:
    dt = 0.01
    path = replace(
        _path("baseline", {"hip": 0.05}, prior=0.0, rate=100.0),
        activation=0.1,
        activation_dynamics=ActivationDynamics(
            tau_act=0.015, tau_deact=0.06, min_activation=0.01
        ),
    )
    activation = path.activation_dynamics.update(0.0, path.activation, dt)
    fiber = EquilibriumSolver(path.muscle).solve_fiber_length(
        path.muscle_tendon_length_m, activation
    )
    tendon_length = path.muscle_tendon_length_m - fiber
    tendon_force = path.muscle.params.F_max * path.muscle.tendon_force(
        tendon_length / path.muscle.params.l_slack
    )
    target = path.moment_arms_m_or_rad["hip"] * tendon_force

    result = allocate_muscle_excitations(
        (path,),
        ("hip",),
        (target,),
        coordinate_effort_units=("N*m",),
        dt_s=dt,
        effort_tolerances=(1e-6,),
        force_equilibrium_tolerance_n=1e-4,
        max_nfev=1,
    )

    assert result.status == "budget_exhausted"
    assert not result.feasible
    assert result.excitations["baseline"] == pytest.approx(0.0)
    assert result.generalized_effort["hip"] == pytest.approx(target, abs=1e-6)
    assert result.nfev == 1


def test_floor_plateau_seed_finds_feasible_above_floor_candidate() -> None:
    dt = 0.01
    paths = (
        replace(
            _path("flexor", {"hip": 0.05}, prior=0.0, rate=100.0),
            activation=0.1,
        ),
        replace(
            _path("extensor", {"hip": -0.05}, prior=0.0, rate=100.0),
            activation=0.1,
        ),
    )
    witness = (0.15, 0.05)
    target = 0.0
    for path, raw in zip(paths, witness, strict=True):
        activation = path.activation_dynamics.update(raw, path.activation, dt)
        fiber = EquilibriumSolver(path.muscle).solve_fiber_length(
            path.muscle_tendon_length_m, activation
        )
        tendon_length = path.muscle_tendon_length_m - fiber
        force = path.muscle.params.F_max * path.muscle.tendon_force(
            tendon_length / path.muscle.params.l_slack
        )
        target += path.moment_arms_m_or_rad["hip"] * force

    assert target > 0.0
    assert all(
        raw > path.activation_dynamics.min_activation
        for path, raw in zip(paths, witness, strict=True)
    )
    result = allocate_muscle_excitations(
        paths,
        ("hip",),
        (target,),
        coordinate_effort_units=("N*m",),
        dt_s=dt,
        effort_tolerances=(1e-6,),
        force_equilibrium_tolerance_n=1e-4,
        max_nfev=80,
    )

    assert result.feasible
    assert result.status == "feasible"
    assert result.generalized_effort["hip"] == pytest.approx(target, abs=1e-6)
    for path in paths:
        raw = result.excitations[path.name]
        assert (
            result.effective_bounds[path.name][0]
            <= raw
            <= result.effective_bounds[path.name][1]
        )
        assert result.effective_excitations[path.name] == pytest.approx(
            max(raw, path.activation_dynamics.min_activation)
        )
    assert result.nfev > 1
    assert result.nfev <= 80
    assert any(
        result.excitations[path.name] > path.activation_dynamics.min_activation
        for path in paths
    )


def test_floor_seed_respects_slew_upper_bound_below_floor() -> None:
    path = replace(
        _path("slew-limited", {"hip": 0.05}, prior=0.0, rate=0.5),
        activation=0.1,
        activation_dynamics=ActivationDynamics(
            tau_act=0.015, tau_deact=0.06, min_activation=0.01
        ),
    )
    result = allocate_muscle_excitations(
        (path,),
        ("hip",),
        (1.0,),
        coordinate_effort_units=("N*m",),
        dt_s=0.01,
        effort_tolerances=(1e-6,),
        max_nfev=12,
    )

    assert result.status == "infeasible"
    assert not result.feasible
    assert result.effective_bounds["slew-limited"] == pytest.approx((0.0, 0.005))
    assert result.excitations["slew-limited"] == pytest.approx(0.0)
    assert result.nfev <= 12


def test_fixed_floor_control_preserves_raw_excitation() -> None:
    path = replace(
        _path("fixed-floor", {"hip": 0.05}, prior=0.005, rate=0.0),
        activation=0.1,
        activation_dynamics=ActivationDynamics(
            tau_act=0.015, tau_deact=0.06, min_activation=0.01
        ),
    )
    result = allocate_muscle_excitations(
        (path,),
        ("hip",),
        (0.0,),
        coordinate_effort_units=("N*m",),
        dt_s=0.01,
        effort_tolerances=(1.0,),
        force_equilibrium_tolerance_n=1.0,
        max_nfev=5,
    )

    assert result.excitations["fixed-floor"] == pytest.approx(0.005)
    assert result.effective_bounds["fixed-floor"] == pytest.approx((0.005, 0.005))
    assert result.effective_excitations["fixed-floor"] == pytest.approx(0.01)
    assert result.nfev == 1


def test_effort_units_and_order_must_match_every_coordinate() -> None:
    with pytest.raises(ValueError, match="match coordinate order"):
        allocate_muscle_excitations(
            [_path("biarticular", {"hip": 0.03, "slide": 1.0})],
            ("hip", "slide"),
            (1.0, 2.0),
            coordinate_effort_units=("N*m",),
            dt_s=0.001,
            effort_tolerances=(0.1, 0.1),
        )


def test_mixed_translation_and_rotation_units_preserve_named_order() -> None:
    result = allocate_muscle_excitations(
        [_path("biarticular", {"hip": 0.03, "slide": 1.0})],
        ("slide", "hip"),
        (0.0, 0.0),
        coordinate_effort_units=("N", "N*m"),
        dt_s=0.001,
        effort_tolerances=(0.1, 0.1),
    )
    assert tuple(result.residual_by_coordinate) == ("slide", "hip")
    assert tuple(result.generalized_effort) == ("slide", "hip")


def test_two_muscles_reproduce_three_coordinate_effort_without_omitting_axis() -> None:
    paths = (
        _path("biarticular", {"hip": 0.03, "knee": -0.02}, prior=0.2),
        _path("knee_only", {"knee": 0.04}, prior=0.3),
    )
    excitations = (0.52, 0.61)
    target_activation = [
        path.activation_dynamics.update(u, path.activation, 0.001)
        for path, u in zip(paths, excitations, strict=True)
    ]
    forces = []
    for path, activation in zip(paths, target_activation, strict=True):
        fiber = EquilibriumSolver(path.muscle).solve_fiber_length(
            path.muscle_tendon_length_m, activation
        )
        tendon_length = path.muscle_tendon_length_m - fiber
        forces.append(
            path.muscle.params.F_max
            * path.muscle.tendon_force(tendon_length / path.muscle.params.l_slack)
        )
    target = (
        0.03 * forces[0],
        -0.02 * forces[0] + 0.04 * forces[1],
        0.0,
    )
    result = allocate_muscle_excitations(
        paths,
        ("hip", "knee", "unactuated"),
        target,
        coordinate_effort_units=("N*m", "N*m", "N*m"),
        dt_s=0.001,
        effort_tolerances=(2e-3, 2e-3, 1e-4),
        force_equilibrium_tolerance_n=1e-4,
    )
    assert result.feasible
    assert result.excitations["biarticular"] == pytest.approx(excitations[0], abs=2e-3)
    assert result.excitations["knee_only"] == pytest.approx(excitations[1], abs=2e-3)
    assert tuple(result.residual_by_coordinate) == ("hip", "knee", "unactuated")


def test_one_step_uses_activation_lag_and_public_force_components() -> None:
    path = _path("biarticular", {"hip": 0.03, "knee": -0.02})
    dt = 0.001
    target_excitation = 0.55
    activation = path.activation_dynamics.update(target_excitation, path.activation, dt)
    fiber = EquilibriumSolver(path.muscle).solve_fiber_length(
        path.muscle_tendon_length_m, activation
    )
    tendon_length = path.muscle_tendon_length_m - fiber
    tendon_force = path.muscle.params.F_max * path.muscle.tendon_force(
        tendon_length / path.muscle.params.l_slack
    )
    target = (0.03 * tendon_force, -0.02 * tendon_force)
    result = allocate_muscle_excitations(
        [path],
        ("hip", "knee"),
        target,
        coordinate_effort_units=("N*m", "N*m"),
        dt_s=dt,
        effort_tolerances=(1e-4, 1e-4),
        force_equilibrium_tolerance_n=1e-4,
    )
    expected_activation = path.activation_dynamics.update(
        result.excitations["biarticular"], path.activation, dt
    )
    assert result.status == "feasible"
    assert result.activations["biarticular"] == pytest.approx(expected_activation)
    assert result.muscle_forces_n["biarticular"]["active_fiber_along_fiber_n"] > 0
    assert result.muscle_forces_n["biarticular"][
        "force_equilibrium_residual_n"
    ] == pytest.approx(0.0, abs=1e-4)
    assert result.muscle_forces_n["biarticular"]["tendon_n"] == pytest.approx(
        result.muscle_forces_n["biarticular"]["fiber_tendon_axis_n"], abs=1e-4
    )
    assert result.generalized_effort["hip"] == pytest.approx(target[0], abs=1e-4)
    assert result.generalized_effort["knee"] == pytest.approx(target[1], abs=1e-4)
    assert path.activation == 0.2


def test_unactuated_coordinate_stays_in_complete_residual() -> None:
    result = allocate_muscle_excitations(
        [_path("hip_knee", {"hip": 0.03, "knee": -0.02})],
        ("hip", "knee", "ankle"),
        (0.4, -0.2, 0.1),
        coordinate_effort_units=("N*m", "N*m", "N*m"),
        dt_s=0.001,
        effort_tolerances=(1e-4, 1e-4, 1e-4),
        max_nfev=30,
    )
    assert result.status == "infeasible"
    assert result.residual_by_coordinate["ankle"] == pytest.approx(-0.1)
    assert not result.feasible


def test_zero_rate_is_a_fixed_channel_not_a_widened_box() -> None:
    path = _path("fixed", {"hip": 0.02}, prior=0.3, rate=0.0)
    result = allocate_muscle_excitations(
        [path],
        ("hip",),
        (0.0,),
        coordinate_effort_units=("N*m",),
        dt_s=0.001,
        effort_tolerances=(1e-3,),
        max_nfev=10,
    )
    assert result.excitations["fixed"] == pytest.approx(0.3)
    assert result.effective_bounds["fixed"] == pytest.approx((0.3, 0.3))


def test_passive_fiber_force_is_retained_as_a_separate_component() -> None:
    path = _path("passive", {"hip": 0.02}, prior=0.001, rate=0.0)
    result = allocate_muscle_excitations(
        [path],
        ("hip",),
        (0.0,),
        coordinate_effort_units=("N*m",),
        dt_s=0.001,
        effort_tolerances=(1.0,),
        force_equilibrium_tolerance_n=1.0,
    )
    assert result.muscle_forces_n["passive"]["passive_fiber_along_fiber_n"] > 0
    assert result.muscle_forces_n["passive"]["damping_fiber_along_fiber_n"] == 0


def test_pennation_projection_is_reported_separately() -> None:
    path = _path("pennated", {"hip": 0.02}, prior=0.3, rate=0.0, pennation=0.2)
    result = allocate_muscle_excitations(
        [path],
        ("hip",),
        (0.0,),
        coordinate_effort_units=("N*m",),
        dt_s=0.001,
        effort_tolerances=(1.0,),
        force_equilibrium_tolerance_n=1.0,
    )
    forces = result.muscle_forces_n["pennated"]
    assert forces["active_tendon_axis_n"] == pytest.approx(
        forces["active_fiber_along_fiber_n"] * math.cos(0.2)
    )
    assert forces["force_equilibrium_residual_n"] == pytest.approx(
        forces["fiber_tendon_axis_n"] - forces["tendon_n"]
    )


def test_rate_cap_bounds_first_requested_excitation_from_last_applied_value() -> None:
    path = _path("rate-limited", {"hip": 0.02}, prior=0.2, rate=0.5)
    result = allocate_muscle_excitations(
        [path],
        ("hip",),
        (100.0,),
        coordinate_effort_units=("N*m",),
        dt_s=0.1,
        effort_tolerances=(0.1,),
    )
    assert result.excitations["rate-limited"] <= 0.2 + 0.5 * 0.1
    assert result.excitations["rate-limited"] >= 0.2 - 0.5 * 0.1
    assert result.effective_bounds["rate-limited"] == pytest.approx((0.15, 0.25))


@pytest.mark.parametrize(
    ("dt", "rate"), [(0.0, 1.0), (-1.0, 1.0), (0.001, -1.0), (float("nan"), 1.0)]
)
def test_invalid_dt_or_rate_is_rejected(dt: float, rate: float) -> None:
    with pytest.raises(ValueError):
        allocate_muscle_excitations(
            [_path("m", {"hip": 0.02}, rate=rate)],
            ("hip",),
            (0.0,),
            coordinate_effort_units=("N*m",),
            dt_s=dt,
            effort_tolerances=(0.1,),
        )


@pytest.mark.parametrize(
    "path",
    [
        _path("reversed", {"hip": 0.02}),
        _path("nonfinite-arm", {"hip": float("nan")}),
    ],
)
def test_invalid_box_or_nonfinite_moment_arm_is_rejected(
    path: MuscleRecruitmentPath,
) -> None:
    if path.name == "reversed":
        path = replace(path, excitation_bounds=(0.8, 0.2))
    with pytest.raises(ValueError):
        allocate_muscle_excitations(
            [path],
            ("hip",),
            (0.0,),
            coordinate_effort_units=("N*m",),
            dt_s=0.001,
            effort_tolerances=(0.1,),
        )


def test_nonfinite_requested_effort_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        allocate_muscle_excitations(
            [_path("m", {"hip": 0.02})],
            ("hip",),
            (float("inf"),),
            coordinate_effort_units=("N*m",),
            dt_s=0.001,
            effort_tolerances=(0.1,),
        )


def test_bad_finite_equilibrium_root_is_not_reported_feasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        EquilibriumSolver, "solve_fiber_length", lambda self, *args, **kwargs: 0.1
    )
    result = allocate_muscle_excitations(
        [_path("bad-root", {"hip": 0.02})],
        ("hip",),
        (0.0,),
        coordinate_effort_units=("N*m",),
        dt_s=0.001,
        effort_tolerances=(1.0,),
        force_equilibrium_tolerance_n=1e-5,
        max_nfev=80,
    )
    assert result.status == "infeasible"
    assert not result.feasible


def test_empty_bound_intersection_is_rejected() -> None:
    path = replace(
        _path("empty", {"hip": 0.02}, prior=0.2, rate=0.0), excitation_bounds=(0.5, 0.8)
    )
    with pytest.raises(ValueError, match="empty excitation"):
        allocate_muscle_excitations(
            [path],
            ("hip",),
            (0.0,),
            coordinate_effort_units=("N*m",),
            dt_s=0.001,
            effort_tolerances=(0.1,),
        )


def test_raw_zero_excitation_keeps_slew_box_and_reports_kernel_floor() -> None:
    path = replace(
        _path("at-rest", {"hip": 0.02}, prior=0.0, rate=0.0),
        activation=0.0,
        previous_excitation=0.0,
    )
    result = allocate_muscle_excitations(
        [path],
        ("hip",),
        (0.0,),
        coordinate_effort_units=("N*m",),
        dt_s=0.001,
        effort_tolerances=(1.0,),
    )
    assert result.excitations["at-rest"] == pytest.approx(0.0)
    assert result.effective_bounds["at-rest"] == pytest.approx((0.0, 0.0))
    assert result.effective_excitations["at-rest"] == pytest.approx(
        path.activation_dynamics.min_activation
    )
    assert result.activations["at-rest"] == pytest.approx(
        path.activation_dynamics.min_activation
    )


def test_optimizer_budget_exhaustion_never_claims_feasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BudgetResult:
        success = False
        status = 0
        nfev = 3
        message = "maximum function evaluations reached"

    equilibrium_calls = 0
    original_solve = EquilibriumSolver.solve_fiber_length

    def count_solve(self: EquilibriumSolver, *args: object, **kwargs: object) -> float:
        nonlocal equilibrium_calls
        equilibrium_calls += 1
        return original_solve(self, *args, **kwargs)

    def finite_difference_probes(
        objective: object, x0: list[float], **kwargs: object
    ) -> BudgetResult:
        for point in (x0, [x0[0] + 1e-6], [x0[0] - 1e-6]):
            try:
                objective(point)
            except RuntimeError:
                break
        return BudgetResult()

    monkeypatch.setattr(EquilibriumSolver, "solve_fiber_length", count_solve)
    monkeypatch.setattr(muscle_allocation, "least_squares", finite_difference_probes)
    result = allocate_muscle_excitations(
        [_path("bounded", {"hip": 0.02})],
        ("hip",),
        (0.0,),
        coordinate_effort_units=("N*m",),
        dt_s=0.001,
        effort_tolerances=(1e3,),
        max_nfev=3,
    )
    assert result.status == "budget_exhausted"
    assert not result.feasible
    assert result.nfev == 3
    assert equilibrium_calls == 3


def test_optimizer_failure_is_distinct_from_infeasible_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SolverFailure:
        success = False
        status = -1
        nfev = 1
        message = "improper input parameters"

    monkeypatch.setattr(
        muscle_allocation,
        "least_squares",
        lambda *args, **kwargs: SolverFailure(),
    )
    result = allocate_muscle_excitations(
        [_path("solver-failure", {"hip": 0.02})],
        ("hip",),
        (0.0,),
        coordinate_effort_units=("N*m",),
        dt_s=0.001,
        effort_tolerances=(1e3,),
    )
    assert result.status == "solver_failed"
    assert not result.feasible
    assert "solver-failure" in result.excitations
