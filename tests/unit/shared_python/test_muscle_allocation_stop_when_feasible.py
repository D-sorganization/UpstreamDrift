"""Opt-in ``stop_when_feasible`` contract for bounded muscle allocation (#13256).

Reviewed contract from the F08/#12399 handoff: stopping happens during a counted
evaluation, never by relabelling budget exhaustion; the reported candidate is the
exact feasible witness, not the lowest-L2 candidate.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from src.shared.python.biomechanics import muscle_allocation
from src.shared.python.biomechanics.activation_dynamics import ActivationDynamics
from src.shared.python.biomechanics.hill_muscle import HillMuscleModel, MuscleParameters
from src.shared.python.biomechanics.muscle_allocation import (
    MuscleRecruitmentPath,
    _bounds,
    _Evaluation,
    _ExcitationSearcher,
    _parse_allocation_request,
    allocate_muscle_excitations,
)

pytestmark = pytest.mark.unit

DT = 0.01
Factory = Callable[[MuscleRecruitmentPath, float], Callable[[float], _Evaluation]]


def _path(
    name: str, arms: dict[str, float], *, prior: float = 0.2
) -> MuscleRecruitmentPath:
    model = HillMuscleModel(
        MuscleParameters(
            F_max=100.0, l_opt=0.1, l_slack=0.2, damping=0.0, pennation_angle=0.0
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
        max_excitation_rate_per_s=1000.0,
    )


def _effort_at_prior(path: MuscleRecruitmentPath, coordinate: str) -> float:
    result = allocate_muscle_excitations(
        (path,),
        (coordinate,),
        (0.0,),
        coordinate_effort_units=("N*m",),
        dt_s=DT,
        effort_tolerances=(1e9,),
        max_nfev=1,
    )
    return result.generalized_effort[coordinate]


def _forbid_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("SciPy must not be invoked")

    monkeypatch.setattr(muscle_allocation, "least_squares", boom)


def test_default_mode_never_reports_the_feasibility_stop_reason() -> None:
    path = _path("m", {"hip": 0.05})
    target = _effort_at_prior(path, "hip")
    result = allocate_muscle_excitations(
        (path,),
        ("hip",),
        (target,),
        coordinate_effort_units=("N*m",),
        dt_s=DT,
        effort_tolerances=(1e-6,),
    )
    assert result.termination_reason != "declared_feasibility_satisfied"


def test_feasible_initial_candidate_returns_at_nfev_one_without_scipy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _path("m", {"hip": 0.05})
    target = _effort_at_prior(path, "hip")
    _forbid_scipy(monkeypatch)
    result = allocate_muscle_excitations(
        (path,),
        ("hip",),
        (target,),
        coordinate_effort_units=("N*m",),
        dt_s=DT,
        effort_tolerances=(1e-6,),
        stop_when_feasible=True,
    )
    assert result.nfev == 1
    assert result.feasible
    assert result.status == "feasible"
    assert result.termination_reason == "declared_feasibility_satisfied"
    assert result.excitations["m"] == pytest.approx(path.previous_excitation)


def test_stop_when_feasible_requires_a_bool() -> None:
    path = _path("m", {"hip": 0.05})
    with pytest.raises(ValueError, match="stop_when_feasible"):
        allocate_muscle_excitations(
            (path,),
            ("hip",),
            (0.0,),
            coordinate_effort_units=("N*m",),
            dt_s=DT,
            effort_tolerances=(1.0,),
            stop_when_feasible=1,  # type: ignore[arg-type]
        )


# --- scripted-optimizer scenarios -------------------------------------------------


def _fake_factory(gain: float, equilibrium_n: float = 0.0) -> Factory:
    """Tendon force = gain * excitation; equilibrium residual is constant."""

    def factory(
        _path_: MuscleRecruitmentPath, _dt: float
    ) -> Callable[[float], _Evaluation]:
        def evaluate(excitation: float) -> _Evaluation:
            return _Evaluation(
                activation=excitation,
                effective_excitation=excitation,
                components={"tendon_n": gain * excitation},
                tendon_force=gain * excitation,
                equilibrium_residual_n=equilibrium_n,
            )

        return evaluate

    return factory


def _searcher(
    paths: tuple[MuscleRecruitmentPath, ...],
    coords: tuple[str, ...],
    *,
    max_nfev: int,
    stop: bool,
    factory: Factory | None = None,
    tol: float = 1.0,
) -> _ExcitationSearcher:
    request = _parse_allocation_request(
        coords,
        [0.0] * len(coords),
        ["N*m"] * len(coords),
        [tol] * len(coords),
        DT,
        1e-4,
        max_nfev,
    )
    boxes = [_bounds(p, DT) for p in paths]
    return _ExcitationSearcher(
        paths,
        coords,
        request,
        boxes,
        factory or _fake_factory(10.0),
        stop_when_feasible=stop,
    )


def _script(monkeypatch: pytest.MonkeyPatch, points: list[list[float]]) -> None:
    """Replace SciPy with a fake that evaluates scripted free-channel points."""

    def fake(objective: Any, x0: Any, **_kwargs: Any) -> OptimizeResult:
        x = np.asarray(x0)
        for point in points:
            x = np.asarray(point)
            objective(x)
        return OptimizeResult(
            x=x, success=True, status=1, message="scripted", nfev=len(points)
        )

    monkeypatch.setattr(muscle_allocation, "least_squares", fake)


def _two_channel() -> tuple[tuple[MuscleRecruitmentPath, ...], tuple[str, ...]]:
    # Prior 0.11 on "a" gives normalized residual 1.1 (gain 10, tol 1); "b" is 0.
    pa = replace(_path("a", {"ca": 1.0}, prior=0.11), activation=0.11)
    pb = replace(_path("b", {"cb": 1.0}, prior=0.0), activation=0.0)
    return (pa, pb), ("ca", "cb")


def test_lower_l2_infeasible_candidate_never_replaces_the_feasible_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, coords = _two_channel()
    # Initial residuals (1.1, 0): L2 1.1, infeasible. Probe (0.08, 0.08) gives
    # (0.8, 0.8): L2 1.13 (higher) but feasible, so it is the witness.
    _script(monkeypatch, [[0.08, 0.08], [0.0, 0.0]])
    out = _searcher(paths, coords, max_nfev=10, stop=True).run()
    assert out.declared_feasibility_satisfied
    assert out.evaluation_count == 2
    assert out.best_candidate is not None
    assert out.best_candidate.tolist() == pytest.approx([0.08, 0.08])
    # Default mode keeps the lowest-L2 candidate (the infeasible initial one).
    _script(monkeypatch, [[0.08, 0.08]])
    default = _searcher(paths, coords, max_nfev=10, stop=False).run()
    assert not default.declared_feasibility_satisfied
    assert default.best_candidate is not None
    assert default.best_candidate.tolist() == pytest.approx([0.11, 0.0])


def test_no_evaluation_happens_after_the_feasible_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, coords = _two_channel()
    _script(monkeypatch, [[0.08, 0.08], [0.0, 0.0], [0.0, 0.0]])
    out = _searcher(paths, coords, max_nfev=10, stop=True).run()
    assert out.evaluation_count == 2


def test_feasible_final_permitted_candidate_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, coords = _two_channel()
    _script(monkeypatch, [[0.08, 0.08]])
    out = _searcher(paths, coords, max_nfev=2, stop=True).run()
    assert out.declared_feasibility_satisfied
    assert out.evaluation_count == 2
    assert not out.evaluation_budget_exhausted


def test_infeasible_final_candidate_stays_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, coords = _two_channel()
    # The second probe would exceed the budget of 2 and is never evaluated.
    _script(monkeypatch, [[0.5, 0.5], [0.08, 0.08]])
    out = _searcher(paths, coords, max_nfev=2, stop=True).run()
    assert not out.declared_feasibility_satisfied
    assert out.evaluation_budget_exhausted
    assert out.evaluation_count == 2


def test_one_failed_coordinate_prevents_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, coords = _two_channel()
    _script(monkeypatch, [[0.08, 0.12]])  # b residual 1.2 > tolerance 1
    out = _searcher(paths, coords, max_nfev=2, stop=True).run()
    assert not out.declared_feasibility_satisfied


def test_one_failed_equilibrium_prevents_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, coords = _two_channel()
    _script(monkeypatch, [[0.08, 0.08]])
    out = _searcher(
        paths,
        coords,
        max_nfev=5,
        stop=True,
        factory=_fake_factory(10.0, equilibrium_n=1.0),  # 1 N >> 1e-4 N
    ).run()
    assert not out.declared_feasibility_satisfied


def test_nonfinite_output_prevents_acceptance() -> None:
    paths, coords = _two_channel()
    out = _searcher(
        paths,
        coords,
        max_nfev=1,
        stop=True,
        factory=_fake_factory(float("nan")),
    ).run()
    assert not out.declared_feasibility_satisfied


def test_candidate_outside_the_declared_box_is_not_a_witness() -> None:
    paths, coords = _two_channel()
    searcher = _searcher(paths, coords, max_nfev=3, stop=True)
    inside = searcher._within_declared_box(np.array([0.01, 0.01]))
    outside = searcher._within_declared_box(np.array([0.01, -0.5]))
    assert inside
    assert not outside


def test_mixed_fixed_and_free_channels_keep_raw_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = replace(_path("a", {"ca": 1.0}, prior=0.05), excitation_bounds=(0.05, 0.05))
    pb = replace(_path("b", {"cb": 1.0}, prior=0.2), activation=0.2)
    paths, coords = (pa, pb), ("ca", "cb")
    _script(monkeypatch, [[0.08]])  # only "b" is free
    out = _searcher(paths, coords, max_nfev=5, stop=True).run()
    assert out.declared_feasibility_satisfied
    assert out.best_candidate is not None
    assert out.best_candidate.tolist() == pytest.approx([0.05, 0.08])
    assert pa.previous_excitation == 0.05  # prior state is immutable
