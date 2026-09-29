from __future__ import annotations

import numpy as np
import pytest
from src.engines.physics_engines.opensim.python.motion_matching.fit_multistart import (
    AllStartsFailedError,
    MultistartOptions,
    fit_swing_opensim_multistart,
    generate_multistart_seeds,
)
from src.engines.physics_engines.opensim.python.motion_matching.fit_swing import (
    FitOptions,
    FitResult,
)
from src.shared.python.motion_matching.club_target import ClubTarget, SourceProvenance


def _target() -> ClubTarget:
    time = np.array([0.0, 0.01, 0.02], dtype=np.float64)
    butt = np.column_stack((time, np.zeros_like(time), np.zeros_like(time)))
    clubhead = butt + np.array([0.0, 0.0, 1.0])
    club_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (time.size, 1))
    return ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=club_quat,
        impact_idx=2,
        source=SourceProvenance(
            filename="unit.csv",
            format="synthetic",
            subject_id="unit",
            trial_id="multistart",
            sha256="0" * 64,
        ),
    )


def _result(
    *,
    theta_value: float,
    final_cost: float,
    history: tuple[float, ...] = (1.0,),
    iterations: int = 3,
    n_evaluations: int = 5,
    solver_status: str = "success",
) -> FitResult:
    return FitResult(
        theta_optimal=np.full(7, theta_value, dtype=np.float64),
        final_cost=final_cost,
        final_rmse_m=final_cost / 10.0,
        solver_status=solver_status,
        iterations=iterations,
        n_evaluations=n_evaluations,
        wall_clock_s=0.01,
        message=f"cost={final_cost}",
        history=history,
        method="fake",
        git_commit="unit",
        engine_version="unit",
        target_hash="unit",
        timestamp_utc="unit",
    )


def test_generate_multistart_seeds_is_ordered_and_deterministic() -> None:
    assert generate_multistart_seeds(n_starts=4, base_seed=123) == (
        66316749,
        2930678937,
        2546691363,
        231159515,
    )
    assert generate_multistart_seeds(n_starts=4, base_seed=123) == (
        66316749,
        2930678937,
        2546691363,
        231159515,
    )


def test_multistart_returns_lowest_cost_result_and_preserves_fields() -> None:
    target = _target()
    by_seed = {
        66316749: _result(
            theta_value=1.0,
            final_cost=9.0,
            history=(9.0,),
            iterations=11,
            n_evaluations=13,
            solver_status="warning",
        ),
        2930678937: _result(
            theta_value=2.0,
            final_cost=1.5,
            history=(5.0, 1.5),
            iterations=17,
            n_evaluations=19,
            solver_status="success",
        ),
    }

    def runner(target: ClubTarget, options: FitOptions) -> FitResult:
        return by_seed[options.rng_seed]

    best = fit_swing_opensim_multistart(
        target,
        fit_options=FitOptions(n_joints=1),
        multistart_options=MultistartOptions(n_starts=2, base_seed=123),
        start_runner=runner,
    )

    np.testing.assert_allclose(best.theta_optimal, np.full(7, 2.0))
    assert best.final_cost == 1.5
    assert best.history == (5.0, 1.5)
    assert best.iterations == 17
    assert best.n_evaluations == 19
    assert best.solver_status == "success"
    assert best.meta["multistart"]["best_start_index"] == 1


def test_multistart_returns_success_with_failed_start_diagnostics() -> None:
    target = _target()

    def runner(target: ClubTarget, options: FitOptions) -> FitResult:
        if options.rng_seed == 66316749:
            raise RuntimeError("start exploded")
        return _result(theta_value=3.0, final_cost=2.0)

    best = fit_swing_opensim_multistart(
        target,
        fit_options=FitOptions(n_joints=1),
        multistart_options=MultistartOptions(n_starts=2, base_seed=123),
        start_runner=runner,
    )

    multistart = best.meta["multistart"]
    assert best.final_cost == 2.0
    assert multistart["failed_start_count"] == 1
    assert multistart["failures"] == [
        {
            "start_index": 0,
            "seed": 66316749,
            "error_type": "RuntimeError",
            "message": "start exploded",
        }
    ]


def test_multistart_raises_typed_error_when_all_starts_fail() -> None:
    target = _target()

    def runner(target: ClubTarget, options: FitOptions) -> FitResult:
        raise ValueError(f"bad seed {options.rng_seed}")

    with pytest.raises(AllStartsFailedError, match="2 OpenSim multistart attempts"):
        fit_swing_opensim_multistart(
            target,
            fit_options=FitOptions(n_joints=1),
            multistart_options=MultistartOptions(n_starts=2, base_seed=123),
            start_runner=runner,
        )


def test_multistart_uses_fresh_simulator_factory_per_start() -> None:
    target = _target()
    calls: list[tuple[int, int]] = []
    seen_simulators: list[object] = []

    def simulator_factory(start_index: int, seed: int):
        token = object()
        calls.append((start_index, seed))
        seen_simulators.append(token)

        def simulate(theta: np.ndarray) -> object:
            return token

        return simulate

    def runner(target: ClubTarget, options: FitOptions) -> FitResult:
        assert options.simulate_fn is not None
        return _result(
            theta_value=float(len(seen_simulators)),
            final_cost=float(options.rng_seed % 10),
        )

    fit_swing_opensim_multistart(
        target,
        fit_options=FitOptions(n_joints=1),
        multistart_options=MultistartOptions(n_starts=3, base_seed=123),
        start_runner=runner,
        simulate_factory=simulator_factory,
    )

    assert calls == [
        (0, 66316749),
        (1, 2930678937),
        (2, 2546691363),
    ]
    assert len(set(map(id, seen_simulators))) == 3


def test_multistart_validates_options() -> None:
    with pytest.raises(ValueError, match="n_starts"):
        MultistartOptions(n_starts=0)
    with pytest.raises(ValueError, match="base_seed"):
        MultistartOptions(base_seed=-1)
    with pytest.raises(ValueError, match="max_workers"):
        MultistartOptions(max_workers=0)
