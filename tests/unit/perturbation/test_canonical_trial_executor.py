"""Tests for deterministic serial execution of canonical Tools plans."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python import perturbation
from src.shared.python.perturbation.canonical_trial_executor import (
    execute_serial_variation,
)
from src.shared.python.perturbation.trial_evidence import (
    CanonicalTrialEvidence,
    ClosestApproach,
    SampledInput,
    TrialTrace,
)

pytestmark = pytest.mark.unit

_PLAN_SHA = "a" * 64
_TOOLS_REVISION = "b" * 40
_ENGINE_REVISION = "c" * 40


def test_serial_executor_is_exposed_by_public_perturbation_package() -> None:
    assert perturbation.execute_serial_variation is execute_serial_variation


def _trace() -> TrialTrace:
    return TrialTrace(
        times_s=np.array([0.0, 0.01]),
        q=np.zeros((2, 1)),
        v=np.zeros((2, 1)),
        coordinate_ids=("club_angle",),
        coordinate_units=("rad",),
        velocity_units=("rad/s",),
        markers_m=np.zeros((2, 1, 3)),
        marker_ids=("clubhead",),
        frame_id="world-z-up",
        alignment_id="downswing-start/v1",
        complete=True,
    )


class _Gateway:
    def __init__(self, samples: np.ndarray) -> None:
        self.samples = samples

    def sample_inputs(self, _plan: object) -> np.ndarray:
        return self.samples


class _Collector:
    def collect_success(
        self,
        trial_index: int,
        plan_seed: int,
        sampled_row: np.ndarray,
        result: object,
    ) -> CanonicalTrialEvidence:
        assert result == "no impact"
        return self._record(
            trial_index,
            plan_seed,
            sampled_row,
            outcome="no_impact",
            failure_reason=None,
            trace=_trace(),
        )

    def collect_failure(
        self,
        trial_index: int,
        plan_seed: int,
        sampled_row: np.ndarray,
        error: Exception,
    ) -> CanonicalTrialEvidence:
        return self._record(
            trial_index,
            plan_seed,
            sampled_row,
            outcome="numerical_failure",
            failure_reason=f"{type(error).__name__}: {error}",
            trace=None,
        )

    @staticmethod
    def _record(
        trial_index: int,
        plan_seed: int,
        sampled_row: np.ndarray,
        *,
        outcome: str,
        failure_reason: str | None,
        trace: TrialTrace | None,
    ) -> CanonicalTrialEvidence:
        closest = (
            ClosestApproach(0.01, 0.02, "clubhead", "ball-center", False)
            if outcome == "no_impact"
            else None
        )
        return CanonicalTrialEvidence(
            trial_index=trial_index,
            seed=plan_seed,
            plan_sha256=_PLAN_SHA,
            tools_revision=_TOOLS_REVISION,
            engine_id="pendulum",
            engine_revision=_ENGINE_REVISION,
            model_id="two-dof-double-pendulum/v1",
            sampled_inputs=tuple(
                SampledInput(f"input_{index}", float(value), "1")
                for index, value in enumerate(sampled_row)
            ),
            outcome=outcome,  # type: ignore[arg-type]
            trace=trace,
            closest_approach=closest,
            failure_reason=failure_reason,
        )


def test_serial_executor_retains_miss_and_numerical_failure_rows_in_order() -> None:
    plan = SimpleNamespace(n_runs=2, seed=17)
    gateway = _Gateway(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def runner(row: np.ndarray) -> str:
        assert row.flags.writeable is False
        if row[0] == 3.0:
            raise FloatingPointError("non-finite acceleration")
        return "no impact"

    records = execute_serial_variation(plan, gateway, runner, _Collector())

    assert tuple(record.trial_index for record in records) == (0, 1)
    assert tuple(record.outcome for record in records) == (
        "no_impact",
        "numerical_failure",
    )
    assert records[1].failure_reason == ("FloatingPointError: non-finite acceleration")


def test_serial_executor_replays_identical_samples_and_records() -> None:
    plan = SimpleNamespace(n_runs=2, seed=17)
    gateway = _Gateway(np.array([[1.0], [2.0]]))
    collector = _Collector()

    first = execute_serial_variation(plan, gateway, lambda _row: "no impact", collector)
    second = execute_serial_variation(
        plan, gateway, lambda _row: "no impact", collector
    )

    assert [record.sampled_inputs for record in first] == [
        record.sampled_inputs for record in second
    ]
    assert [record.outcome for record in first] == [record.outcome for record in second]


@pytest.mark.parametrize(
    "samples",
    [
        np.array([1.0, 2.0]),
        np.array([[1.0], [2.0], [3.0]]),
        np.array([[1.0], [np.nan]]),
        np.empty((2, 0)),
    ],
)
def test_serial_executor_rejects_invalid_tools_sample_matrix(
    samples: np.ndarray,
) -> None:
    plan = SimpleNamespace(n_runs=2, seed=17)

    with pytest.raises(ValueError, match="sample matrix"):
        execute_serial_variation(
            plan,
            _Gateway(samples),
            lambda _row: "no impact",
            _Collector(),
        )


def test_serial_executor_does_not_relabel_programming_errors_as_trial_data() -> None:
    plan = SimpleNamespace(n_runs=1, seed=17)

    def broken_runner(_row: np.ndarray) -> object:
        raise TypeError("adapter contract bug")

    with pytest.raises(TypeError, match="adapter contract bug"):
        execute_serial_variation(
            plan,
            _Gateway(np.array([[1.0]])),
            broken_runner,
            _Collector(),
        )


def test_serial_executor_rejects_collector_identity_drift() -> None:
    plan = SimpleNamespace(n_runs=1, seed=17)

    class WrongIndexCollector(_Collector):
        def collect_success(
            self,
            trial_index: int,
            plan_seed: int,
            sampled_row: np.ndarray,
            result: object,
        ) -> CanonicalTrialEvidence:
            return super().collect_success(
                trial_index + 1, plan_seed, sampled_row, result
            )

    with pytest.raises(ValueError, match="trial identity"):
        execute_serial_variation(
            plan,
            _Gateway(np.array([[1.0]])),
            lambda _row: "no impact",
            WrongIndexCollector(),
        )
