from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.perturbation.trial_adapter_contracts import (
    TrialEvidenceIdentity,
    collect_trial_failure,
    make_trial_evidence_identity,
    require_fixed_step_horizon,
    require_localized_time_window,
    require_plan_execution_identity,
    require_trial_result_geometry,
    require_trace_index,
    sampled_inputs_from_row,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _Column:
    key: str
    unit: str


def test_fixed_step_horizon_requires_positive_integral_step_count() -> None:
    assert require_fixed_step_horizon(0.1, 0.01) == 10

    for duration, dt in ((0.0, 0.01), (0.1, 0.0), (0.1, 0.03)):
        with pytest.raises(ValueError):
            require_fixed_step_horizon(duration, dt)


def test_localized_time_window_is_normalized_and_bounded() -> None:
    assert require_localized_time_window((0, 0.25), 0.5) == (0.0, 0.25)

    for window in (None, (0.2,), (-0.1, 0.2), (0.2, 0.2), (0.2, 0.6)):
        with pytest.raises(ValueError):
            require_localized_time_window(window, 0.5)


def test_trace_index_uses_one_shared_bounds_contract() -> None:
    trace = SimpleNamespace(t=np.array([0.0, 0.1, 0.2]))
    assert require_trace_index(trace, 2, "closest_sample_index") == 2
    assert require_trace_index(trace, None, "contact_index", allow_none=True) is None

    with pytest.raises(ValueError, match="closest_sample_index"):
        require_trace_index(trace, 3, "closest_sample_index")
    with pytest.raises(ValueError, match="contact_index"):
        require_trace_index(trace, None, "contact_index")


def test_sampled_inputs_preserve_column_order_units_and_finite_values() -> None:
    columns = (_Column("a", "N·m"), _Column("b", "rad"))
    inputs = sampled_inputs_from_row(np.array([1.5, -0.2]), columns)

    assert tuple((item.name, item.value, item.unit) for item in inputs) == (
        ("a", 1.5, "N·m"),
        ("b", -0.2, "rad"),
    )
    with pytest.raises(ValueError, match="plan columns"):
        sampled_inputs_from_row(np.array([1.5]), columns)
    with pytest.raises(ValueError, match="finite"):
        sampled_inputs_from_row(np.array([1.5, np.nan]), columns)


def test_plan_execution_identity_rejects_bool_and_nonpositive_counts() -> None:
    plan = SimpleNamespace(n_runs=4, seed=11)
    assert require_plan_execution_identity(plan) == (4, 11)

    for n_runs, seed in ((True, 11), (0, 11), (4, True), (4, 1.5)):
        with pytest.raises((TypeError, ValueError)):
            require_plan_execution_identity(SimpleNamespace(n_runs=n_runs, seed=seed))


def test_evidence_identity_builds_typed_failure_without_fabricated_outputs() -> None:
    identity = TrialEvidenceIdentity(
        plan_sha256="a" * 64,
        scenario_sha256="b" * 64,
        execution_config_sha256="c" * 64,
        tools_revision="d" * 40,
        engine_id="engine.test",
        engine_revision="e" * 40,
        model_id="model.test",
    )
    sampled = sampled_inputs_from_row(np.array([1.0]), (_Column("a", "N·m"),))

    record = identity.failure(
        trial_index=2,
        seed=11,
        sampled_inputs=sampled,
        error=FloatingPointError("diverged"),
    )

    assert record.outcome == "numerical_failure"
    assert record.trace is None
    assert record.impact is None
    assert record.shot_result is None
    assert record.failure_reason == "FloatingPointError: diverged"


def test_result_geometry_contract_rejects_invalid_distance() -> None:
    trace = SimpleNamespace(t=np.array([0.0, 0.1]))
    require_trial_result_geometry(trace, 1, 0.25)
    with pytest.raises(ValueError, match="finite and non-negative"):
        require_trial_result_geometry(trace, 1, -0.25)


def test_shared_failure_collector_retains_identity_and_inputs() -> None:
    identity = make_trial_evidence_identity(
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 40,
        "engine.test",
        "e" * 40,
        "model.test",
    )
    columns = (_Column("torque", "N*m"),)
    failure = collect_trial_failure(
        identity,
        2,
        17,
        np.array([3.0]),
        columns,
        RuntimeError("boom"),
    )
    assert failure.outcome == "numerical_failure"
    assert failure.sampled_inputs[0].value == 3.0
    assert failure.failure_reason == "RuntimeError: boom"
