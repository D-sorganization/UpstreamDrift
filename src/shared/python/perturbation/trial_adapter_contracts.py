"""Shared contracts for canonical engine trial adapters.

Engine adapters own model-specific execution and observation mappings. This
module owns the small invariant operations that must remain identical across
those adapters: fixed-step horizons, localized windows, trace-index bounds,
sampled-input rows, plan identity, and immutable evidence identity fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .trial_evidence import (
    CanonicalTrialEvidence,
    ClosestApproach,
    ImpactObservation,
    SampledInput,
    TrialOutcome,
    TrialTrace,
)


class SampledColumn(Protocol):
    """Narrow ordered plan-column surface required for retained inputs."""

    @property
    def key(self) -> str:
        """Return the canonical variable identifier."""
        ...

    @property
    def unit(self) -> str:
        """Return the canonical variable unit."""
        ...


class TimedTrace(Protocol):
    """Narrow trace surface required for event-index validation."""

    t: np.ndarray


def require_fixed_step_horizon(duration_s: float, dt_s: float) -> int:
    """Return a positive integral step count or fail at the config boundary."""
    duration = float(duration_s)
    step = float(dt_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be positive and finite")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be positive and finite")
    step_count = duration / step
    if not math.isclose(step_count, round(step_count), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("duration_s must contain an integer number of steps")
    return round(step_count)


def require_localized_time_window(
    window: object,
    duration_s: float,
) -> tuple[float, float]:
    """Normalize one half-open time window bounded by the trial duration."""
    if not isinstance(window, tuple) or len(window) != 2:
        raise ValueError("localized torque requires one half-open time window")
    if not all(type(value) in (int, float) for value in window):
        raise ValueError("localized torque time window must be finite")
    start_s, end_s = (float(value) for value in window)
    if not math.isfinite(start_s) or not math.isfinite(end_s):
        raise ValueError("localized torque time window must be finite")
    if start_s < 0.0 or start_s >= end_s:
        raise ValueError("localized torque time window must satisfy 0 <= start < end")
    duration = float(duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be positive and finite")
    if end_s > duration:
        raise ValueError("localized torque time window exceeds trial duration")
    return start_s, end_s


def require_trace_index(
    trace: TimedTrace,
    index: int | None,
    label: str,
    *,
    allow_none: bool = False,
) -> int | None:
    """Validate an optional integer event index against a retained trace."""
    if index is None:
        if allow_none:
            return None
        raise ValueError(f"{label} must identify a trace sample")
    time_values = trace.t
    sample_count = int(time_values.size)
    if type(index) is not int or not 0 <= index < sample_count:
        raise ValueError(f"{label} must identify a trace sample")
    return index


def require_trial_result_geometry(
    trace: TimedTrace,
    closest_sample_index: int,
    closest_distance_m: float,
) -> None:
    """Validate the geometry fields common to raw adapter results."""
    require_trace_index(trace, closest_sample_index, "closest_sample_index")
    if not math.isfinite(closest_distance_m) or closest_distance_m < 0.0:
        raise ValueError("closest_distance_m must be finite and non-negative")


def sampled_inputs_from_row(
    sampled_row: np.ndarray,
    columns: Sequence[SampledColumn],
) -> tuple[SampledInput, ...]:
    """Retain one finite scalar per ordered canonical plan column."""
    row = np.asarray(sampled_row, dtype=float).reshape(-1)
    if row.shape != (len(columns),):
        raise ValueError("sampled row does not match plan columns")
    if not np.isfinite(row).all():
        raise ValueError("sampled row values must be finite")
    return tuple(
        SampledInput(column.key, float(value), column.unit)
        for column, value in zip(columns, row, strict=True)
    )


def require_plan_execution_identity(plan: object) -> tuple[int, int]:
    """Return the canonical positive run count and integer seed."""
    n_runs = getattr(plan, "n_runs", None)
    seed = getattr(plan, "seed", None)
    if type(n_runs) is not int or n_runs <= 0:
        raise ValueError("plan n_runs must be a positive integer")
    if type(seed) is not int:
        raise ValueError("plan seed must be an integer")
    return n_runs, seed


@dataclass(frozen=True)
class TrialEvidenceIdentity:
    """Immutable provenance fields shared by every row from one adapter."""

    plan_sha256: str
    scenario_sha256: str
    execution_config_sha256: str
    tools_revision: str
    engine_id: str
    engine_revision: str
    model_id: str

    def build(
        self,
        *,
        trial_index: int,
        seed: int,
        sampled_inputs: tuple[SampledInput, ...],
        outcome: TrialOutcome,
        trace: TrialTrace | None,
        impact: ImpactObservation | None = None,
        closest_approach: ClosestApproach | None = None,
        failure_reason: str | None = None,
    ) -> CanonicalTrialEvidence:
        """Build one evidence row while delegating outcome DbC validation."""
        return CanonicalTrialEvidence(
            trial_index=trial_index,
            seed=seed,
            plan_sha256=self.plan_sha256,
            scenario_sha256=self.scenario_sha256,
            execution_config_sha256=self.execution_config_sha256,
            tools_revision=self.tools_revision,
            engine_id=self.engine_id,
            engine_revision=self.engine_revision,
            model_id=self.model_id,
            sampled_inputs=sampled_inputs,
            outcome=outcome,
            trace=trace,
            impact=impact,
            closest_approach=closest_approach,
            failure_reason=failure_reason,
        )

    def failure(
        self,
        *,
        trial_index: int,
        seed: int,
        sampled_inputs: tuple[SampledInput, ...],
        error: Exception,
    ) -> CanonicalTrialEvidence:
        """Build a typed numerical-failure row without fabricated outputs."""
        if not isinstance(error, Exception):
            raise TypeError("error must be an Exception")
        return self.build(
            trial_index=trial_index,
            seed=seed,
            sampled_inputs=sampled_inputs,
            outcome="numerical_failure",
            trace=None,
            failure_reason=f"{type(error).__name__}: {error}",
        )


@dataclass(frozen=True)
class TrialObservation:
    """Outcome-specific trace and event evidence retained by an adapter."""

    outcome: TrialOutcome
    trace: TrialTrace | None
    impact: ImpactObservation | None = None
    closest_approach: ClosestApproach | None = None


def make_trial_evidence_identity(
    plan_sha256: str,
    scenario_sha256: str,
    execution_config_sha256: str,
    tools_revision: str,
    engine_id: str,
    engine_revision: str,
    model_id: str,
) -> TrialEvidenceIdentity:
    """Construct the immutable identity shared by one adapter execution."""
    return TrialEvidenceIdentity(
        plan_sha256=plan_sha256,
        scenario_sha256=scenario_sha256,
        execution_config_sha256=execution_config_sha256,
        tools_revision=tools_revision,
        engine_id=engine_id,
        engine_revision=engine_revision,
        model_id=model_id,
    )


def collect_trial_evidence(
    identity: TrialEvidenceIdentity,
    trial_index: int,
    seed: int,
    sampled_row: np.ndarray,
    columns: Sequence[SampledColumn],
    observation: TrialObservation,
) -> CanonicalTrialEvidence:
    """Build canonical evidence directly from one ordered sampled row."""
    return identity.build(
        trial_index=trial_index,
        seed=seed,
        sampled_inputs=sampled_inputs_from_row(sampled_row, columns),
        outcome=observation.outcome,
        trace=observation.trace,
        impact=observation.impact,
        closest_approach=observation.closest_approach,
    )


def collect_trial_failure(
    identity: TrialEvidenceIdentity,
    trial_index: int,
    seed: int,
    sampled_row: np.ndarray,
    columns: Sequence[SampledColumn],
    error: Exception,
) -> CanonicalTrialEvidence:
    """Build a typed failure directly from one ordered sampled row."""
    return identity.failure(
        trial_index=trial_index,
        seed=seed,
        sampled_inputs=sampled_inputs_from_row(sampled_row, columns),
        error=error,
    )


__all__ = [
    "TrialEvidenceIdentity",
    "TrialObservation",
    "collect_trial_evidence",
    "collect_trial_failure",
    "make_trial_evidence_identity",
    "require_fixed_step_horizon",
    "require_localized_time_window",
    "require_plan_execution_identity",
    "require_trace_index",
    "require_trial_result_geometry",
    "sampled_inputs_from_row",
]
