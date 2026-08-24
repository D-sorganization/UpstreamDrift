"""Fail-closed compatibility and numerical parity for canonical trials."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trial_evidence import CanonicalTrialEvidence, TrialTrace


class CrossEngineCompatibilityError(ValueError):
    """Raised before comparison when trial semantics are not equivalent."""


@dataclass(frozen=True)
class CrossEngineTolerances:
    """Declared absolute tolerances with one value per generalized coordinate."""

    time_atol_s: float
    coordinate_atol: tuple[float, ...]
    velocity_atol: tuple[float, ...]
    marker_atol_m: float

    def __post_init__(self) -> None:
        values = (
            self.time_atol_s,
            *self.coordinate_atol,
            *self.velocity_atol,
            self.marker_atol_m,
        )
        if not self.coordinate_atol or not self.velocity_atol:
            raise ValueError("coordinate and velocity tolerances must not be empty")
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("all cross-engine tolerances must be finite and positive")


@dataclass(frozen=True)
class CrossEngineParityMetrics:
    """Typed discrepancies after all semantic compatibility gates pass."""

    reference_engine_id: str
    candidate_engine_id: str
    outcome_match: bool
    max_time_error_s: float
    max_coordinate_normalized_error: float
    max_velocity_normalized_error: float
    max_marker_error_m: float
    tolerance_equivalent: bool


def _require_equal(reference: object, candidate: object, label: str) -> None:
    if reference != candidate:
        raise CrossEngineCompatibilityError(f"cross-engine {label} mismatch")


def _require_trial_identity(
    reference: CanonicalTrialEvidence, candidate: CanonicalTrialEvidence
) -> None:
    _require_equal(reference.trial_index, candidate.trial_index, "trial index")
    _require_equal(reference.seed, candidate.seed, "seed")
    _require_equal(reference.plan_sha256, candidate.plan_sha256, "plan digest")
    _require_equal(
        reference.scenario_sha256,
        candidate.scenario_sha256,
        "scenario digest",
    )
    _require_equal(reference.tools_revision, candidate.tools_revision, "Tools revision")
    _require_equal(reference.model_id, candidate.model_id, "model identity")
    _require_equal(reference.sampled_inputs, candidate.sampled_inputs, "sampled inputs")


def _require_trace(record: CanonicalTrialEvidence, role: str) -> TrialTrace:
    trace = record.trace
    if trace is None or not trace.complete:
        raise CrossEngineCompatibilityError(
            f"{role} trial requires one complete trace for numerical parity"
        )
    return trace


def _require_trace_semantics(reference: TrialTrace, candidate: TrialTrace) -> None:
    _require_equal(reference.frame_id, candidate.frame_id, "frame")
    _require_equal(reference.alignment_id, candidate.alignment_id, "alignment")
    _require_equal(reference.coordinate_ids, candidate.coordinate_ids, "coordinate IDs")
    _require_equal(
        reference.coordinate_units, candidate.coordinate_units, "coordinate units"
    )
    _require_equal(reference.velocity_units, candidate.velocity_units, "velocity units")
    _require_equal(reference.marker_ids, candidate.marker_ids, "marker IDs")
    _require_equal(reference.times_s.shape, candidate.times_s.shape, "time-grid shape")
    _require_equal(reference.q.shape, candidate.q.shape, "coordinate shape")
    _require_equal(reference.v.shape, candidate.v.shape, "velocity shape")
    _require_equal(reference.markers_m.shape, candidate.markers_m.shape, "marker shape")


def _require_tolerance_dimensions(
    trace: TrialTrace, tolerances: CrossEngineTolerances
) -> None:
    width = trace.q.shape[1]
    if (
        len(tolerances.coordinate_atol) != width
        or len(tolerances.velocity_atol) != width
    ):
        raise CrossEngineCompatibilityError(
            "cross-engine tolerance dimensions must match coordinate topology"
        )


def _max_normalized_error(
    reference: np.ndarray, candidate: np.ndarray, tolerances: tuple[float, ...]
) -> float:
    scale = np.asarray(tolerances, dtype=float)
    normalized = np.abs(candidate - reference) / scale[np.newaxis, :]
    return float(np.max(normalized))


def _max_marker_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    difference = candidate - reference
    distances = np.sqrt(np.einsum("tmc,tmc->tm", difference, difference))
    return float(np.max(distances))


def compare_cross_engine_trials(
    reference: CanonicalTrialEvidence,
    candidate: CanonicalTrialEvidence,
    tolerances: CrossEngineTolerances,
) -> CrossEngineParityMetrics:
    """Validate semantics, then report declared-tolerance trace discrepancies."""

    if not isinstance(reference, CanonicalTrialEvidence) or not isinstance(
        candidate, CanonicalTrialEvidence
    ):
        raise TypeError("reference and candidate must be CanonicalTrialEvidence")
    if not isinstance(tolerances, CrossEngineTolerances):
        raise TypeError("tolerances must be CrossEngineTolerances")
    _require_trial_identity(reference, candidate)
    reference_trace = _require_trace(reference, "reference")
    candidate_trace = _require_trace(candidate, "candidate")
    _require_trace_semantics(reference_trace, candidate_trace)
    _require_tolerance_dimensions(reference_trace, tolerances)

    time_error = float(
        np.max(np.abs(candidate_trace.times_s - reference_trace.times_s))
    )
    coordinate_error = _max_normalized_error(
        reference_trace.q,
        candidate_trace.q,
        tolerances.coordinate_atol,
    )
    velocity_error = _max_normalized_error(
        reference_trace.v,
        candidate_trace.v,
        tolerances.velocity_atol,
    )
    marker_error = _max_marker_error(
        reference_trace.markers_m, candidate_trace.markers_m
    )
    equivalent = (
        time_error <= tolerances.time_atol_s
        and coordinate_error <= 1.0
        and velocity_error <= 1.0
        and marker_error <= tolerances.marker_atol_m
    )
    return CrossEngineParityMetrics(
        reference_engine_id=reference.engine_id,
        candidate_engine_id=candidate.engine_id,
        outcome_match=reference.outcome == candidate.outcome,
        max_time_error_s=time_error,
        max_coordinate_normalized_error=coordinate_error,
        max_velocity_normalized_error=velocity_error,
        max_marker_error_m=marker_error,
        tolerance_equivalent=equivalent,
    )


__all__ = [
    "CrossEngineCompatibilityError",
    "CrossEngineParityMetrics",
    "CrossEngineTolerances",
    "compare_cross_engine_trials",
]
