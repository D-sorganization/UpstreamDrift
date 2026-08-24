"""Typed complete-trial evidence for canonical variation execution.

These records contain only UpstreamDrift-owned execution evidence. Tools owns
the sampled plan and analysis mathematics; the exact plan digest and Tools
revision bind each trial back to that authority.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

TRIAL_EVIDENCE_SCHEMA_VERSION = "upstream-tools-variation-trial/v1"
TrialOutcome = Literal["hit", "no_impact", "numerical_failure", "partial_valid_trace"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
_OUTCOMES = frozenset({"hit", "no_impact", "numerical_failure", "partial_valid_trace"})


def _require_stable_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not _STABLE_ID.fullmatch(value):
        raise ValueError(f"{name} must be a stable non-empty identifier")
    return value


def _require_unique_ids(values: tuple[str, ...], name: str) -> None:
    if not values or any(not _STABLE_ID.fullmatch(value) for value in values):
        raise ValueError(f"{name} must contain stable identifiers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


def _readonly_array(value: object, name: str, ndim: int) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class SampledInput:
    """One named scalar input or output with explicit units."""

    name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        _require_stable_id(self.name, "sample name")
        if not np.isfinite(self.value):
            raise ValueError("sample value must be finite")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("sample unit must be non-empty")


def _validate_samples(values: tuple[SampledInput, ...], name: str) -> None:
    if any(not isinstance(value, SampledInput) for value in values):
        raise TypeError(f"{name} must contain SampledInput records")
    identifiers = tuple(value.name for value in values)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} names must be unique")


@dataclass(frozen=True)
class ImpactObservation:
    """Impact time and available finite impact-state observables."""

    time_s: float
    state: tuple[SampledInput, ...]

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("impact time_s must be finite and non-negative")
        _validate_samples(self.state, "impact state")


@dataclass(frozen=True)
class ClosestApproach:
    """Miss-safe closest-approach/contact observation."""

    time_s: float
    distance_m: float
    source_marker_id: str
    target_id: str
    contact_observed: bool

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("closest-approach time_s must be finite and non-negative")
        if not np.isfinite(self.distance_m) or self.distance_m < 0.0:
            raise ValueError(
                "closest-approach distance_m must be finite and non-negative"
            )
        _require_stable_id(self.source_marker_id, "source_marker_id")
        _require_stable_id(self.target_id, "target_id")
        if type(self.contact_observed) is not bool:
            raise TypeError("contact_observed must be bool")


@dataclass(frozen=True)
class TrialTrace:
    """Frame- and identifier-safe pre-impact model trace."""

    times_s: np.ndarray
    q: np.ndarray
    v: np.ndarray
    coordinate_ids: tuple[str, ...]
    coordinate_units: tuple[str, ...]
    velocity_units: tuple[str, ...]
    markers_m: np.ndarray
    marker_ids: tuple[str, ...]
    frame_id: str
    alignment_id: str
    complete: bool

    def __post_init__(self) -> None:
        times = _readonly_array(self.times_s, "times_s", 1)
        q = _readonly_array(self.q, "q", 2)
        v = _readonly_array(self.v, "v", 2)
        markers = _readonly_array(self.markers_m, "markers_m", 3)
        if times.size == 0:
            raise ValueError("times_s must not be empty")
        if times[0] < 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be non-negative and strictly increasing")
        if q.shape != v.shape or q.shape[0] != times.size:
            raise ValueError("q and v must share a time-aligned shape")
        _require_unique_ids(self.coordinate_ids, "coordinate IDs")
        if q.shape[1] != len(self.coordinate_ids):
            raise ValueError("coordinate IDs must identify every q and v column")
        if (
            len(self.coordinate_units) != q.shape[1]
            or len(self.velocity_units) != q.shape[1]
            or any(not unit for unit in self.coordinate_units + self.velocity_units)
        ):
            raise ValueError("coordinate and velocity units must identify every column")
        if markers.shape[0] != times.size or markers.shape[2] != 3:
            raise ValueError("markers_m must have shape (time, marker, 3)")
        _require_unique_ids(self.marker_ids, "marker IDs")
        if markers.shape[1] != len(self.marker_ids):
            raise ValueError("marker IDs must identify every marker column")
        _require_stable_id(self.frame_id, "frame_id")
        _require_stable_id(self.alignment_id, "alignment_id")
        if type(self.complete) is not bool:
            raise TypeError("complete must be bool")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "markers_m", markers)


@dataclass(frozen=True)
class CanonicalTrialEvidence:
    """One complete typed trial, including valid misses and failures."""

    trial_index: int
    seed: int
    plan_sha256: str
    tools_revision: str
    engine_id: str
    engine_revision: str
    model_id: str
    sampled_inputs: tuple[SampledInput, ...]
    outcome: TrialOutcome
    trace: TrialTrace | None
    impact: ImpactObservation | None = None
    shot_result: tuple[SampledInput, ...] | None = None
    closest_approach: ClosestApproach | None = None
    failure_reason: str | None = None
    schema_version: str = TRIAL_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.trial_index) is not int or self.trial_index < 0:
            raise ValueError("trial_index must be a non-negative integer")
        if type(self.seed) is not int:
            raise TypeError("seed must be an integer")
        if not _SHA256.fullmatch(self.plan_sha256):
            raise ValueError("plan_sha256 must be a 64-character lowercase digest")
        for value, name in (
            (self.tools_revision, "tools_revision"),
            (self.engine_revision, "engine_revision"),
        ):
            if not _COMMIT.fullmatch(value):
                raise ValueError(f"{name} must be a 40-character lowercase commit")
        _require_stable_id(self.engine_id, "engine_id")
        _require_stable_id(self.model_id, "model_id")
        _validate_samples(self.sampled_inputs, "sampled_inputs")
        if not self.sampled_inputs:
            raise ValueError("sampled_inputs must not be empty")
        if self.shot_result is not None:
            _validate_samples(self.shot_result, "shot_result")
        if self.outcome not in _OUTCOMES:
            raise ValueError("outcome is unsupported")
        if self.schema_version != TRIAL_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("schema_version is incompatible")
        self._validate_outcome()
        self._validate_event_times()

    def _validate_outcome(self) -> None:
        if self.outcome == "hit":
            if self.impact is None:
                raise ValueError("hit outcome requires impact evidence")
            if self.trace is None or not self.trace.complete:
                raise ValueError("hit outcome requires a complete trace")
            if self.failure_reason is not None:
                raise ValueError("hit outcome must not contain a failure reason")
            return
        if self.outcome == "no_impact":
            if self.closest_approach is None:
                raise ValueError("no_impact outcome requires closest approach evidence")
            if self.trace is None or not self.trace.complete:
                raise ValueError("no_impact outcome requires a complete trace")
            if self.impact is not None or self.shot_result is not None:
                raise ValueError(
                    "no_impact outcome must have null impact and shot data"
                )
            if self.failure_reason is not None:
                raise ValueError("no_impact outcome must not contain a failure reason")
            return
        if self.impact is not None or self.shot_result is not None:
            raise ValueError("failure outcomes must not contain impact or shot data")
        if not isinstance(self.failure_reason, str) or not self.failure_reason.strip():
            raise ValueError("failure outcomes require a failure reason")
        if self.outcome == "partial_valid_trace":
            if self.trace is None or self.trace.complete:
                raise ValueError("partial_valid_trace requires an incomplete trace")
        elif self.trace is not None:
            raise ValueError("numerical_failure must not contain a trace")

    def _validate_event_times(self) -> None:
        if self.trace is None:
            return
        last_time = float(self.trace.times_s[-1])
        for event in (self.impact, self.closest_approach):
            if event is not None and event.time_s > last_time:
                raise ValueError("event time lies outside the retained trace")


__all__ = [
    "TRIAL_EVIDENCE_SCHEMA_VERSION",
    "CanonicalTrialEvidence",
    "ClosestApproach",
    "ImpactObservation",
    "SampledInput",
    "TrialOutcome",
    "TrialTrace",
]
