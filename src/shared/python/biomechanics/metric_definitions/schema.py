"""Types and schema for the canonical golf metric-definitions registry.

This module defines **structure only**: no metric computations or API wiring.
Epic #14421 / slice GM-1a (#14422).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DimensionKind(str, Enum):
    """Whether the measurand is reported in the plane or in full space."""

    two_d = "2d"
    three_d = "3d"


class SourceKind(str, Enum):
    """Origin of the numeric series."""

    measured = "measured"
    modelled = "modelled"


class NormalisationKind(str, Enum):
    """Declared normalisation for publishing and cross-study comparison."""

    none = "none"
    percent_body_weight = "percent_body_weight"
    peak_to_peak_unit = "peak_to_peak_unit"
    address_relative = "address_relative"
    event_relative_101 = "event_relative_101"


class FilterDifferentiationDescriptor(BaseModel):
    """Differentiation / filtering contract (Argon dissent on GM-1a).

    Preconditions: ``method`` names the algorithm in plain language; ``min_sampling_hz``
    guards against applying the filter below a trusted sampling rate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lowpass_cutoff_hz: float | None = Field(
        default=None,
        description="Optional low-pass cutoff in Hz; None when not low-pass filtered.",
    )
    method: str = Field(
        ...,
        min_length=1,
        description="Filter and/or differentiation method identifier.",
    )
    min_sampling_hz: float = Field(
        ...,
        gt=0.0,
        description="Minimum input sampling rate (Hz) required for this descriptor.",
    )

    @field_validator("lowpass_cutoff_hz")
    @classmethod
    def _positive_cutoff(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if v <= 0:
            msg = "lowpass_cutoff_hz must be positive when set"
            raise ValueError(msg)
        return v


class EventTimeNormalisationDescriptor(BaseModel):
    """Declared event-relative 101-point time normalisation contract (#14427).

    Defines 101-point resampling (0% to 100% in 1% increments) between named swing events.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_sequence: tuple[str, ...] = Field(
        default=("address", "impact"),
        min_length=2,
        description="Ordered sequence of named swing events spanning the 101-point window.",
    )
    n_points: int = Field(
        default=101,
        ge=2,
        description="Number of resampled points (default: 101 for 0%..100%).",
    )
    method: str = Field(
        default="linear",
        min_length=1,
        description="Interpolation method identifier ('linear' or 'cubic').",
    )

    @field_validator("event_sequence")
    @classmethod
    def _non_blank_sequence(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for item in v:
            if not isinstance(item, str) or not item.strip():
                msg = "event_sequence entries must be non-empty strings"
                raise ValueError(msg)
        return v


class GolfMetricDefinitionEntry(BaseModel):
    """One row in the metric-definitions registry.

    Every field is required for publication unless noted. ``cites`` is mandatory:
    each entry must name at least one external reference anchor (DOI, paper id,
    or in-repo ADR id).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_id: str = Field(..., min_length=1, description="Stable registry id.")
    segments: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="Body segments the measurand spans, in proximal-to-distal order.",
    )
    landmark_proxy: str = Field(
        ...,
        min_length=1,
        description="Joint-centre lines vs explicit bony landmarks, etc.",
    )
    dimension: DimensionKind
    reference_frame: str = Field(..., min_length=1)
    cardan_order: str = Field(
        ...,
        min_length=1,
        description="Cardan / Euler axis order for any reported attitude deltas.",
    )
    sign_convention: str = Field(..., min_length=1)
    units: str = Field(..., min_length=1)
    event_anchors: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="Named swing events where the metric is reported.",
    )
    normalisation: NormalisationKind
    source: SourceKind
    definition_version: str = Field(..., min_length=1, description="Semantic version.")
    cites: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="Mandatory bibliographic or ADR references.",
    )
    filter_descriptor: FilterDifferentiationDescriptor
    time_normalisation: EventTimeNormalisationDescriptor | None = Field(
        default=None,
        description="Optional 101-point event-relative time normalisation descriptor (#14427).",
    )

    @field_validator("segments", "event_anchors", "cites")
    @classmethod
    def _non_blank_tuple(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for item in v:
            if not isinstance(item, str) or not item.strip():
                msg = "tuple entries must be non-empty strings"
                raise ValueError(msg)
        return v


RegistryDocument = dict[str, GolfMetricDefinitionEntry]
"""Mapping ``metric_id`` -> definition (for JSON loaders)."""
