"""Golf metric-definitions registry (schema and normalisation; issue #14422, #14427)."""

from __future__ import annotations

from .event_time_normalisation import (
    TimeNormalisedResult,
    resample_event_relative_101,
)
from .schema import (
    DimensionKind,
    EventTimeNormalisationDescriptor,
    FilterDifferentiationDescriptor,
    GolfMetricDefinitionEntry,
    NormalisationKind,
    SourceKind,
)

__all__ = [
    "DimensionKind",
    "EventTimeNormalisationDescriptor",
    "FilterDifferentiationDescriptor",
    "GolfMetricDefinitionEntry",
    "NormalisationKind",
    "SourceKind",
    "TimeNormalisedResult",
    "resample_event_relative_101",
]
