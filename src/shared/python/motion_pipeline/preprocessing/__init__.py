"""
Preprocessing service for motion capture pipeline.

Part of issue #4564. Consolidates scattered preprocessing logic into
a single composable preprocessing service with a clear pipeline API.
"""

from .gap_fill import GapFillStrategy, gap_fill
from .filter import FilterType, apply_filter
from .resample import resample
from .normalize import normalize_coordinates, convert_units
from .pipeline import PreprocessingPipeline, PreprocessingStep

__all__ = [
    "GapFillStrategy",
    "gap_fill",
    "FilterType",
    "apply_filter",
    "resample",
    "normalize_coordinates",
    "convert_units",
    "PreprocessingPipeline",
    "PreprocessingStep",
]
