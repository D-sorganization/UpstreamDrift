"""Signal filtering package.

Re-exports all public symbols from submodules to maintain full backward
compatibility with ``from .filters import X`` imports.
"""

from __future__ import annotations

from .advanced_filters import (
    AdaptiveFilter,
    apply_bilateral_filter,
    apply_gaussian_smoothing,
)
from .filter_design import (
    FilterDesign,
    FilterDesigner,
    FilterSpec,
    FilterType,
    _normalize_cutoff,
)
from .shortcuts import create_butterworth_filter, create_chebyshev_filter
from .smoothing_filters import (
    apply_exponential_smoothing,
    apply_filter,
    apply_median_filter,
    apply_moving_average,
    apply_savgol,
    create_moving_average_filter,
    create_savgol_filter,
)

__all__ = [
    # filter_design
    "FilterType",
    "FilterDesign",
    "FilterSpec",
    "FilterDesigner",
    "_normalize_cutoff",
    # smoothing_filters
    "apply_filter",
    "apply_moving_average",
    "apply_savgol",
    "apply_median_filter",
    "apply_exponential_smoothing",
    "create_moving_average_filter",
    "create_savgol_filter",
    # advanced_filters
    "apply_gaussian_smoothing",
    "apply_bilateral_filter",
    "AdaptiveFilter",
    # shortcuts
    "create_butterworth_filter",
    "create_chebyshev_filter",
]
