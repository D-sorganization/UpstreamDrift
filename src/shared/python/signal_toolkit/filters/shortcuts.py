"""Convenience wrapper functions for common filter types."""

from __future__ import annotations

from .filter_design import FilterDesign, FilterDesigner, FilterSpec, FilterType


def create_butterworth_filter(
    filter_type: str,
    cutoff: float | tuple[float, float],
    fs: float,
    order: int = 4,
) -> FilterSpec:
    """Create a Butterworth filter (convenience wrapper).

    Args:
        filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop', 'notch'.
        cutoff: Cutoff frequency or (low, high) tuple.
        fs: Sampling frequency.
        order: Filter order.

    Returns:
        FilterSpec.
    """
    if not (filter_type is not None):
        raise ValueError("filter_type must be provided")
    if not (filter_type is not None):
        raise ValueError("filter_type must be provided")
    ft = FilterType(filter_type)
    return FilterDesigner.butterworth(ft, cutoff, fs, order)


def create_chebyshev_filter(
    filter_type: str,
    cutoff: float | tuple[float, float],
    fs: float,
    order: int = 4,
    ripple_db: float = 1.0,
) -> FilterSpec:
    """Create a Chebyshev Type I filter (convenience wrapper).

    Args:
        filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop', 'notch'.
        cutoff: Cutoff frequency or (low, high) tuple.
        fs: Sampling frequency.
        order: Filter order.
        ripple_db: Passband ripple in dB.

    Returns:
        FilterSpec.
    """
    if not (filter_type is not None):
        raise ValueError("filter_type must be provided")
    if not (filter_type is not None):
        raise ValueError("filter_type must be provided")
    ft = FilterType(filter_type)
    return FilterDesigner.chebyshev1(ft, cutoff, fs, order, ripple_db)


__all__ = [
    "create_butterworth_filter",
    "create_chebyshev_filter",
    # Re-exported for convenience
    "FilterDesign",
    "FilterSpec",
    "FilterType",
]
