"""
Clipboard utilities for making simulation data copyable.

Provides context menu actions and helpers for copying plot data,
matrix values, and equation text to the system clipboard.

Design by Contract
------------------
- copy_matrix_to_clipboard(data) requires data to be a 2D np.ndarray.
- copy_series_to_clipboard(x, y) requires x, y to be 1D arrays of equal length.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def matrix_to_tsv(data: np.ndarray) -> str:
    """Convert a 2D numpy array to tab-separated text for clipboard.

    Pre: data.ndim == 2
    Post: returned string has data.shape[0] lines, each with data.shape[1] tab-separated values.
    """
    assert data.ndim == 2, f"Expected 2D array, got {data.ndim}D"
    lines = []
    for row in data:
        lines.append("\t".join(f"{v:.6g}" for v in row))
    result = "\n".join(lines)
    assert result.count("\n") == data.shape[0] - 1
    return result


def series_to_tsv(
    x: np.ndarray, y: np.ndarray, x_label: str = "x", y_label: str = "y"
) -> str:
    """Convert two 1D arrays to tab-separated text with header.

    Pre: x.shape == y.shape, both 1D
    Post: returned string has len(x)+1 lines (header + data).
    """
    assert x.ndim == 1 and y.ndim == 1, "Both arrays must be 1D"
    assert len(x) == len(y), f"Length mismatch: {len(x)} vs {len(y)}"
    header = f"{x_label}\t{y_label}"
    lines = [header]
    for xi, yi in zip(x, y):
        lines.append(f"{xi:.6g}\t{yi:.6g}")
    return "\n".join(lines)


def scalar_dict_to_text(d: dict[str, float], title: str = "") -> str:
    """Convert a dict of scalar values to readable text for clipboard.

    Pre: all values are numeric.
    Post: returned string contains one line per key-value pair.
    """
    assert d is not None, "d must be provided"
    lines = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
    for key, val in d.items():
        lines.append(f"{key}: {val:.6g}")
    return "\n".join(lines)
