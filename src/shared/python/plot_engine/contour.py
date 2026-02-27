"""Contour and heatmap data preparation utilities.

Provides grid interpolation and data preparation for contour plots
and heatmaps, bridging scatter data to gridded formats.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata


def scatter_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    resolution: int = 100,
    method: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate scattered (x, y, z) data onto a regular grid.

    Args:
        x: X coordinates of scatter points.
        y: Y coordinates of scatter points.
        z: Z values at scatter points.
        resolution: Number of grid points per axis.
        method: Interpolation method ('linear', 'nearest', 'cubic').

    Returns:
        Tuple of (x_grid, y_grid, z_grid) where x_grid and y_grid are 1D
        arrays of grid coordinates and z_grid is the 2D interpolated values.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    # Remove NaN entries
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]

    if len(x) < 3:
        raise ValueError("At least 3 valid points required for grid interpolation")

    x_lin = np.linspace(x.min(), x.max(), resolution)
    y_lin = np.linspace(y.min(), y.max(), resolution)
    x_mesh, y_mesh = np.meshgrid(x_lin, y_lin)

    z_grid = griddata(
        np.column_stack([x, y]),
        z,
        (x_mesh, y_mesh),
        method=method,
    )

    return x_lin, y_lin, z_grid


def correlation_matrix(
    data: np.ndarray,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute correlation matrix from columnar data.

    Args:
        data: 2D array where columns are variables.
        labels: Optional labels for columns.

    Returns:
        Tuple of (correlation_matrix, labels).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("Data must be 2D (rows x columns)")

    n_cols = data.shape[1]
    if labels is None:
        labels = [f"Var {i}" for i in range(n_cols)]

    corr = np.corrcoef(data, rowvar=False)
    return corr, labels
