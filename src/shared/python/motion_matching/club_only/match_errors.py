"""Separate in-plane and original 3D club match errors (CO-04)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ClubMatchErrorReport",
    "separate_plane_and_3d_errors",
]


@dataclass(frozen=True)
class ClubMatchErrorReport:
    """In-plane tracking error kept distinct from original 3D Euclidean error."""

    in_plane_rmse_m: float
    original_3d_rmse_m: float
    out_of_plane_rmse_m: float
    n_valid: int

    def __post_init__(self) -> None:
        for name, value in (
            ("in_plane_rmse_m", self.in_plane_rmse_m),
            ("original_3d_rmse_m", self.original_3d_rmse_m),
            ("out_of_plane_rmse_m", self.out_of_plane_rmse_m),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.n_valid < 0:
            raise ValueError("n_valid must be >= 0")


def separate_plane_and_3d_errors(
    *,
    predicted_xyz_m: np.ndarray,
    measured_xyz_m: np.ndarray,
    plane_origin: np.ndarray,
    plane_basis: np.ndarray,
    observed_mask: np.ndarray,
) -> ClubMatchErrorReport:
    """Report in-plane RMSE and original 3D RMSE without collapsing them."""
    pred = np.asarray(predicted_xyz_m, dtype=np.float64)
    meas = np.asarray(measured_xyz_m, dtype=np.float64)
    origin = np.asarray(plane_origin, dtype=np.float64).reshape(3)
    basis = np.asarray(plane_basis, dtype=np.float64)
    mask = np.asarray(observed_mask, dtype=bool)

    if pred.shape != meas.shape or pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError("predicted/measured must share shape (N, 3)")
    if basis.shape != (3, 3):
        raise ValueError("plane_basis must be shape (3, 3)")
    if mask.shape != (pred.shape[0],):
        raise ValueError("observed_mask length must match frame count")
    if not np.all(np.isfinite(basis)):
        raise ValueError("plane_basis must be finite")

    valid = mask & np.all(np.isfinite(pred), axis=1) & np.all(np.isfinite(meas), axis=1)
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        raise ValueError("no valid frames to score")

    _ = origin  # reserved for absolute plane-frame diagnostics
    err = pred[valid] - meas[valid]
    original_3d = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))

    # Plane coords: columns of basis are [u, v, n]; difference vectors omit origin.
    plane_err = err @ basis
    in_plane = float(np.sqrt(np.mean(plane_err[:, 0] ** 2 + plane_err[:, 1] ** 2)))
    out_of_plane = float(np.sqrt(np.mean(plane_err[:, 2] ** 2)))

    return ClubMatchErrorReport(
        in_plane_rmse_m=in_plane,
        original_3d_rmse_m=original_3d,
        out_of_plane_rmse_m=out_of_plane,
        n_valid=n_valid,
    )
