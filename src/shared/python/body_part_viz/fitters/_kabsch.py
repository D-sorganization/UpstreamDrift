"""Pure-NumPy Kabsch algorithm for optimal rigid rotation.

Reused by :mod:`cluster_kabsch` and :mod:`procrustes_anisotropic`. Decoupled
into its own module so additional fitters can adopt it without coupling.
"""

from __future__ import annotations

import numpy as np

__all__ = ["kabsch_rotation", "stack_cluster"]


def stack_cluster(
    markers_xyz: dict[str, np.ndarray], names: tuple[str, ...]
) -> np.ndarray:
    """Stack named ``(T, 3)`` trajectories into a ``(T, N, 3)`` cluster.

    Raises
    ------
    KeyError
        If any named marker is missing from ``markers_xyz``.
    ValueError
        If a marker has the wrong shape or a different frame count from
        the first marker.
    """
    arrays: list[np.ndarray] = []
    n_frames: int | None = None
    for name in names:
        if name not in markers_xyz:
            raise KeyError(f"missing marker {name!r} in markers_xyz")
        arr = markers_xyz[name]
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"marker {name!r} must have shape (T, 3); got {arr.shape}")
        if n_frames is None:
            n_frames = arr.shape[0]
        elif arr.shape[0] != n_frames:
            raise ValueError(
                f"marker {name!r} has {arr.shape[0]} frames; expected {n_frames}"
            )
        arrays.append(arr)
    return np.stack(arrays, axis=1)


def kabsch_rotation(p_centred: np.ndarray, q_centred: np.ndarray) -> np.ndarray:
    """Return ``R`` minimising ``‖R @ P.T - Q.T‖_F`` with ``det(R) == +1``.

    Parameters
    ----------
    p_centred:
        ``(N, 3)`` source points, centroid-subtracted.
    q_centred:
        ``(N, 3)`` target points, centroid-subtracted.

    Returns
    -------
    rotation:
        ``(3, 3)`` proper rotation matrix (no reflection) such that
        ``q_centred ≈ p_centred @ rotation.T``.

    Raises
    ------
    ValueError
        If shapes mismatch, are not 2-D, or do not have 3 columns.
    """
    if p_centred.ndim != 2 or q_centred.ndim != 2:
        raise ValueError(
            "kabsch_rotation expects 2-D arrays; "
            f"got P.ndim={p_centred.ndim}, Q.ndim={q_centred.ndim}"
        )
    if p_centred.shape != q_centred.shape:
        raise ValueError(
            "kabsch_rotation requires matching shapes; "
            f"got P.shape={p_centred.shape}, Q.shape={q_centred.shape}"
        )
    if p_centred.shape[1] != 3:
        raise ValueError(
            f"kabsch_rotation requires 3 columns; got {p_centred.shape[1]}"
        )

    covariance = p_centred.T @ q_centred
    u_mat, _singular, vt_mat = np.linalg.svd(covariance)

    # Reflection guard: ensure proper rotation (det == +1).
    det_sign = float(np.linalg.det(vt_mat.T @ u_mat.T))
    sign_diag = np.array([1.0, 1.0, 1.0 if det_sign > 0.0 else -1.0])
    rotation = vt_mat.T @ np.diag(sign_diag) @ u_mat.T
    return rotation
