"""Anisotropic-scale Procrustes fitter for marker clusters.

Centroid + Kabsch rotation, then anisotropic per-axis scale solved via
``scipy.linalg.lstsq`` on the centred-rotated cluster vs the rest cluster.

Documented as the most-flexible but least-stable fitter; recommended only
when ≥4 markers are available and the user wants anisotropic stretch. With
fewer than 4 markers we log a warning and fall back to Kabsch behaviour
(unit scale).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.linalg import lstsq

from .._types import FittedShape
from ..bindings import BindingKind, MarkerBinding
from ..contracts import BodyPartShape
from ._kabsch import kabsch_rotation, stack_cluster

__all__ = ["ProcrustesAnisotropicFitter"]

_LOG = logging.getLogger(__name__)
_MIN_MARKERS_FOR_ANISO = 4


class ProcrustesAnisotropicFitter:
    """Fitter for :data:`BindingKind.CLUSTER` with anisotropic scale.

    Per frame:

    * centroid + Kabsch rotation (as in :class:`ClusterKabschFitter`).
    * Solve ``rotated_centred @ diag(s) ≈ rest_centred`` for ``s`` via
      least squares (``scipy.linalg.lstsq``).
    """

    def fit(
        self,
        shape: BodyPartShape,
        binding: MarkerBinding,
        markers_xyz: dict[str, np.ndarray],
    ) -> FittedShape:
        """Fit ``shape`` to ``markers_xyz`` with anisotropic scale."""
        if binding.kind is not BindingKind.CLUSTER:
            raise TypeError(
                "ProcrustesAnisotropicFitter requires BindingKind.CLUSTER; "
                f"got {binding.kind}"
            )

        cluster = stack_cluster(markers_xyz, binding.marker_names)
        n_frames, n_markers, _ = cluster.shape

        aniso_enabled = n_markers >= _MIN_MARKERS_FOR_ANISO
        if not aniso_enabled:
            _LOG.warning(
                "ProcrustesAnisotropicFitter: %d markers < %d; "
                "falling back to Kabsch (unit scale).",
                n_markers,
                _MIN_MARKERS_FOR_ANISO,
            )

        valid_mask = np.isfinite(cluster).all(axis=(1, 2))

        centroid = np.zeros((n_frames, 3), dtype=float)
        rotation = np.broadcast_to(np.eye(3), (n_frames, 3, 3)).copy()
        scale = np.ones((n_frames, 3), dtype=float)

        if not valid_mask.any():
            return FittedShape(
                shape_id=shape.shape_id,
                binding=binding,
                centroid=centroid,
                rotation_matrix=rotation,
                scale=scale,
                valid_mask=valid_mask,
            )

        rest_idx = int(np.flatnonzero(valid_mask)[0])
        rest_cluster = cluster[rest_idx]
        rest_centroid = rest_cluster.mean(axis=0)
        rest_centred = rest_cluster - rest_centroid

        for t in np.flatnonzero(valid_mask):
            frame = cluster[t]
            frame_centroid = frame.mean(axis=0)
            frame_centred = frame - frame_centroid

            rot = kabsch_rotation(rest_centred, frame_centred)
            centroid[t] = frame_centroid
            rotation[t] = rot

            if aniso_enabled:
                # Bring frame back into rest-frame coordinates: ``frame_in_rest``
                # ≈ rest_centred * scale_diag. Solve column-by-column for s.
                frame_in_rest = frame_centred @ rot
                scale[t] = _solve_axiswise_scale(rest_centred, frame_in_rest)

        return FittedShape(
            shape_id=shape.shape_id,
            binding=binding,
            centroid=centroid,
            rotation_matrix=rotation,
            scale=scale,
            valid_mask=valid_mask,
        )


def _solve_axiswise_scale(
    rest_centred: np.ndarray, frame_in_rest: np.ndarray
) -> np.ndarray:
    """Return ``(3,)`` positive scale solving ``rest * s ≈ frame_in_rest``.

    Each axis is independent: ``s_i = lstsq(rest[:, i:i+1], frame[:, i])``.
    """
    out = np.ones(3, dtype=float)
    for axis_idx in range(3):
        col_rest = rest_centred[:, axis_idx : axis_idx + 1]
        col_frame = frame_in_rest[:, axis_idx]
        if float(np.linalg.norm(col_rest)) == 0.0:
            out[axis_idx] = 1.0
            continue
        sol, *_ = lstsq(col_rest, col_frame)
        value = float(sol[0])
        # Guard FittedShape's strictly-positive postcondition.
        if not np.isfinite(value) or value <= 0.0:
            value = 1.0
        out[axis_idx] = value
    return out
