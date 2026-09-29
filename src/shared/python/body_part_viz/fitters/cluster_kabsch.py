"""Cluster-Kabsch rigid shape fitter.

Per-frame centroid + Kabsch rotation against the rest cluster. Optional
uniform scale via centred-norm ratio when ``enable_scale=True``.
"""

from __future__ import annotations

import numpy as np

from .._types import FittedShape
from ..bindings import BindingKind, MarkerBinding
from ..contracts import BodyPartShape
from ._kabsch import kabsch_rotation, stack_cluster

__all__ = ["ClusterKabschFitter"]


class ClusterKabschFitter:
    """Fitter for :data:`BindingKind.CLUSTER` bindings (≥3 markers).

    Pure rigid by default. With ``enable_scale=True`` an isotropic scale is
    estimated from the per-frame centred-norm ratio.
    """

    def __init__(self, *, enable_scale: bool = False) -> None:
        self._enable_scale = bool(enable_scale)

    def fit(
        self,
        shape: BodyPartShape,
        binding: MarkerBinding,
        markers_xyz: dict[str, np.ndarray],
    ) -> FittedShape:
        """Fit ``shape`` to the marker cluster in ``binding``."""
        if binding.kind is not BindingKind.CLUSTER:
            raise TypeError(
                f"ClusterKabschFitter requires BindingKind.CLUSTER; got {binding.kind}"
            )

        cluster = stack_cluster(markers_xyz, binding.marker_names)
        n_frames, n_markers, _ = cluster.shape

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
        rest_norm = float(np.linalg.norm(rest_centred))

        for t in np.flatnonzero(valid_mask):
            frame = cluster[t]
            frame_centroid = frame.mean(axis=0)
            frame_centred = frame - frame_centroid

            rot = kabsch_rotation(rest_centred, frame_centred)
            centroid[t] = frame_centroid
            rotation[t] = rot

            if self._enable_scale and rest_norm > 0.0:
                ratio = float(np.linalg.norm(frame_centred)) / rest_norm
                scale[t] = np.array([ratio, ratio, ratio])

        # DbC: positive scale on valid frames is required by FittedShape.
        if self._enable_scale and not bool(np.all(scale[valid_mask] > 0.0)):
            raise ValueError(
                "cluster collapsed to a point on at least one valid frame; "
                "uniform scale is undefined"
            )

        return FittedShape(
            shape_id=shape.shape_id,
            binding=binding,
            centroid=centroid,
            rotation_matrix=rotation,
            scale=scale,
            valid_mask=valid_mask,
        )
