"""Forward kinematics and skinning that redo only what a pose changed (#15150).

A curl search poses one hand tens of thousands of times, and between two
poses one or two joints differ. :class:`PoseKinematics` keeps the last local
and world transform of every bone and the last rows of the skinning matrix
that were asked for, and recomputes a bone only when what it is made of, or
a bone above it, changed. What it recomputes it computes with
the same operations in the same order as a hand that keeps nothing, so the
results are equal bit for bit (a unit test holds the two together).

Inputs are compared by identity: a rotation or offset matrix handed in must be
replaced, never changed in place.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.hand_backends.skinning import (
    SkinnedHandAsset,
)

Array = NDArray[np.float64]
_EYE3 = np.eye(3)
_EYE4 = np.eye(4)


class PoseKinematics:
    """Hinges, bone transforms and skinned points of one rest asset."""

    def __init__(self, asset: SkinnedHandAsset) -> None:
        self._heads = asset.bone_heads
        self._parents = [int(p) for p in asset.bone_parents]
        axes = asset.flex_axes
        k = np.zeros((len(axes), 3, 3))
        k[:, 0, 1], k[:, 0, 2] = -axes[:, 2], axes[:, 1]
        k[:, 1, 0], k[:, 1, 2] = axes[:, 2], -axes[:, 0]
        k[:, 2, 0], k[:, 2, 1] = -axes[:, 1], axes[:, 0]
        self._hinge_k, self._hinge_kk = k, k @ k
        # Linear blend skinning as one matrix product: weights times [v, 1].
        rest = asset.rest_vertices
        homogeneous = np.hstack([rest, np.ones((len(rest), 1))])
        dense = asset.dense_weights()
        self._blend = (dense[:, :, None] * homogeneous[:, None, :]).reshape(
            len(dense), -1
        )
        count = len(self._parents)
        self._made_of: list[tuple[Array | None, Array | None] | None] = [None] * count
        self._local: list[Array | None] = [None] * count
        self._world = np.empty((count, 4, 4))
        self._rows: tuple[tuple[str, bytes], Array] | None = None

    def hinge(self, bone: int, angle: float) -> Array:
        """Rodrigues rotation about ``bone``'s flexion axis (precomputed)."""
        k, kk = self._hinge_k[bone], self._hinge_kk[bone]
        return _EYE3 + math.sin(angle) * k + (1.0 - math.cos(angle)) * kk

    def _local_of(self, bone: int, rot: Array | None, offset: Array | None) -> Array:
        local = _EYE4.copy()
        if rot is not None:
            head = self._heads[bone]
            local[:3, :3] = rot
            local[:3, 3] = head - rot @ head
        return local if offset is None else offset @ local

    def world(
        self, rotations: Sequence[Array | None], offsets: Sequence[Array | None]
    ) -> Array:
        """``(B, 4, 4)`` rest-to-posed transform of every bone, parents first.

        ``rotations[i]`` turns bone ``i`` about its head and ``offsets[i]`` is
        a rigid move applied on top; ``None`` is none. The array returned is
        this object's own and is overwritten by the next call: read it, and
        copy it to keep it.
        """
        out, parents, locals_ = self._world, self._parents, self._local
        moved = [False] * len(parents)
        for i, (rot, offset) in enumerate(zip(rotations, offsets, strict=True)):
            parent, last = parents[i], self._made_of[i]
            if last is None or last[0] is not rot or last[1] is not offset:
                self._made_of[i] = (rot, offset)
                still = rot is None and offset is None
                locals_[i] = None if still else self._local_of(i, rot, offset)
            elif parent < 0 or not moved[parent]:
                continue
            moved[i], local = True, locals_[i]
            if local is None:
                out[i] = _EYE4 if parent < 0 else out[parent]
            else:
                out[i] = local if parent < 0 else out[parent] @ local
        return out

    def skinned(self, world: Array, ids: NDArray[np.int64] | None = None) -> Array:
        """Rest vertices (only ``ids``, if given) skinned by ``world``."""
        stacked = world[:, :3, :].transpose(0, 2, 1).reshape(-1, 3)
        if ids is None:
            return np.asarray(self._blend @ stacked)
        key = (ids.dtype.str, ids.tobytes())
        if self._rows is None or self._rows[0] != key:
            self._rows = (key, self._blend[ids])
        return np.asarray(self._rows[1] @ stacked)


__all__ = ["PoseKinematics"]
