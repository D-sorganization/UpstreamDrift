"""Capsule shape: cylinder body + two hemisphere caps.

The capsule's local frame has the long axis along x. The cylinder portion
spans ``x in [0, length]``; the hemispheres protrude beyond that, so the
overall extent in x is ``[-radius, length + radius]``.
"""

from __future__ import annotations

import numpy as np

from .._types import FittedShape
from ._transform import apply_fitted_to_rest_vertices

__all__ = ["CapsuleShape"]


def _hemisphere_vertices_faces(
    radius: float, n_lon: int, n_lat: int, *, top: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return vertices/faces for a hemisphere of ``radius``.

    The hemisphere is centred at the origin with its open rim in the
    ``yz``-plane. ``top=True`` returns the +x half (cap of x-axis); the
    rim ring lies at ``x=0`` and the apex at ``x=+radius``.
    ``top=False`` mirrors to the -x side.

    Vertices are emitted from the rim toward the apex so that the rim
    ring is the FIRST ``n_lon`` vertices — this lets the caller stitch
    the hemisphere to a cylinder ring without duplicating those vertices.
    """
    sign = 1.0 if top else -1.0
    rings = np.linspace(0.0, np.pi / 2.0, n_lat + 1)
    lons = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    cos_lon = np.cos(lons)
    sin_lon = np.sin(lons)

    verts: list[np.ndarray] = []
    for i, phi in enumerate(rings):
        if i == n_lat:
            continue
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        ring = np.stack(
            [
                np.full(n_lon, sign * radius * sin_phi),
                radius * cos_phi * cos_lon,
                radius * cos_phi * sin_lon,
            ],
            axis=-1,
        )
        verts.append(ring)
    apex = np.array([[sign * radius, 0.0, 0.0]])
    verts.append(apex)
    vertices = np.concatenate(verts, axis=0).astype(np.float64)
    apex_idx = n_lat * n_lon

    faces: list[tuple[int, int, int]] = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            v00 = i * n_lon + j
            v01 = i * n_lon + j_next
            v10 = (i + 1) * n_lon + j
            v11 = (i + 1) * n_lon + j_next
            if top:
                faces.append((v00, v10, v11))
                faces.append((v00, v11, v01))
            else:
                faces.append((v00, v11, v10))
                faces.append((v00, v01, v11))
    last_ring = (n_lat - 1) * n_lon
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        if top:
            faces.append((last_ring + j, apex_idx, last_ring + j_next))
        else:
            faces.append((last_ring + j, last_ring + j_next, apex_idx))
    return vertices, np.asarray(faces, dtype=np.int64)


class CapsuleShape:
    """A capsule (cylinder body + two hemisphere caps) along the local x-axis."""

    shape_id: str
    rest_dimensions: tuple[float, ...]

    def __init__(
        self,
        length: float,
        radius: float,
        n_facets: int = 16,
        n_lat: int = 8,
        *,
        shape_id: str = "capsule",
    ) -> None:
        if not np.isfinite(float(length)) or float(length) <= 0.0:
            raise ValueError(f"length must be finite and > 0; got {length}")
        if not np.isfinite(float(radius)) or float(radius) <= 0.0:
            raise ValueError(f"radius must be finite and > 0; got {radius}")
        if not isinstance(n_facets, int) or isinstance(n_facets, bool):
            raise TypeError(f"n_facets must be int; got {type(n_facets).__name__}")
        if not isinstance(n_lat, int) or isinstance(n_lat, bool):
            raise TypeError(f"n_lat must be int; got {type(n_lat).__name__}")
        if n_facets < 3:
            raise ValueError(f"n_facets must be >= 3; got {n_facets}")
        if n_lat < 2:
            raise ValueError(f"n_lat must be >= 2; got {n_lat}")
        if not isinstance(shape_id, str) or not shape_id:
            raise ValueError(f"shape_id must be non-empty str; got {shape_id!r}")

        self.shape_id = shape_id
        self.rest_dimensions = (float(length), float(radius))
        self._n_facets = n_facets
        self._n_lat = n_lat
        self._build(float(length), float(radius), n_facets, n_lat)

    def _build(self, length: float, radius: float, n_facets: int, n_lat: int) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, n_facets, endpoint=False)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        base_ring = np.stack(
            [np.zeros(n_facets), radius * cos_a, radius * sin_a], axis=-1
        )
        top_ring = np.stack(
            [np.full(n_facets, length), radius * cos_a, radius * sin_a], axis=-1
        )

        bot_verts, bot_faces = _hemisphere_vertices_faces(
            radius, n_facets, n_lat, top=False
        )
        top_verts, top_faces = _hemisphere_vertices_faces(
            radius, n_facets, n_lat, top=True
        )
        # Bottom hemisphere rim (first n_facets verts) is identical to base
        # ring; reuse those vertices instead of duplicating. Translate the
        # remaining hemisphere verts by (0, 0, 0) -- already at origin.
        bot_extra = bot_verts[n_facets:]
        # Top hemisphere rim should align with top_ring at x=length;
        # translate the entire hemisphere by +length on x.
        top_translated = top_verts + np.array([length, 0.0, 0.0])
        top_extra = top_translated[n_facets:]

        n_base = n_facets
        n_top = n_facets
        n_bot_extra = bot_extra.shape[0]

        vertices = np.concatenate(
            [base_ring, top_ring, bot_extra, top_extra], axis=0
        ).astype(np.float64)

        # Side faces (cylinder body), reusing base_ring (idx 0..n_facets-1)
        # and top_ring (idx n_facets..2*n_facets-1).
        side_faces: list[tuple[int, int, int]] = []
        for j in range(n_facets):
            j_next = (j + 1) % n_facets
            v_b0 = j
            v_b1 = j_next
            v_t0 = n_facets + j
            v_t1 = n_facets + j_next
            side_faces.append((v_b0, v_t0, v_t1))
            side_faces.append((v_b0, v_t1, v_b1))

        # Re-index hemisphere faces.
        # Bottom: indices < n_facets refer to its rim (== base_ring [0..n_facets));
        # indices >= n_facets refer to bot_extra (offset by 2*n_facets).
        bot_offset = 2 * n_facets - n_facets  # extra-block start - n_facets
        bot_faces_remap = np.where(
            bot_faces < n_facets, bot_faces, bot_faces + bot_offset
        )
        # Top: rim refers to top_ring [n_facets..2*n_facets); extras live
        # after bot_extra, i.e. starting at 2*n_facets + n_bot_extra.
        top_extra_start = 2 * n_facets + n_bot_extra
        top_offset_rim = n_facets  # rim 0..n_facets -> top_ring n_facets..2n_facets
        top_offset_extra = top_extra_start - n_facets
        top_faces_remap = np.where(
            top_faces < n_facets,
            top_faces + top_offset_rim,
            top_faces + top_offset_extra,
        )

        all_faces = np.concatenate(
            [
                np.asarray(side_faces, dtype=np.int64),
                bot_faces_remap.astype(np.int64),
                top_faces_remap.astype(np.int64),
            ],
            axis=0,
        )
        self._vertices = vertices
        self._faces = all_faces
        self._n_base = n_base
        self._n_top = n_top

    @property
    def n_facets(self) -> int:
        return self._n_facets

    @property
    def n_lat(self) -> int:
        return self._n_lat

    def vertices_at_rest(self) -> np.ndarray:
        return self._vertices.copy()

    def faces(self) -> np.ndarray:
        return self._faces.copy()

    def transform(self, fitted: FittedShape) -> np.ndarray:
        return apply_fitted_to_rest_vertices(self._vertices, fitted)
