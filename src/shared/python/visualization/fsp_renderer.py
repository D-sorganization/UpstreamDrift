"""Translucent FSP plane renderer (Phase 3, issue #5504).

Renders a square mesh aligned with the best-fit Functional Swing Plane
onto any viewport that satisfies the minimal :class:`Viewport`
``Protocol`` -- nothing more than ``add_mesh`` / ``remove_mesh``.

Color coding (configurable on :class:`FspRenderConfig`):

* **Green**  -- mean signed deviation is within ``on_plane_threshold``;
  the swing is "on plane".
* **Orange** -- positive mean deviation (clubhead drifts above the FSP).
  Heuristically a **steep** swing for a right-handed golfer.
* **Blue**   -- negative mean deviation (clubhead drifts below the FSP).
  Heuristically a **shallow** swing.

Design by Contract:

* ``render`` requires the viewport to expose ``add_mesh`` -- raises
  ``TypeError`` otherwise.
* ``render`` requires ``fsp_result.plane.normal`` to be a length-3 vector --
  raises ``ValueError`` otherwise.
* Postcondition of ``render``: exactly one mesh handle is held by the
  renderer; the returned handle equals ``self._mesh_handle``.

The renderer is **stateful**: calling ``render`` twice removes the
previous mesh before adding the new one, so repeated re-renders do not
leak handles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Viewport protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Viewport(Protocol):
    """Minimal viewport protocol -- anything with these methods works."""

    def add_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        color: tuple,
        alpha: float,
    ) -> object:
        """Submit a triangular mesh; return an opaque handle."""

    def remove_mesh(self, handle: object) -> None:
        """Remove a previously-added mesh by handle."""


# ---------------------------------------------------------------------------
# Render configuration
# ---------------------------------------------------------------------------


@dataclass
class FspRenderConfig:
    """Visual configuration for the FSP plane mesh.

    Attributes:
        on_plane_color:        RGB triple used when mean deviation is
            within ``on_plane_threshold`` of zero.  Default: green.
        steep_color:           RGB triple for positive mean deviation
            (clubhead above the plane).  Default: orange.
        shallow_color:         RGB triple for negative mean deviation
            (clubhead below the plane).  Default: blue.
        alpha:                 Mesh opacity in ``[0, 1]``.  Default: 0.35.
        plane_size:            Half-side length of the square mesh in
            metres.  Default: 2.0 -- the mesh is 4 m x 4 m.
        on_plane_threshold:    Absolute deviation in metres below which
            the swing is considered on-plane.  Default: 1e-3 (1 mm).
    """

    on_plane_color: tuple = (0.2, 0.8, 0.2)
    steep_color: tuple = (1.0, 0.5, 0.1)
    shallow_color: tuple = (0.2, 0.4, 1.0)
    alpha: float = 0.35
    plane_size: float = 2.0
    on_plane_threshold: float = 1e-3


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class FspRenderer:
    """Translucent FSP mesh renderer.

    The renderer holds a single mesh handle.  Successive ``render`` calls
    replace the previous handle (no leaks), and ``clear`` removes it.
    """

    def __init__(self, config: FspRenderConfig | None = None) -> None:
        self._config: FspRenderConfig = config or FspRenderConfig()
        self._mesh_handle: object | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        viewport: Any,
        fsp_result: Any,
        mean_deviation: float = 0.0,
    ) -> object:
        """Build a square plane mesh aligned with the FSP and add it.

        Args:
            viewport:        Anything implementing :class:`Viewport`.
            fsp_result:      Object with ``plane.normal`` and
                ``plane.centroid`` 3-vectors.
            mean_deviation:  Signed mean clubhead deviation in metres.
                Sign and magnitude (relative to
                ``config.on_plane_threshold``) drive the mesh color.

        Returns:
            The mesh handle returned by ``viewport.add_mesh``.

        Raises:
            TypeError:  If *viewport* does not implement ``add_mesh``.
            ValueError: If ``fsp_result.plane.normal`` is not length-3
                or is zero-length.
        """
        self._require_viewport(viewport)
        normal, centroid = self._extract_plane(fsp_result)

        vertices, faces = self._build_square_mesh(
            normal=normal,
            centroid=centroid,
            half_side=self._config.plane_size,
        )
        color = self._pick_color(mean_deviation)

        # Replace any previous mesh so we never leak handles.
        self._remove_existing(viewport)

        handle = viewport.add_mesh(
            vertices,
            faces,
            color=color,
            alpha=self._config.alpha,
        )
        self._mesh_handle = handle
        logger.debug(
            "FspRenderer.render: handle=%r mean_deviation=%.4f color=%r",
            handle,
            mean_deviation,
            color,
        )
        return handle

    def clear(self, viewport: Any) -> None:
        """Remove the current FSP mesh from *viewport* (no-op if none)."""
        self._remove_existing(viewport)

    # ------------------------------------------------------------------
    # DbC + plane extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_viewport(viewport: Any) -> None:
        if not callable(getattr(viewport, "add_mesh", None)):
            raise TypeError(
                "FspRenderer.render: viewport must implement add_mesh(); "
                f"got {type(viewport).__name__}"
            )

    @staticmethod
    def _extract_plane(fsp_result: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(normal, centroid)`` as float64 arrays of shape (3,)."""
        plane = getattr(fsp_result, "plane", None)
        if plane is None:
            raise ValueError(
                "FspRenderer.render: fsp_result must have a `.plane` attribute"
            )
        normal = np.asarray(plane.normal, dtype=np.float64)
        centroid = np.asarray(plane.centroid, dtype=np.float64)
        if normal.shape != (3,):
            raise ValueError(
                f"FspRenderer.render: plane.normal must be length-3; got shape {normal.shape}"
            )
        if centroid.shape != (3,):
            raise ValueError(
                f"FspRenderer.render: plane.centroid must be length-3; got shape {centroid.shape}"
            )
        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            raise ValueError("FspRenderer.render: plane.normal must be non-zero")
        return normal / norm, centroid

    # ------------------------------------------------------------------
    # Mesh + color helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_square_mesh(
        normal: np.ndarray, centroid: np.ndarray, half_side: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a 4-vertex, 2-triangle square mesh centred on *centroid*.

        The mesh is perpendicular to *normal* with corners at
        ``centroid +/- u * half_side +/- v * half_side`` where ``u`` and
        ``v`` are orthonormal in-plane basis vectors.
        """
        u, v = _orthonormal_basis(normal)
        size = float(half_side)
        c = centroid
        v0 = c + u * size + v * size
        v1 = c - u * size + v * size
        v2 = c - u * size - v * size
        v3 = c + u * size - v * size
        vertices = np.array([v0, v1, v2, v3], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        return vertices, faces

    def _pick_color(self, mean_deviation: float) -> tuple:
        threshold = self._config.on_plane_threshold
        if abs(mean_deviation) <= threshold:
            return self._config.on_plane_color
        if mean_deviation > 0.0:
            return self._config.steep_color
        return self._config.shallow_color

    def _remove_existing(self, viewport: Any) -> None:
        if self._mesh_handle is None:
            return
        try:
            viewport.remove_mesh(self._mesh_handle)
        except Exception as exc:  # pragma: no cover - viewport-specific
            logger.debug("FspRenderer.clear: remove_mesh raised %r", exc)
        self._mesh_handle = None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _orthonormal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal in-plane basis vectors ``(u, v)``.

    Picks a reference direction that is not parallel to *normal*,
    projects out the normal component, normalises to obtain ``u``, then
    sets ``v = normal x u``.
    """
    # Pick the world axis least aligned with the normal as our seed.
    abs_n = np.abs(normal)
    seed_axis = int(np.argmin(abs_n))
    seed = np.zeros(3, dtype=np.float64)
    seed[seed_axis] = 1.0

    u = seed - float(np.dot(seed, normal)) * normal
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1e-12:
        # Fallback: another axis must work.
        seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u = seed - float(np.dot(seed, normal)) * normal
        u_norm = float(np.linalg.norm(u))
    u = u / u_norm
    v = np.cross(normal, u)
    v_norm = float(np.linalg.norm(v))
    if v_norm > 0.0:
        v = v / v_norm
    return u, v
