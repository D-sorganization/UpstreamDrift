"""Marching-cubes mesher and Taubin smoother for the SDF hand.

``scikit-image`` is imported inside :func:`marching_cubes_mesh` so pose and
SDF tests run without the ``sdf-hand`` extra. A missing install raises
``ImportError`` naming that extra.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.hand_backends.sdf_field import (
    Primitive,
    evaluate_hand_sdf,
)


def mesh_is_watertight(faces: np.ndarray) -> bool:
    """True when every triangle edge is shared by exactly two faces."""
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        return False
    edges = np.sort(
        np.stack(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
            axis=1,
        ).reshape(-1, 2),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def taubin_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int,
    lam: float,
    mu: float,
) -> np.ndarray:
    """Taubin low-pass smooth. Moves vertices only, so edge topology is kept.

    Preconditions: ``iterations >= 1`` and ``0 < lam < -mu``.
    Postcondition: same vertex count, finite coordinates.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if not (mu < 0.0 and 0.0 < lam < -mu):
        raise ValueError(f"Taubin needs 0 < lambda < -mu, got lambda={lam}, mu={mu}")
    count = len(vertices)
    # Each undirected edge contributes both directions.
    src = np.stack(
        (faces[:, 0], faces[:, 1], faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 0])
    )
    dst = np.stack(
        (faces[:, 1], faces[:, 0], faces[:, 2], faces[:, 1], faces[:, 0], faces[:, 2])
    )
    matrix = sparse.coo_matrix(
        (np.ones(dst.size, dtype=float), (src.ravel(), dst.ravel())),
        shape=(count, count),
    ).tocsr()
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    degree[degree == 0.0] = 1.0
    neighbor = sparse.csr_matrix(matrix.multiply((1.0 / degree)[:, np.newaxis]))
    smoothed = np.array(vertices, dtype=float, copy=True)
    for _ in range(iterations):
        smoothed = smoothed + lam * (neighbor @ smoothed - smoothed)
        smoothed = smoothed + mu * (neighbor @ smoothed - smoothed)
    if not np.isfinite(smoothed).all():
        raise ValueError("Taubin smooth produced non-finite vertices")
    return smoothed


def _bounds(
    primitives: tuple[Primitive, ...], pad: float
) -> tuple[np.ndarray, np.ndarray]:
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for prim in primitives:
        if prim.shape == "ellipsoid":
            radius = max(prim.radii) + pad
            centre = np.asarray(prim.start, dtype=float)
            lows.append(centre - radius)
            highs.append(centre + radius)
            continue
        radius = max(prim.radius0, prim.radius1, 0.0) + pad
        start = np.asarray(prim.start, dtype=float)
        end = np.asarray(prim.end, dtype=float)
        lows.append(np.minimum(start, end) - radius)
        highs.append(np.maximum(start, end) + radius)
    if not lows:
        raise ValueError("cannot mesh an empty primitive list")
    return np.min(lows, axis=0), np.max(highs, axis=0)


def _sample(
    primitives: tuple[Primitive, ...],
    blends: dict[str, float],
    lo: np.ndarray,
    hi: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    span = hi - lo
    longest = float(np.max(span))
    if longest <= 0.0:
        raise ValueError("SDF bounds have no extent")
    voxel = longest / (resolution - 1)
    counts = np.maximum(np.ceil(span / voxel).astype(int) + 1, 2)
    axes = [np.linspace(lo[i], hi[i], int(counts[i])) for i in range(3)]
    grid = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([axis.ravel() for axis in grid])
    field = evaluate_hand_sdf(primitives, points, blends).reshape(grid[0].shape)
    spacing = np.array([axis[1] - axis[0] for axis in axes], dtype=float)
    return field, spacing


def _boundary_positive(field: np.ndarray) -> bool:
    faces = (
        field[0],
        field[-1],
        field[:, 0],
        field[:, -1],
        field[:, :, 0],
        field[:, :, -1],
    )
    return all(bool(np.nanmin(face) > 0.0) for face in faces)


def _orient_outward(vertices: np.ndarray, faces: np.ndarray) -> Mesh:
    mesh = Mesh(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64))
    if mesh.volume() < 0.0:
        wound = np.ascontiguousarray(mesh.faces[:, ::-1])
        mesh = Mesh(mesh.vertices, wound)
    if mesh.volume() <= 0.0:
        raise ValueError("marched hand mesh does not enclose a positive volume")
    return mesh


def marching_cubes_mesh(
    primitives: tuple[Primitive, ...],
    blends: dict[str, float],
    *,
    resolution: int,
    bounds_pad_m: float,
    taubin_iterations: int,
    taubin_lambda: float,
    taubin_mu: float,
) -> Mesh:
    """Watertight hand mesh from the zero level set.

    Preconditions: ``resolution >= 8`` and a positive bounds pad. The grid is
    expanded once if the isosurface meets the boundary. Postcondition: outward
    winding, positive volume, finite vertices.
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise ImportError(
            "SDF hand meshing needs scikit-image. Install the sdf-hand extra: "
            "pip install 'scikit-image>=0.22'"
        ) from exc
    if resolution < 8:
        raise ValueError(f"resolution must be >= 8, got {resolution}")
    if bounds_pad_m <= 0.0:
        raise ValueError(f"bounds_pad_m must be positive, got {bounds_pad_m!r}")
    pad = float(bounds_pad_m)
    field: np.ndarray | None = None
    origin = np.zeros(3)
    spacing = np.ones(3)
    for attempt in range(2):
        origin, hi = _bounds(primitives, pad)
        field, spacing = _sample(primitives, blends, origin, hi, resolution)
        if _boundary_positive(field) or attempt == 1:
            break
        pad *= 2.0
    if field is None or not _boundary_positive(field):
        raise ValueError(
            "SDF isosurface meets the sample boundary, so the mesh would be open"
        )
    if not np.isfinite(field).all():
        raise ValueError("SDF sample contains non-finite distances")
    try:
        vertices, faces, _normals, _values = marching_cubes(
            field,
            level=0.0,
            spacing=(float(spacing[0]), float(spacing[1]), float(spacing[2])),
            gradient_direction="ascent",
            allow_degenerate=False,
        )
    except (ValueError, RuntimeError) as exc:
        raise ValueError(
            f"marching cubes failed to extract a hand surface: {exc}"
        ) from exc
    vertices = np.asarray(vertices, dtype=float) + origin
    faces = np.asarray(faces, dtype=np.int64)
    smoothed = taubin_smooth(
        vertices,
        faces,
        iterations=taubin_iterations,
        lam=taubin_lambda,
        mu=taubin_mu,
    )
    return _orient_outward(smoothed, faces)


__all__ = ["marching_cubes_mesh", "mesh_is_watertight", "taubin_smooth"]
