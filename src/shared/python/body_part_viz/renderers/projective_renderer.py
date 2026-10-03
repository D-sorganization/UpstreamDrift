"""Original-camera triangle surfaces with deterministic nearest-depth compositing."""

from dataclasses import dataclass
from typing import Protocol
import numpy as np
from ..overlay_options import ShapeOverlayOptions


class SurfaceCamera(Protocol):
    """Canonical projection plus camera coordinates, never an independent camera."""

    def project(self, points: np.ndarray) -> np.ndarray: ...
    def camera_points(self, points: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class SurfaceMesh:
    """One immutable world-space visual proxy, colors in BGR byte order."""

    identity: str
    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[int, int, int]

    def __post_init__(self) -> None:
        if np.asarray(self.vertices).dtype.kind not in "fiu":
            raise TypeError("Surface vertices require real numeric geometry")
        v = np.array(self.vertices, dtype=float, copy=True)
        f = np.array(self.faces, copy=True)
        if not isinstance(self.identity, str) or not self.identity:
            raise ValueError("Surface identity must be a nonempty string")
        if v.ndim != 2 or v.shape[1] != 3 or len(v) < 3 or not np.isfinite(v).all():
            raise ValueError("Surface vertices must be finite (N,3)")
        if f.ndim != 2 or f.shape[1] != 3 or not len(f) or f.dtype.kind not in "iu":
            raise ValueError("Surface faces must be nonempty integer triangles")
        if np.any(f < 0) or np.any(f >= len(v)):
            raise ValueError("Surface faces exceed vertex indices")
        if (
            not isinstance(self.color, tuple)
            or len(self.color) != 3
            or any(type(c) is not int or not 0 <= c <= 255 for c in self.color)
        ):
            raise ValueError("Surface color must be three BGR bytes")
        v.setflags(write=False)
        f.setflags(write=False)
        object.__setattr__(self, "vertices", v)
        object.__setattr__(self, "faces", f)


@dataclass(frozen=True)
class SurfaceLayer:
    """Covered pixels/depth/geometry identities; uncovered depth is infinity."""

    pixels: np.ndarray
    mask: np.ndarray
    depth: np.ndarray
    geometry_ids: np.ndarray

    def __post_init__(self) -> None:
        arrays = [
            np.array(value, copy=True)
            for value in (self.pixels, self.mask, self.depth, self.geometry_ids)
        ]
        pixels, mask, depth, identities = arrays
        if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
            raise ValueError("Surface pixels must be BGR uint8")
        shape = pixels.shape[:2]
        if 0 in shape or any(
            value.shape != shape for value in (mask, depth, identities)
        ):
            raise ValueError("Surface layer dimensions must agree and be nonempty")
        if mask.dtype != np.bool_ or depth.dtype.kind != "f":
            raise ValueError("Surface mask/depth must be bool/float arrays")
        if not np.isfinite(depth[mask]).all() or np.any(depth[mask] <= 0):
            raise ValueError("Covered surface depth must be positive and finite")
        if not np.isposinf(depth[~mask]).all():
            raise ValueError("Uncovered surface depth must be positive infinity")
        if any(not isinstance(value, str) or not value for value in identities[mask]):
            raise ValueError("Covered geometry identities must be nonempty strings")
        if any(value != "" for value in identities[~mask]):
            raise ValueError("Uncovered geometry identities must be empty")
        for name, value in zip(
            ("pixels", "mask", "depth", "geometry_ids"), arrays, strict=True
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass
class _SurfaceBuffers:
    """Mutable raster workspace, sealed into the public result after rendering."""

    pixels: np.ndarray
    mask: np.ndarray
    depth: np.ndarray
    geometry_ids: np.ndarray


def _clipped_triangle(world: np.ndarray, depths: np.ndarray) -> list[np.ndarray]:
    """Clip in world space against camera z=1e-6 before canonical projection."""
    near = 1e-6
    polygon = []
    for i in range(3):
        a, b = world[i - 1], world[i]
        za, zb = depths[i - 1], depths[i]
        if (za >= near) != (zb >= near):
            polygon.append(a + (b - a) * ((near - za) / (zb - za)))
        if zb >= near:
            polygon.append(b)
    return [
        np.asarray([polygon[0], polygon[i], polygon[i + 1]])
        for i in range(1, len(polygon) - 1)
    ]


def _raster_triangle(
    layer: _SurfaceBuffers,
    pixels: np.ndarray,
    depths: np.ndarray,
    mesh: SurfaceMesh,
    grid: tuple[np.ndarray, np.ndarray],
) -> None:
    """Perspective-correct depth evaluated at integer pixel centers."""
    h, w = layer.mask.shape
    lo = np.maximum(np.ceil(pixels.min(axis=0)), [0, 0])
    hi = np.minimum(np.floor(pixels.max(axis=0)), [w - 1, h - 1])
    if np.any(lo > hi):
        return
    a, b, c = pixels
    denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denom) <= 1e-12:
        return
    rows = slice(int(lo[1]), int(hi[1]) + 1)
    columns = slice(int(lo[0]), int(hi[0]) + 1)
    xs, ys = grid[0][rows, columns], grid[1][rows, columns]
    u = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / denom
    v = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / denom
    t = 1 - u - v
    covered = (u >= -1e-10) & (v >= -1e-10) & (t >= -1e-10)
    inv = u / depths[0] + v / depths[1] + t / depths[2]
    z = np.divide(1.0, inv, out=np.full_like(inv, np.inf), where=covered & (inv > 0))
    old = layer.depth[ys, xs]
    win = covered & (z < old - 1e-10)
    layer.depth[ys[win], xs[win]] = z[win]
    layer.pixels[ys[win], xs[win]] = mesh.color
    layer.mask[ys[win], xs[win]] = True
    layer.geometry_ids[ys[win], xs[win]] = mesh.identity


def _raster_faces(
    projected: np.ndarray, faces: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Keep exactly faces with an integer-center bounding box and nonzero area."""
    if not np.isfinite(projected).all():
        raise ValueError("Surface projection must be finite")
    triangles = projected[faces]
    lo = np.maximum(np.ceil(triangles.min(axis=1)), [0, 0])
    hi = np.minimum(np.floor(triangles.max(axis=1)), np.asarray(size) - 1)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    denominator = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (
        a[:, 1] - c[:, 1]
    )
    keep = np.all(lo <= hi, axis=1) & (np.abs(denominator) > 1e-12)
    return faces[keep]


def validate_surface_size(size: tuple[int, int]) -> None:
    """Require canonical positive integer width/height before geometry evaluation."""
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or any(type(v) is not int or v <= 0 for v in size)
    ):
        raise ValueError("Surface dimensions must be positive integer width/height")


def render_surface_layer(
    meshes: tuple[SurfaceMesh, ...], camera: SurfaceCamera, size: tuple[int, int]
) -> SurfaceLayer:
    """Rasterize bound surfaces; sorting makes equal-depth ties deterministic."""
    validate_surface_size(size)
    if not isinstance(meshes, tuple) or any(
        not isinstance(m, SurfaceMesh) for m in meshes
    ):
        raise TypeError("Surface meshes must be a typed tuple")
    if len({m.identity for m in meshes}) != len(meshes):
        raise ValueError("Surface identities must be unique")
    w, h = size
    layer = _SurfaceBuffers(
        np.zeros((h, w, 3), np.uint8),
        np.zeros((h, w), bool),
        np.full((h, w), np.inf),
        np.full((h, w), "", dtype=object),
    )
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid = (grid_x, grid_y)
    for mesh in sorted(meshes, key=lambda m: m.identity):
        depths = camera.camera_points(mesh.vertices)[:, 2]
        if np.all(depths >= 1e-6):
            projected = camera.project(mesh.vertices)
            for face in _raster_faces(projected, mesh.faces, size):
                _raster_triangle(layer, projected[face], depths[face], mesh, grid)
            continue
        for face in mesh.faces:
            for triangle in _clipped_triangle(mesh.vertices[face], depths[face]):
                projected = camera.project(triangle)
                z = camera.camera_points(triangle)[:, 2]
                _raster_triangle(layer, projected, z, mesh, grid)
    return SurfaceLayer(layer.pixels, layer.mask, layer.depth, layer.geometry_ids)


def validate_surface_source(source: np.ndarray) -> None:
    """Require a nonempty original BGR byte image before any geometry evaluation."""
    if not isinstance(source, np.ndarray):
        raise TypeError("Surface source must be a BGR array")
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Source must be nonempty BGR uint8 dimensions")
    if 0 in source.shape[:2]:
        raise ValueError("Source must be nonempty BGR uint8 dimensions")


def composite_surface(
    source: np.ndarray, layer: SurfaceLayer, options: ShapeOverlayOptions
) -> np.ndarray:
    """Blend once over the original pixels; no black background contribution."""
    if not isinstance(options, ShapeOverlayOptions) or not isinstance(
        layer, SurfaceLayer
    ):
        raise TypeError("Surface compositor requires typed layer/options")
    validate_surface_source(source)
    if source.shape != layer.pixels.shape:
        raise ValueError("Source must match surface BGR uint8 dimensions")
    result = source.copy()
    if options.opacity:
        blended = (
            source[layer.mask].astype(float) * (1 - options.opacity)
            + layer.pixels[layer.mask] * options.opacity
        )
        result[layer.mask] = np.rint(blended).astype(np.uint8)
    return result
