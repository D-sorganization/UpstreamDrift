"""Procedurally regenerate the bundled default body-part STL meshes.

Run from this directory:

    python3 _generate.py

All meshes are generated via :mod:`trimesh.creation` primitives, decimated to
at most 5000 triangles, and written next to this script as ``<name>.stl``.
The generator is deterministic (``numpy.random.default_rng(42)``); the
output bytes are stable across runs on the same trimesh version.

Generic, low-poly, public-domain shapes only -- no anatomical data is
captured from any specific person.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import trimesh

_MAX_TRIANGLES = 5000
_HERE = Path(__file__).resolve().parent

# (name, builder) -- each builder returns a trimesh.Trimesh sized in metres.
_RNG = np.random.default_rng(42)


def _ellipsoid(a: float, b: float, c: float, subdivisions: int = 2) -> trimesh.Trimesh:
    """Unit icosphere scaled to the given semi-axes."""
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    sphere.apply_scale((a, b, c))
    return sphere


def _decimate(mesh: trimesh.Trimesh, max_triangles: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= max_triangles:
        return mesh
    # trimesh.simplify_quadric_decimation requires a face-count target.
    decimated = mesh.simplify_quadric_decimation(face_count=max_triangles)
    if len(decimated.faces) > max_triangles:
        # Fall back: clamp to the budget by repeating until under.
        decimated = decimated.simplify_quadric_decimation(face_count=max_triangles)
    return decimated


def _build_head() -> trimesh.Trimesh:
    # Anonymized human-head proxy: ellipsoid, semi-axes from rest_dimensions/2.
    return _ellipsoid(0.09, 0.11, 0.10, subdivisions=2)


def _build_torso() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(0.30, 0.40, 0.20))


def _build_upper_arm() -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=0.04, height=0.30, sections=24)


def _build_forearm() -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=0.035, height=0.27, sections=24)


def _build_hand() -> trimesh.Trimesh:
    # Ellipsoid semi-axes from rest_dimensions/2.
    return _ellipsoid(0.05, 0.04, 0.015, subdivisions=2)


def _build_thigh() -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=0.06, height=0.45, sections=32)


def _build_shin() -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=0.05, height=0.42, sections=32)


def _build_foot() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(0.25, 0.10, 0.07))


_BUILDERS: tuple[tuple[str, Callable[[], trimesh.Trimesh]], ...] = (
    ("head", _build_head),
    ("torso", _build_torso),
    ("upper_arm", _build_upper_arm),
    ("forearm", _build_forearm),
    ("hand", _build_hand),
    ("thigh", _build_thigh),
    ("shin", _build_shin),
    ("foot", _build_foot),
)


def generate_all(out_dir: Path = _HERE) -> dict[str, int]:
    """Regenerate every default mesh; return {name: triangle_count}."""
    # Touch the RNG so the determinism contract is honoured even when the
    # builders themselves do not consume random numbers (future-proofing).
    _RNG.standard_normal(1)

    counts: dict[str, int] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, build in _BUILDERS:
        mesh = build()
        mesh = _decimate(mesh, _MAX_TRIANGLES)
        # Re-centre on bbox centre so importers do not have to assume.
        mesh.apply_translation(-mesh.bounding_box.centroid)
        path = out_dir / f"{name}.stl"
        mesh.export(str(path))
        counts[name] = int(len(mesh.faces))
    return counts


def main() -> None:
    counts = generate_all()
    width = max(len(n) for n in counts)
    for name, tris in counts.items():
        sys.stdout.write(f"  {name:<{width}}  {tris:>5d} triangles\n")


if __name__ == "__main__":
    main()
