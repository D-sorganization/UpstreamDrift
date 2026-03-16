"""Demonstrate TopographyData: load terrains and sample elevations.

Usage::

    python3 examples/topography_demo.py

Creates three terrain types (flat, sloped, undulating), queries elevation
at a grid of sample points, and prints a small ASCII cross-section.
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402

from src.shared.python.physics.topography import (  # noqa: E402
    create_flat_terrain,
    create_sloped_terrain,
    create_undulating_terrain,
)


def _ascii_profile(terrain, y: float = 50.0, n_samples: int = 20) -> str:
    """Return a one-line ASCII elevation profile along x at fixed y."""
    assert terrain is not None, "Terrain object must be provided"
    assert n_samples > 1, "n_samples must be greater than 1"
    xs = np.linspace(0, 100, n_samples)
    elevations = [terrain.get_elevation_at(np.array([x, y])) for x in xs]
    e_min, e_max = min(elevations), max(elevations)
    span = max(e_max - e_min, 0.01)
    chars = " ._-=+*#@"
    result = ""
    for e in elevations:
        idx = int((e - e_min) / span * (len(chars) - 1))
        result += chars[idx]
    return result


def _show_terrain(name: str, terrain) -> None:
    assert name, "Name must not be empty"
    assert terrain is not None, "Terrain object must be provided"
    print(f"\n--- {name} ---")
    xs = np.linspace(0, 100, 6)
    ys = np.linspace(0, 100, 6)
    print(f"  {'x':>6}  {'y':>6}  {'elev (m)':>10}")
    for x in xs:
        for y in ys:
            elev = terrain.get_elevation_at(np.array([x, y]))
            print(f"  {x:6.1f}  {y:6.1f}  {elev:10.3f}")
    print(f"  ASCII profile (y=50): {_ascii_profile(terrain)}")


def main() -> None:
    """Load and sample three terrain types."""
    flat = create_flat_terrain(width=100.0, height=100.0, elevation=5.0)
    _show_terrain("Flat terrain (elevation=5 m)", flat)

    sloped = create_sloped_terrain(
        width=100.0,
        height=100.0,
        slope_direction=np.array([1.0, 0.0]),
        slope_magnitude=0.05,
        base_elevation=0.0,
    )
    _show_terrain("Sloped terrain (5% grade downhill in x)", sloped)

    undulating = create_undulating_terrain(
        width=100.0,
        height=100.0,
        amplitude=2.0,
        wavelength=30.0,
        base_elevation=0.0,
    )
    _show_terrain("Undulating terrain (amplitude=2 m, λ=30 m)", undulating)

    # Combining terrain + ball landing check
    print("\n--- Landing detection on sloped terrain ---")
    print("(Ball lands when trajectory z ≤ terrain elevation)")
    for x in np.linspace(0, 200, 5):
        elev = sloped.get_elevation_at(np.array([min(x, 99.9), 50.0]))
        print(f"  x={x:6.1f} m  terrain_elev={elev:.2f} m")


if __name__ == "__main__":
    main()
