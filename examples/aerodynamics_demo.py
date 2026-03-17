"""Demonstrate AerodynamicsEngine: compute forces for various ball speeds.

Usage::

    python3 examples/aerodynamics_demo.py

Prints drag, lift, Magnus, and total force magnitudes for a range of
ball speeds, and shows the effect of toggling individual force components.
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402

from src.shared.python.physics.aerodynamics import (  # noqa: E402
    AerodynamicsConfig,
    AerodynamicsEngine,
)


def _report_forces(engine: AerodynamicsEngine, speed_ms: float, label: str) -> None:
    """Print force components for a ball travelling at *speed_ms* m/s."""
    assert engine is not None, "Engine must be provided"
    assert speed_ms >= 0.0, "Speed must be non-negative"
    assert label, "Label must not be empty"

    velocity = np.array([speed_ms, 0.0, 0.0])
    spin = np.array([0.0, 300.0, 0.0])  # ~2 870 rpm back-spin

    forces = engine.compute_forces(velocity, spin)
    import numpy.linalg as npla

    drag_n = float(npla.norm(forces["drag"]))
    lift_n = float(npla.norm(forces["lift"]))
    magnus_n = float(npla.norm(forces["magnus"]))
    total_n = float(npla.norm(forces["total"]))

    print(
        f"{label:30s}  speed={speed_ms:5.1f} m/s"
        f"  drag={drag_n:6.3f} N"
        f"  lift={lift_n:6.3f} N"
        f"  magnus={magnus_n:6.3f} N"
        f"  total={total_n:6.3f} N"
    )


def main() -> None:
    """Compare aerodynamic forces with different configurations."""
    # Default config: all effects on
    full_cfg = AerodynamicsConfig()
    engine_full = AerodynamicsEngine(config=full_cfg)

    # Drag-only (lift + Magnus disabled)
    drag_only_cfg = AerodynamicsConfig(lift_enabled=False, magnus_enabled=False)
    engine_drag = AerodynamicsEngine(config=drag_only_cfg)

    print("Aerodynamic force breakdown")
    print("=" * 90)

    for speed in (20.0, 40.0, 60.0, 70.0):
        _report_forces(engine_full, speed, "full (drag+lift+magnus)")
        _report_forces(engine_drag, speed, "drag-only")
        print()

    # Show Reynolds-number correction effect at low vs high speed
    print("Reynolds number correction comparison (70 m/s)")
    print("-" * 60)
    for reynolds in (True, False):
        cfg = AerodynamicsConfig(reynolds_correction_enabled=reynolds)
        engine = AerodynamicsEngine(config=cfg)
        label = f"reynolds_correction={reynolds}"
        _report_forces(engine, 70.0, label)


if __name__ == "__main__":
    main()
