"""Pinocchio golf simulation package.

Provides the Pinocchio-based physics analysis tools for golf swing
simulation, including GUI components, analysis controllers, and
visualization mixins.
"""

from src.engines.physics_engines.pinocchio.python.pinocchio_golf.diff_ik import (
    PINOCCHIO_AVAILABLE,
    differential_ik,
    lm_step,
    solve_dual_frame_ik,
)

__all__ = [
    "PINOCCHIO_AVAILABLE",
    "differential_ik",
    "lm_step",
    "solve_dual_frame_ik",
]

# Register the embed adapter with the launcher's embeddable-tool
# registry on import. Wrapped in ``contextlib.suppress(ImportError)``
# so ``import pinocchio_golf`` continues to work in headless contexts
# where PyQt6 / the ``pinocchio`` wheel (transitively pulled in by the
# GUI module) is unavailable. See Subtask 5 / #4998 of EPIC #4993.
import contextlib  # noqa: E402

with contextlib.suppress(ImportError):
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf import (  # noqa: F401
        _embed_adapter,
    )
