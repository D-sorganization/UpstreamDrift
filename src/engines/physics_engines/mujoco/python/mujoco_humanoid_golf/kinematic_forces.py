"""Kinematic-dependent force analysis for golf swing biomechanics (coordinator).

See module docstring in _kinematic_force_data.py for unit conventions,
coordinate frame definitions, numerical tolerances, and known limitations.

Implementation split across:
- _kinematic_force_data.py: MjDataContext, KinematicForceData, version check, CSV export
- _kfa_core.py: _KFACoreMixin — init, body lookup, Jacobian helpers
- _kfa_forces.py: _KFAForcesMixin — Coriolis, gravity, mass matrix computations
- _kfa_analysis.py: _KFAAnalysisMixin — power, kinetic energy, trajectory analysis
- _kfa_effective_mass.py: _KFAEffectiveMassMixin — effective mass computation
"""

from __future__ import annotations

import mujoco

from ._kfa_analysis import _KFAAnalysisMixin
from ._kfa_core import _KFACoreMixin
from ._kfa_effective_mass import _KFAEffectiveMassMixin
from ._kfa_forces import _KFAForcesMixin

# Re-export public names for backward compatibility
from ._kinematic_force_data import (
    KinematicForceData,
    MjDataContext,
    export_kinematic_forces_to_csv,
)


class KinematicForceAnalyzer(
    _KFACoreMixin,
    _KFAForcesMixin,
    _KFAAnalysisMixin,
    _KFAEffectiveMassMixin,
):
    """Analyze kinematic-dependent forces in golf swing.

    This class computes Coriolis, centrifugal, and other velocity-dependent
    forces that can be determined from kinematics alone. These forces are
    essential for understanding swing dynamics without requiring full
    inverse dynamics.

    Key Applications:
    - Analyze forces in captured motion data (from motion capture)
    - Understand velocity-dependent effects
    - Study energy transfer mechanisms
    - Evaluate dynamic coupling between joints
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize kinematic force analyzer.

        MEMORY ALLOCATION (Issue A-004): This initializer allocates a complete
        MjData structure for scratch computations, which can be several MB for
        complex models. This is intentional for thread safety and performance,
        but users should be aware of the memory footprint when creating multiple
        analyzer instances.

        Memory usage breakdown (approximate):
        - _perturb_data: ~size_of(MjData) ≈ O(nv² + nbody)
        - Jacobian buffers: 2 × 3 × nv floats ≈ 24×nv bytes
        - Total additional memory: ~few MB for typical humanoid models

        For memory-constrained environments, consider:
        - Reusing a single analyzer instance across analyses
        - Using lazy initialization (allocate _perturb_data on first use)
        - Sharing analyzer instances across threads (with proper synchronization)

        Args:
            model: MuJoCo model
            data: MuJoCo data (shared reference, not modified by compute methods)
        """
        self._init_core(model, data)


__all__ = [
    "KinematicForceAnalyzer",
    "KinematicForceData",
    "MjDataContext",
    "export_kinematic_forces_to_csv",
]
