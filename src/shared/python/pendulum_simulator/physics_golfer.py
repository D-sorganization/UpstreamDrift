"""
Golfer upper-body physics using Lagrangian formulation with a closed kinematic loop.

Topology (from the sketch)
--------------------------
The system is anchored at a fixed pivot (origin). A massless standoff
extends downward to the hub point. The standoff length adjusts where
the center of rotation sits relative to the body's center of mass.

From the hub point, two upper-body (scapula) segments extend to the
shoulder joints via revolute joints. These upper-body segments represent
the torso and carry significant mass (~2× arm mass each).

Each shoulder branches into an independent arm chain (upper arm → forearm →
hand). Both wrist endpoints attach to different points on a shared club
segment, closing the kinematic loop.

    Origin (fixed pivot)
      │
      │  Standoff (massless, adjustable length for COM offset)
      │
    Hub ────── R UBody (upper body R) ──── RS (right shoulder)
      │                                      │
      │                                     RE (right elbow)
      │                                      │
      │                                     RH (right hand)──┐
      │                                                      Club ── Clubhead
      │                                     LH (left hand)───┘
      │                                      │
      │                                     LE (left elbow)
      │                                      │
      └────── L UBody (upper body L) ──── LS (left shoulder)

Segments (up to 10 total):
    1. Standoff (origin → hub point) — massless, for COM rotation adjustment
    2. R Upper Body / Scapula (hub → right shoulder) — represents right torso
    3. Right Shoulder → Right Elbow (right upper arm)
    4. Right Elbow → Right Hand (right forearm)
    5. L Upper Body / Scapula (hub → left shoulder) — represents left torso
    6. Left Shoulder → Left Elbow (left upper arm)
    7. Left Elbow → Left Hand (left forearm)
    8. Club (shaft + clubhead)

Generalized coordinates (open-chain, before constraint):
    q = [theta_hub,          # hub rotation (absolute, from downward vertical)
         alpha_rs,           # right shoulder relative angle
         alpha_re,           # right elbow relative angle
         alpha_rh,           # right wrist relative angle
         alpha_ls,           # left shoulder relative angle
         alpha_le,           # left elbow relative angle
         alpha_lh,           # left wrist relative angle
         theta_club]         # club absolute angle

The closed loop imposes a holonomic constraint:
    Right-hand grip position == Club grip point (right)
    Left-hand grip position  == Club grip point (left)
This gives 2 × 2 = 4 scalar constraints on 8 DOFs → 4 independent DOFs.

We use a minimal-coordinate formulation with constraint enforcement via
the augmented Lagrangian / Baumgarte stabilization approach.

Coordinate convention:
    Angles measured from downward vertical, positive counterclockwise.
    World frame: x→right, y→up, origin at hub.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from . import native_backend as _native_backend
from .constants import GRAVITY_MSS

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GolferParams:
    """Immutable physical parameters for the golfer upper-body model.

    Contract:
        - All lengths and masses must be strictly positive.
        - Gravity must be non-negative.
        - Grip offsets must be non-negative and sum ≤ club length.
    """

    # Segment masses (kg)
    m_hub: float  # standoff mass — should be ~0 (massless, for COM offset only)
    m_r_upper: float  # right upper arm
    m_r_fore: float  # right forearm
    m_l_upper: float  # left upper arm
    m_l_fore: float  # left forearm
    m_club: float  # club shaft + head

    # Segment lengths (m)
    L_hub: float  # standoff length (adjusts COM rotation center)
    L_r_upper: float  # right upper arm length
    L_r_fore: float  # right forearm length
    L_l_upper: float  # left upper arm length
    L_l_fore: float  # left forearm length
    L_club: float  # club total length

    # Shoulder offsets from hub (m) — how far RS and LS are from hub center
    d_rs: float  # distance from hub to right shoulder along hub bar
    d_ls: float  # distance from hub to left shoulder along hub bar

    # Grip positions on club (distance from club base)
    grip_right: float  # right hand grip distance from club base
    grip_left: float  # left hand grip distance from club base

    # Clubhead mass (point mass at tip)
    m_clubhead: float = 0.2

    # Gravity
    g: float = GRAVITY_MSS

    # Dissipation (default: no losses)
    b_hub: float = 0.0
    b_rs: float = 0.0
    b_re: float = 0.0
    b_rh: float = 0.0
    b_ls: float = 0.0
    b_le: float = 0.0
    b_lh: float = 0.0

    # Upper body / scapula joint parameters (#1104)
    # These segments connect the hub to the shoulder joints via revolute joints.
    # They represent the upper torso and should carry significant mass (~2× arm mass).
    # When L_rscap/L_lscap = 0, the upper body segment is absent (backwards compatible).
    L_rscap: float = 0.0  # right upper body segment length
    L_lscap: float = 0.0  # left upper body segment length
    m_rscap: float = 0.0  # right upper body mass (recommended: ~7 kg)
    m_lscap: float = 0.0  # left upper body mass (recommended: ~7 kg)
    b_rscap: float = 0.0  # right upper body damping
    b_lscap: float = 0.0  # left upper body damping

    def __post_init__(self) -> None:
        for name, val in [
            ("m_hub", self.m_hub),
            ("m_r_upper", self.m_r_upper),
            ("m_r_fore", self.m_r_fore),
            ("m_l_upper", self.m_l_upper),
            ("m_l_fore", self.m_l_fore),
            ("m_club", self.m_club),
        ]:
            if not (val > 0):
                raise ValueError(f"{name} must be positive, got {val}")

        for name, val in [
            ("L_hub", self.L_hub),
            ("L_r_upper", self.L_r_upper),
            ("L_r_fore", self.L_r_fore),
            ("L_l_upper", self.L_l_upper),
            ("L_l_fore", self.L_l_fore),
            ("L_club", self.L_club),
        ]:
            if not (val > 0):
                raise ValueError(f"{name} must be positive, got {val}")

        if not (self.d_rs >= 0):
            raise ValueError(f"d_rs must be non-negative, got {self.d_rs}")
        if not (self.d_ls >= 0):
            raise ValueError(f"d_ls must be non-negative, got {self.d_ls}")
        if not (self.grip_right >= 0):
            raise ValueError(f"grip_right must be non-negative, got {self.grip_right}")
        if not (self.grip_left >= 0):
            raise ValueError(f"grip_left must be non-negative, got {self.grip_left}")
        if not (self.grip_right <= self.L_club):
            raise ValueError("grip_right must be ≤ L_club")
        if not (self.grip_left <= self.L_club):
            raise ValueError("grip_left must be ≤ L_club")
        if not (self.g >= 0):
            raise ValueError(f"g must be non-negative, got {self.g}")
        if not (self.m_clubhead >= 0):
            raise ValueError(f"m_clubhead must be non-negative, got {self.m_clubhead}")

        for name in ["b_hub", "b_rs", "b_re", "b_rh", "b_ls", "b_le", "b_lh"]:
            val = getattr(self, name)
            if not (val >= 0):
                raise ValueError(f"{name} must be non-negative, got {val}")


# State: 8 angles + 8 angular velocities = 16 DOF
State = npt.NDArray[np.float64]  # shape (16,)

# Torque function: (t) -> 7 torques (hub, rs, re, rh, ls, le, lh)
TorqueFunc = Callable[[float], tuple[float, float, float, float, float, float, float]]

# Number of generalized coordinates
N_DOF = 8

# Number of constraints (2 loop-closure constraints × 2D = 4)
N_CONSTRAINTS = 4


# Backward compatibility re-exports removed to prevent cyclic import (Issue TDD resolution)
# Restored via lazy __getattr__ below to avoid circular imports while
# maintaining API compatibility for existing callers.

# Mapping from old name → (module, actual_name)
_LAZY_REEXPORTS: dict[str, tuple[str, str]] = {
    # golfer_kinematics
    "forward_kinematics": (".golfer_kinematics", "forward_kinematics"),
    # golfer_constraints
    "constraint_vector": (".golfer_constraints", "constraint_vector"),
    "constraint_jacobian": (
        ".golfer_constraints",
        "analytical_constraint_jacobian",
    ),
    "analytical_constraint_jacobian": (
        ".golfer_constraints",
        "analytical_constraint_jacobian",
    ),
    "numerical_constraint_jacobian": (
        ".golfer_constraints",
        "numerical_constraint_jacobian",
    ),
    "friction_torque_vector": (".golfer_constraints", "friction_torque_vector"),
    "net_joint_forces": (".golfer_constraints", "net_joint_forces"),
    # golfer_dynamics
    "mass_matrix": (".golfer_dynamics", "analytical_mass_matrix"),
    "analytical_mass_matrix": (".golfer_dynamics", "analytical_mass_matrix"),
    "gravity_vector": (".golfer_dynamics", "analytical_gravity_vector"),
    "analytical_gravity_vector": (
        ".golfer_dynamics",
        "analytical_gravity_vector",
    ),
    "coriolis_matrix": (".golfer_dynamics", "analytical_coriolis"),
    "analytical_coriolis": (".golfer_dynamics", "analytical_coriolis"),
    "analytical_fk_jacobians": (".golfer_dynamics", "analytical_fk_jacobians"),
    "kinetic_energy": (".golfer_dynamics", "kinetic_energy"),
    "potential_energy": (".golfer_dynamics", "potential_energy"),
    "potential_energy_from_q": (".golfer_dynamics", "potential_energy_from_q"),
    "total_energy": (".golfer_dynamics", "total_energy"),
}


def __getattr__(name: str) -> Any:
    """Lazy re-export for backward compatibility.

    Defers import of sub-modules until first access, preventing
    circular import errors while preserving the public API.
    """
    if name in _LAZY_REEXPORTS:
        module_path, attr_name = _LAZY_REEXPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, package=__package__)
        value = getattr(mod, attr_name)
        # Cache in module namespace for subsequent accesses
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_native_backend_info() -> dict[str, object]:
    """Expose the golfer native-backend configuration and availability."""
    return _native_backend.get_native_backend_info()
