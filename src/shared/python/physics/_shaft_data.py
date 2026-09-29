from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from src.shared.python.core.physics_constants import GRAPHITE_DENSITY_KG_M3

_GRAPHITE_DENSITY = int(GRAPHITE_DENSITY_KG_M3)
_GRAPHITE_E = 130e9


class ShaftFlexModel(Enum):
    """Shaft flexibility model types."""

    RIGID = auto()  # No deformation
    MODAL = auto()  # Modal representation (dominant modes)
    FINITE_ELEMENT = auto()  # Distributed compliance beam elements


class ShaftMaterial(Enum):
    """Standard shaft materials."""

    STEEL = auto()
    GRAPHITE = auto()
    COMPOSITE = auto()


@dataclass
class ShaftProperties:
    """Physical properties of a golf shaft.

    Attributes:
        length: Total shaft length [m]
        outer_diameter: Outer diameter at each section [m] (N,)
        wall_thickness: Wall thickness at each section [m] (N,)
        station_positions: Position along shaft for property values [m] (N,)
        material: Shaft material type
        youngs_modulus: Young's modulus [Pa]
        density: Material density [kg/m³]
        damping_ratio: Material damping ratio [unitless]
    """

    length: float
    outer_diameter: np.ndarray
    wall_thickness: np.ndarray
    station_positions: np.ndarray
    material: ShaftMaterial = ShaftMaterial.GRAPHITE
    youngs_modulus: float = _GRAPHITE_E
    density: float = _GRAPHITE_DENSITY
    damping_ratio: float = 0.02  # Typical structural damping


@dataclass
class BeamElement:
    """Single beam element for finite element model.

    Attributes:
        node_i: Start node index
        node_j: End node index
        length: Element length [m]
        EI: Bending stiffness [N·m²]
        mass_per_length: Linear mass density [kg/m]
        damping: Damping coefficient
    """

    node_i: int
    node_j: int
    length: float
    EI: float
    mass_per_length: float
    damping: float = 0.0


@dataclass
class ShaftMode:
    """Single vibration mode of the shaft.

    Attributes:
        frequency: Natural frequency [Hz]
        mode_shape: Mode shape (displacement at each station) (N,)
        damping_ratio: Modal damping ratio [unitless]
        description: Mode description (e.g., "1st bending")
    """

    frequency: float
    mode_shape: np.ndarray
    damping_ratio: float = 0.02
    description: str = ""


@dataclass
class ShaftState:
    """Current state of deformable shaft.

    Attributes:
        deflections: Transverse deflection at each station [m] (N,)
        velocities: Transverse velocity at each station [m/s] (N,)
        rotations: Section rotations at each station [rad] (N,)
        modal_amplitudes: Modal coordinate amplitudes (M,) if modal model
        timestamp: Current time [s]
    """

    deflections: np.ndarray
    velocities: np.ndarray
    rotations: np.ndarray
    modal_amplitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamp: float = 0.0
