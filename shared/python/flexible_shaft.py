"""Flexible Beam Shaft Module.

Guideline B5 Implementation: Flexible Beam Shaft.

Provides shaft flexibility modeling options:
- Rigid shaft (baseline)
- Finite element beam model (distributed compliance)
- Modal representation (dominant bending modes)

Shaft properties:
- Stiffness distribution (EI profile along shaft)
- Mass distribution
- Damping characteristics

This module provides the mathematical framework and data structures
for shaft modeling. Physics engine integration is separate.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)

# Standard golf shaft parameters
SHAFT_LENGTH_DRIVER = 1.168  # [m] 46" driver shaft
SHAFT_LENGTH_IRON = 0.965  # [m] 38" 7-iron shaft
STEEL_DENSITY = 7850  # [kg/m³]
GRAPHITE_DENSITY = 1800  # [kg/m³]
STEEL_E = 200e9  # [Pa] Young's modulus for steel
GRAPHITE_E = 130e9  # [Pa] Young's modulus for graphite


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
    youngs_modulus: float = GRAPHITE_E
    density: float = GRAPHITE_DENSITY
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


def compute_section_inertia(
    outer_diameter: float,
    wall_thickness: float,
) -> float:
    """Compute second moment of area for hollow circular section.

    I = π/64 * (D⁴ - d⁴)

    Args:
        outer_diameter: Outer diameter [m]
        wall_thickness: Wall thickness [m]

    Returns:
        Second moment of area [m⁴]
    """
    d_outer = outer_diameter
    d_inner = outer_diameter - 2 * wall_thickness
    d_inner = max(d_inner, 0.0)  # Ensure non-negative

    inertia = np.pi / 64 * (d_outer**4 - d_inner**4)
    return inertia


def compute_section_area(
    outer_diameter: float,
    wall_thickness: float,
) -> float:
    """Compute cross-sectional area for hollow circular section.

    A = π/4 * (D² - d²)

    Args:
        outer_diameter: Outer diameter [m]
        wall_thickness: Wall thickness [m]

    Returns:
        Cross-sectional area [m²]
    """
    d_outer = outer_diameter
    d_inner = outer_diameter - 2 * wall_thickness
    d_inner = max(d_inner, 0.0)

    A = np.pi / 4 * (d_outer**2 - d_inner**2)
    return A


def compute_EI_profile(
    properties: ShaftProperties,
) -> np.ndarray:
    """Compute bending stiffness EI along shaft.

    EI = E * I(x) where I is the section inertia.

    Args:
        properties: Shaft properties

    Returns:
        EI values at each station [N·m²] (N,)
    """
    n_stations = len(properties.station_positions)
    EI = np.zeros(n_stations)

    for i in range(n_stations):
        inertia = compute_section_inertia(
            properties.outer_diameter[i], properties.wall_thickness[i]
        )
        EI[i] = properties.youngs_modulus * inertia

    return EI


def compute_mass_profile(
    properties: ShaftProperties,
) -> np.ndarray:
    """Compute mass per unit length along shaft.

    μ = ρ * A(x)

    Args:
        properties: Shaft properties

    Returns:
        Mass per length at each station [kg/m] (N,)
    """
    n_stations = len(properties.station_positions)
    mass_per_length = np.zeros(n_stations)

    for i in range(n_stations):
        A = compute_section_area(
            properties.outer_diameter[i], properties.wall_thickness[i]
        )
        mass_per_length[i] = properties.density * A

    return mass_per_length


def create_standard_shaft(
    material: ShaftMaterial = ShaftMaterial.GRAPHITE,
    length: float = SHAFT_LENGTH_DRIVER,
    n_stations: int = 11,
    tip_diameter: float = 0.0085,  # [m] 8.5mm tip
    butt_diameter: float = 0.015,  # [m] 15mm butt
    wall_thickness: float = 0.001,  # [m] 1mm wall
) -> ShaftProperties:
    """Create standard tapered golf shaft properties.

    Args:
        material: Shaft material
        length: Total shaft length [m]
        n_stations: Number of stations for property definition
        tip_diameter: Diameter at tip (head end) [m]
        butt_diameter: Diameter at butt (grip end) [m]
        wall_thickness: Wall thickness [m]

    Returns:
        ShaftProperties with linear taper
    """
    # Linear station positions from tip to butt
    stations = np.linspace(0, length, n_stations)

    # Linear taper in diameter
    diameters = np.linspace(tip_diameter, butt_diameter, n_stations)

    # Constant wall thickness (could be varied)
    wall = np.full(n_stations, wall_thickness)

    # Material properties
    if material == ShaftMaterial.STEEL:
        E = STEEL_E
        density = STEEL_DENSITY
    else:
        E = GRAPHITE_E
        density = GRAPHITE_DENSITY

    return ShaftProperties(
        length=length,
        outer_diameter=diameters,
        wall_thickness=wall,
        station_positions=stations,
        material=material,
        youngs_modulus=E,
        density=density,
    )


class ShaftModel(ABC):
    """Abstract base class for shaft flexibility models."""

    @abstractmethod
    def initialize(self, properties: ShaftProperties) -> None:
        """Initialize the model with shaft properties."""

    @abstractmethod
    def get_state(self) -> ShaftState:
        """Get current shaft deformation state."""

    @abstractmethod
    def apply_load(
        self,
        position: float,
        force: np.ndarray,
        moment: np.ndarray | None = None,
    ) -> None:
        """Apply load to shaft at specified position."""

    @abstractmethod
    def step(self, dt: float) -> ShaftState:
        """Advance simulation by dt seconds."""


class RigidShaftModel(ShaftModel):
    """Rigid shaft model (no deformation).

    Serves as baseline for comparison.
    """

    def __init__(self) -> None:
        """Initialize rigid shaft model."""
        self.properties: ShaftProperties | None = None
        self.n_stations = 0

    def initialize(self, properties: ShaftProperties) -> None:
        """Initialize with shaft properties."""
        self.properties = properties
        self.n_stations = len(properties.station_positions)

    def get_state(self) -> ShaftState:
        """Return zero deformation state."""
        return ShaftState(
            deflections=np.zeros(self.n_stations),
            velocities=np.zeros(self.n_stations),
            rotations=np.zeros(self.n_stations),
        )

    def apply_load(
        self,
        position: float,
        force: np.ndarray,
        moment: np.ndarray | None = None,
    ) -> None:
        """Loads have no effect on rigid shaft."""

    def step(self, dt: float) -> ShaftState:
        """Return unchanged state."""
        return self.get_state()


class ModalShaftModel(ShaftModel):
    """Modal representation of shaft dynamics.

    Uses dominant bending modes to represent shaft flexibility
    with reduced computational cost.
    """

    def __init__(self, n_modes: int = 3) -> None:
        """Initialize modal shaft model.

        Args:
            n_modes: Number of bending modes to include
        """
        self.n_modes = n_modes
        self.properties: ShaftProperties | None = None
        self.modes: list[ShaftMode] = []
        self.modal_coords = np.zeros(n_modes)  # Modal amplitudes
        self.modal_velocities = np.zeros(n_modes)  # Modal velocities
        self.n_stations = 0
        self.time = 0.0

    def initialize(self, properties: ShaftProperties) -> None:
        """Initialize model and compute modes.

        Uses approximate analytical mode shapes for cantilevered beam.
        """
        self.properties = properties
        self.n_stations = len(properties.station_positions)

        # Compute equivalent uniform beam properties for mode estimation
        EI = compute_EI_profile(properties)
        mass = compute_mass_profile(properties)

        EI_avg = float(np.mean(EI))
        mass_avg = float(np.mean(mass))
        L = properties.length

        self.modes = []
        x = properties.station_positions / L  # Normalized position

        # Cantilevered beam mode shape coefficients
        # φ_n(x) approximated by polynomial for first modes
        for n in range(1, self.n_modes + 1):
            # Approximate natural frequency for cantilevered beam
            # ω_n = β_n² * sqrt(EI/(μL⁴))
            beta_n = [1.875, 4.694, 7.855][n - 1] if n <= 3 else (2 * n - 1) * np.pi / 2
            omega = beta_n**2 * np.sqrt(EI_avg / (mass_avg * L**4))
            freq = omega / (2 * np.pi)

            # Simplified mode shape (polynomial approximation)
            mode_shape = x**2 * (3 - 2 * x) if n == 1 else x ** (n + 1)
            mode_shape = mode_shape / np.max(np.abs(mode_shape))  # Normalize

            self.modes.append(
                ShaftMode(
                    frequency=freq,
                    mode_shape=mode_shape,
                    damping_ratio=properties.damping_ratio,
                    description=f"Mode {n} bending",
                )
            )

        self.modal_coords = np.zeros(self.n_modes)
        self.modal_velocities = np.zeros(self.n_modes)

    def get_state(self) -> ShaftState:
        """Get current state by superposing modal contributions."""
        if not self.modes:
            return ShaftState(
                deflections=np.zeros(1),
                velocities=np.zeros(1),
                rotations=np.zeros(1),
            )

        # Superpose mode shapes
        deflections = np.zeros(self.n_stations)
        velocities = np.zeros(self.n_stations)

        for i, mode in enumerate(self.modes):
            deflections += self.modal_coords[i] * mode.mode_shape
            velocities += self.modal_velocities[i] * mode.mode_shape

        # Approximate rotations as derivative of deflection
        # θ ≈ dw/dx
        dx = self.properties.length / (self.n_stations - 1) if self.properties else 1.0
        rotations = np.gradient(deflections, dx)

        return ShaftState(
            deflections=deflections,
            velocities=velocities,
            rotations=rotations,
            modal_amplitudes=self.modal_coords.copy(),
            timestamp=self.time,
        )

    def apply_load(
        self,
        position: float,
        force: np.ndarray,
        moment: np.ndarray | None = None,
    ) -> None:
        """Apply modal forces from physical load."""
        if not self.modes or self.properties is None:
            return

        # Find modal participation for load at position
        L = self.properties.length
        x_norm = position / L

        for i, mode in enumerate(self.modes):
            # Interpolate mode shape at load position
            x_stations = self.properties.station_positions / L
            phi_at_load = float(np.interp(x_norm, x_stations, mode.mode_shape))

            # Modal force = physical force projected onto mode
            # (simplified: only using first component of force)
            modal_force = phi_at_load * np.linalg.norm(force)
            self.modal_coords[i] += modal_force * 1e-6  # Scale factor

    def step(self, dt: float) -> ShaftState:
        """Advance modal coordinates by dt."""
        self.time += dt

        for i, mode in enumerate(self.modes):
            omega = 2 * np.pi * mode.frequency
            zeta = mode.damping_ratio

            # Damped harmonic oscillator: q'' + 2ζωq' + ω²q = 0
            # Semi-implicit Euler
            acc = (
                -2 * zeta * omega * self.modal_velocities[i]
                - omega**2 * self.modal_coords[i]
            )
            self.modal_velocities[i] += acc * dt
            self.modal_coords[i] += self.modal_velocities[i] * dt

        return self.get_state()


def compute_static_deflection(
    properties: ShaftProperties,
    load_position: float,
    load_force: float,
) -> np.ndarray:
    """Compute static deflection for cantilever beam with point load.

    Assumes cantilevered at butt end, load applied at position.
    Uses Euler-Bernoulli beam theory.

    For point load P at distance a from fixed end on beam of length L:
    w(x) = Px²(3a-x)/(6EI) for x ≤ a
    w(x) = Pa²(3x-a)/(6EI) for x > a

    Args:
        properties: Shaft properties
        load_position: Position of load from butt end [m]
        load_force: Load magnitude [N]

    Returns:
        Deflection at each station [m]
    """
    EI = compute_EI_profile(properties)
    EI_avg = float(np.mean(EI))  # Use average for simplicity

    stations = properties.station_positions
    a = load_position  # Load position from butt (fixed end)

    deflection = np.zeros(len(stations))

    for i, x in enumerate(stations):
        if x <= a:
            deflection[i] = load_force * x**2 * (3 * a - x) / (6 * EI_avg)
        else:
            deflection[i] = load_force * a**2 * (3 * x - a) / (6 * EI_avg)

    return deflection


def create_shaft_model(
    model_type: ShaftFlexModel,
    properties: ShaftProperties | None = None,
) -> ShaftModel:
    """Factory function to create shaft model.

    Args:
        model_type: Type of shaft model
        properties: Shaft properties (uses default if None)

    Returns:
        Initialized shaft model
    """
    if properties is None:
        properties = create_standard_shaft()

    if model_type == ShaftFlexModel.RIGID:
        model = RigidShaftModel()
    elif model_type == ShaftFlexModel.MODAL:
        model = ModalShaftModel()
    else:
        # Finite element not yet fully implemented
        LOGGER.warning("FE model not fully implemented, using modal")
        model = ModalShaftModel()

    model.initialize(properties)
    return model
