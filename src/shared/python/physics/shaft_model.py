"""Shaft model classes: geometry helpers, material profiles, and flexible shaft models.

Provides:
- compute_section_inertia / compute_section_area — cross-section geometry (cached)
- compute_EI_profile / compute_mass_profile — distributed property arrays
- create_standard_shaft — factory for standard tapered golf shaft
- ShaftModel (ABC), RigidShaftModel, ModalShaftModel
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.contracts import check_positive, require
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.physics.shaft_params import (
    GRAPHITE_DENSITY,
    GRAPHITE_E,
    SHAFT_LENGTH_DRIVER,
    STEEL_DENSITY,
    STEEL_E,
    ShaftMaterial,
    ShaftMode,
    ShaftProperties,
    ShaftState,
)

if TYPE_CHECKING:
    ...

logger = get_logger(__name__)


@functools.lru_cache(maxsize=256)
def compute_section_inertia(
    outer_diameter: float,
    wall_thickness: float,
) -> float:
    """Compute second moment of area for hollow circular section. Cached for performance.

    I = π/64 * (D⁴ - d⁴)

    Args:
        outer_diameter: Outer diameter [m]
        wall_thickness: Wall thickness [m]

    Returns:
        Second moment of area [m⁴]
    """
    check_positive(outer_diameter, "outer_diameter")
    check_positive(wall_thickness, "wall_thickness")
    require(
        wall_thickness <= outer_diameter / 2,
        "wall_thickness must not exceed outer_diameter / 2",
        wall_thickness,
    )
    d_outer = outer_diameter
    d_inner = outer_diameter - 2 * wall_thickness
    d_inner = max(d_inner, 0.0)  # Ensure non-negative

    inertia = np.pi / 64 * (d_outer**4 - d_inner**4)
    return float(inertia)


@functools.lru_cache(maxsize=256)
def compute_section_area(
    outer_diameter: float,
    wall_thickness: float,
) -> float:
    """Compute cross-sectional area for hollow circular section. Cached for performance.

    A = π/4 * (D² - d²)

    Args:
        outer_diameter: Outer diameter [m]
        wall_thickness: Wall thickness [m]

    Returns:
        Cross-sectional area [m²]
    """
    check_positive(outer_diameter, "outer_diameter")
    check_positive(wall_thickness, "wall_thickness")
    require(
        wall_thickness <= outer_diameter / 2,
        "wall_thickness must not exceed outer_diameter / 2",
        wall_thickness,
    )
    d_outer = outer_diameter
    d_inner = outer_diameter - 2 * wall_thickness
    d_inner = max(d_inner, 0.0)

    A = np.pi / 4 * (d_outer**2 - d_inner**2)
    return float(A)


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
    require(material is not None, "material must be provided", material)
    check_positive(length, "length")
    require(n_stations >= 2, "n_stations must be at least 2", n_stations)
    check_positive(tip_diameter, "tip_diameter")
    check_positive(butt_diameter, "butt_diameter")
    check_positive(wall_thickness, "wall_thickness")
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
        if properties is None:
            raise ValueError("properties must be provided")
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
        if dt is None:
            raise ValueError("dt must be provided")
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
        if n_modes is None:
            raise ValueError("n_modes must be provided")
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
        if properties is None:
            raise ValueError("properties must be provided")
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
        if position is None:
            raise ValueError("position must be provided")
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
            # ISSUE #2166: Scale factor needs proper modal mass derivation.
            # Current 1e-6 is an ad-hoc value that produces plausible
            # deflections but lacks rigorous justification.
            self.modal_coords[i] += modal_force * 1e-6

    def step(self, dt: float) -> ShaftState:
        """Advance modal coordinates by dt."""
        if dt is None:
            raise ValueError("dt must be provided")
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
