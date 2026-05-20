from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ._shaft_data import ShaftMode, ShaftProperties, ShaftState
from ._shaft_properties import compute_EI_profile, compute_mass_profile


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
            # Scale factor needs proper modal mass derivation (see issue #2166).
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
