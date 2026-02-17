"""Energy metrics analysis module."""

from typing import Any

import numpy as np

from src.shared.python.core.contracts import ensure, require


class EnergyMetricsMixin:
    """Mixin for computing energy-related metrics."""

    def compute_energy_metrics(
        self,
        kinetic_energy: np.ndarray,
        potential_energy: np.ndarray,
    ) -> dict[str, Any]:
        """Compute energy-related metrics.

        Design by Contract:
            Preconditions:
                - kinetic_energy must be non-empty
                - potential_energy must be non-empty
                - arrays must have the same length
                - all kinetic_energy values must be non-negative
                - all values must be finite
            Postconditions:
                - all returned metric values are finite
                - energy_efficiency is non-negative

        Args:
            kinetic_energy: Kinetic energy time series
            potential_energy: Potential energy time series

        Returns:
            Dictionary of energy metrics
        """
        require(len(kinetic_energy) > 0, "kinetic_energy must be non-empty")
        require(len(potential_energy) > 0, "potential_energy must be non-empty")
        require(
            len(kinetic_energy) == len(potential_energy),
            "kinetic and potential energy arrays must have the same length",
            {"ke_len": len(kinetic_energy), "pe_len": len(potential_energy)},
        )
        require(
            bool(np.all(kinetic_energy >= 0)),
            "kinetic energy must be non-negative",
        )
        require(
            bool(np.all(np.isfinite(kinetic_energy))),
            "kinetic energy must be finite",
        )
        require(
            bool(np.all(np.isfinite(potential_energy))),
            "potential energy must be finite",
        )

        total_energy = kinetic_energy + potential_energy

        # Energy efficiency: ratio of kinetic energy at impact to max total energy
        # Note: self.club_head_speed is expected to be available from the main class
        club_head_speed = getattr(self, "club_head_speed", None)
        if club_head_speed is not None:
            impact_idx = np.argmax(club_head_speed)
            ke_at_impact = kinetic_energy[impact_idx]
            max_total = np.max(total_energy)
            efficiency = (ke_at_impact / max_total * 100) if max_total > 0 else 0.0
        else:
            efficiency = 0.0

        # Energy conservation (should be ~constant for conservative system)
        energy_variation = np.std(total_energy)
        energy_drift = total_energy[-1] - total_energy[0]

        result = {
            "max_kinetic_energy": float(np.max(kinetic_energy)),
            "max_potential_energy": float(np.max(potential_energy)),
            "max_total_energy": float(np.max(total_energy)),
            "energy_efficiency": float(efficiency),
            "energy_variation": float(energy_variation),
            "energy_drift": float(energy_drift),
        }

        # Postcondition: all values must be finite
        for key, val in result.items():
            ensure(np.isfinite(val), f"metric '{key}' must be finite", val)
        ensure(
            result["energy_efficiency"] >= 0,
            "energy_efficiency must be non-negative",
            result["energy_efficiency"],
        )

        return result
