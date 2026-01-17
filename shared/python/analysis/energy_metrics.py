"""Energy metrics analysis module."""

from typing import Any

import numpy as np


class EnergyMetricsMixin:
    """Mixin for computing energy-related metrics."""

    def compute_energy_metrics(
        self,
        kinetic_energy: np.ndarray,
        potential_energy: np.ndarray,
    ) -> dict[str, Any]:
        """Compute energy-related metrics.

        Args:
            kinetic_energy: Kinetic energy time series
            potential_energy: Potential energy time series

        Returns:
            Dictionary of energy metrics
        """
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

        return {
            "max_kinetic_energy": float(np.max(kinetic_energy)),
            "max_potential_energy": float(np.max(potential_energy)),
            "max_total_energy": float(np.max(total_energy)),
            "energy_efficiency": float(efficiency),
            "energy_variation": float(energy_variation),
            "energy_drift": float(energy_drift),
        }
