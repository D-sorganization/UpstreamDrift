"""
Angle of repose calibration experiment.
"""

from __future__ import annotations


class AngleOfReposeExperiment:
    """Simulates pouring particles from a lifted cylinder to measure final pile angle."""

    def __init__(self, backend: str = "chrono") -> None:
        """
        Initialize the experiment.

        Args:
            backend: The simulator backend to use (chrono, liggghts, mpm)
        """
        self.backend = backend
        self.target_angle = 32.0  # degrees

    def run_simulation(self, params: dict) -> float:
        """
        Run the calibration experiment for a given parameter set.

        Args:
            params: Simulation parameters (e.g. friction_coefficient).

        Returns:
            The measured angle of repose in degrees.

        Raises:
            NotImplementedError: Angle of repose calibration requires real
                LIGGGHTS output. Stub removed in #5486.
        """
        raise NotImplementedError(  # tracked: #5486
            "Angle of repose calibration requires real LIGGGHTS output. "
            "Stub removed in #5486."
        )

    def calibrate(self) -> dict:
        """
        Run Bayesian optimization / CMA-ES to find optimal parameters.

        Raises:
            NotImplementedError: Calibration requires a working run_simulation.
        """
        raise NotImplementedError(  # tracked: #5486
            "calibrate() requires a working run_simulation(). Stub removed in #5486."
        )


def compute_angle_of_repose(friction: float, backend: str = "chrono") -> float:
    """Convenience function: run a single angle-of-repose measurement.

    Args:
        friction: Friction coefficient for the granular material.
        backend: Simulator backend to use.

    Returns:
        Measured angle of repose in degrees.

    Raises:
        NotImplementedError: Angle of repose calibration requires real
            LIGGGHTS output. Stub removed in #5486.
    """
    experiment = AngleOfReposeExperiment(backend=backend)
    return experiment.run_simulation({"friction_coefficient": friction})
