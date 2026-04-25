"""Dataset Generator for Neural Network Training.

Produces training datasets from physics engine simulations.
Outputs kinematics, kinetics, and model data into a structured database.
"""

from src.shared.python.dataset_generator.generator import DatasetGenerator

__all__ = ["DatasetGenerator"]
