import numpy as np
from bunkershot3d.calibration.optimizer import CalibrationOptimizer
from bunkershot3d.calibration.angle_of_repose import AngleOfReposeExperiment
from bunkershot3d.calibration.drained_shear_cell import DrainedShearCellExperiment


def test_angle_of_repose_optimization() -> None:
    exp = AngleOfReposeExperiment(backend="mock")
    # Target is 32.0. Mock relation: 20.0 + (friction * 24.0)
    # 32 = 20 + 24f => f = 12/24 = 0.5

    optimizer = CalibrationOptimizer(exp)
    best_params = optimizer.optimize()

    assert "friction_coefficient" in best_params
    assert "restitution_coefficient" in best_params
    assert "error" in best_params

    assert np.isclose(best_params["friction_coefficient"], 0.5, atol=0.05)


def test_drained_shear_cell_optimization() -> None:
    exp = DrainedShearCellExperiment(backend="mock")
    # Target peak is 35.0. Mock: 20.0 + (f * 30.0) => 35 = 20 + 30f => f = 0.5

    optimizer = CalibrationOptimizer(exp)
    best_params = optimizer.optimize()

    assert np.isclose(best_params["friction_coefficient"], 0.5, atol=0.05)
