import pytest
from bunkershot3d.calibration.angle_of_repose import AngleOfReposeExperiment
from bunkershot3d.calibration.drained_shear_cell import DrainedShearCellExperiment


def test_angle_of_repose() -> None:
    exp = AngleOfReposeExperiment(backend="mock")
    angle = exp.run_simulation({"friction_coefficient": 0.5})
    assert angle == 32.0  # 20.0 + (0.5 * 24.0)

    best = exp.calibrate()
    assert "friction_coefficient" in best


def test_drained_shear_cell() -> None:
    exp = DrainedShearCellExperiment(backend="mock")
    phi_peak, phi_res = exp.run_simulation({"friction_coefficient": 0.5})
    assert phi_peak == 35.0  # 20.0 + (0.5 * 30.0)
    assert phi_res == 30.0  # 35.0 - 5.0

    best = exp.calibrate()
    assert "friction_coefficient" in best
