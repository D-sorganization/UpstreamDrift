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


# ---------------------------------------------------------------------------
# Tests for issue #5554 fixes
# ---------------------------------------------------------------------------


def test_angle_of_repose_mock_formula() -> None:
    """Mock path returns the expected analytical value."""
    from bunkershot3d.calibration.angle_of_repose import AngleOfReposeExperiment

    exp = AngleOfReposeExperiment(backend="mock")
    assert exp._use_mock is True
    angle = exp.run_simulation({"friction_coefficient": 0.3})
    assert abs(angle - (20.0 + 0.3 * 24.0)) < 1e-9


def test_angle_of_repose_use_mock_override() -> None:
    """use_mock=True forces mock path regardless of backend string."""
    from bunkershot3d.calibration.angle_of_repose import AngleOfReposeExperiment

    exp = AngleOfReposeExperiment(backend="mpm", use_mock=True)
    assert exp._use_mock is True
    angle = exp.run_simulation({"friction_coefficient": 0.5})
    assert abs(angle - 32.0) < 1e-9


def test_angle_of_repose_bad_backend() -> None:
    """Unsupported backend raises BackendNotImplementedError."""
    from bunkershot3d.calibration.angle_of_repose import AngleOfReposeExperiment
    from bunkershot3d.exceptions import BackendNotImplementedError

    with pytest.raises(BackendNotImplementedError):
        AngleOfReposeExperiment(backend="liggghts")


@pytest.mark.slow
def test_angle_of_repose_mujoco_physical_bounds() -> None:
    """Real MuJoCo hopper experiment returns an angle within physical bounds (5–50 deg)."""
    from bunkershot3d.calibration.angle_of_repose import _mujoco_angle_of_repose

    angle = _mujoco_angle_of_repose(friction=0.5, n_grains=80, settle_steps=1500)
    assert 5.0 <= angle <= 50.0, f"Angle {angle:.1f} out of physical range [5, 50] deg"
