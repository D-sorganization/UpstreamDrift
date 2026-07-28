import numpy as np
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


# ---------------------------------------------------------------------------
# Tests for issue #6644 F5 — calibration optimizer removes internal clip
# ---------------------------------------------------------------------------


def test_optimizer_objective_does_not_clip_internally() -> None:
    """F5: _objective must pass raw params to experiment without clipping them."""
    from unittest.mock import MagicMock

    from bunkershot3d.calibration.optimizer import CalibrationOptimizer

    received = {}
    exp = MagicMock()
    exp.target_angle = 30.0
    exp.run_simulation.side_effect = lambda params: received.update(params) or 28.0

    opt = CalibrationOptimizer(exp)
    # Call with values below the old clip threshold (0.01)
    x = np.array([0.005, 0.003])
    opt._objective(x)

    # Without internal clipping, the experiment receives the raw 0.005 / 0.003 values
    assert abs(received["friction_coefficient"] - 0.005) < 1e-9, (
        f"Expected 0.005 passed through; got {received['friction_coefficient']}"
    )
    assert abs(received["restitution_coefficient"] - 0.003) < 1e-9, (
        f"Expected 0.003 passed through; got {received['restitution_coefficient']}"
    )


def test_optimizer_converges_without_clip() -> None:
    """F5: CalibrationOptimizer.optimize() still converges after removing internal clip."""
    from bunkershot3d.calibration.optimizer import CalibrationOptimizer

    exp = AngleOfReposeExperiment(backend="mock")
    opt = CalibrationOptimizer(exp)
    result = opt.optimize()
    assert "friction_coefficient" in result
    assert "error" in result
    assert 0.01 <= result["friction_coefficient"] <= 1.0


@pytest.mark.slow
def test_angle_of_repose_mujoco_physical_bounds() -> None:
    """Real MuJoCo hopper experiment returns an angle within physical bounds (5–50 deg)."""
    from bunkershot3d.calibration.angle_of_repose import _mujoco_angle_of_repose

    angle = _mujoco_angle_of_repose(friction=0.5, n_grains=80, settle_steps=1500)
    assert 5.0 <= angle <= 50.0, f"Angle {angle:.1f} out of physical range [5, 50] deg"
