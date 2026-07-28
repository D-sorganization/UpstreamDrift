import importlib

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
    assert "error" in best_params
    # #7999: restitution is not read by any experiment, so it is no longer
    # optimised over and must not be reported as if it had been measured.
    assert "restitution_coefficient" not in best_params

    assert np.isclose(best_params["friction_coefficient"], 0.5, atol=0.05)


def test_drained_shear_cell_optimization() -> None:
    exp = DrainedShearCellExperiment(backend="mock")
    # Target peak is 35.0. Mock: 20.0 + (f * 30.0) => 35 = 20 + 30f => f = 0.5

    optimizer = CalibrationOptimizer(exp)
    best_params = optimizer.optimize()

    assert np.isclose(best_params["friction_coefficient"], 0.5, atol=0.05)


def test_wrench_trace_import_does_not_require_scipy_optimize(monkeypatch) -> None:
    """Cross-engine imports should not need calibration optimizer extras."""

    real_import = __import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy.optimize":
            raise ModuleNotFoundError("No module named 'scipy.optimize'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    module = importlib.import_module("src.bunkershot3d.postproc.wrench_trace")

    assert hasattr(module, "WrenchTrace")
