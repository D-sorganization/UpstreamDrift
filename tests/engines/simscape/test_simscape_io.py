"""Tests for src.engines.simscape._simscape_io."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.simscape._errors import SimscapeSimulationError
from src.engines.simscape._output import SimscapeOutput
from src.engines.simscape._simscape_io import (
    _to_matlab_double,
    build_simulation_input,
    logsout_to_simscape_output,
)


def _valid_logsout(n: int = 3, n_joints: int = 2) -> dict:
    return {
        "time": np.linspace(0.0, 0.2, n),
        "q": np.zeros((n, n_joints)),
        "qd": np.zeros((n, n_joints)),
        "qdd": np.zeros((n, n_joints)),
        "tau": np.zeros((n, n_joints)),
        "omega": np.zeros((n, n_joints)),
        "r_butt": np.zeros((n, 3)),
        "r_clubhead": np.zeros((n, 3)),
        "q_club": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        "v_clubhead": np.zeros((n, 3)),
    }


def test_to_matlab_double_uses_mocked_matlab() -> None:
    fake_matlab = MagicMock()
    fake_matlab.double = MagicMock(return_value="ML_DOUBLE")
    with patch.dict(sys.modules, {"matlab": fake_matlab}):
        result = _to_matlab_double(np.array([1.0, 2.0, 3.0]))
    assert result == "ML_DOUBLE"
    fake_matlab.double.assert_called_once()
    arg = fake_matlab.double.call_args[0][0]
    assert arg == [1.0, 2.0, 3.0]


def test_build_simulation_input_happy_path() -> None:
    eng = MagicMock()
    eng.eval.return_value = "SIM_INPUT_V1"
    eng.setVariable.return_value = "SIM_INPUT_V2"
    fake_matlab = MagicMock()
    fake_matlab.double = MagicMock(return_value="DBL")
    coeffs = np.arange(14, dtype=np.float64)
    with patch.dict(sys.modules, {"matlab": fake_matlab}):
        out = build_simulation_input(
            eng, model_name="MyModel", coeffs=coeffs, n_joints=2
        )
    assert out == "SIM_INPUT_V2"
    eng.eval.assert_called_once()
    assert "Simulink.SimulationInput('MyModel')" in eng.eval.call_args[0][0]
    eng.setVariable.assert_called_once_with(
        "SIM_INPUT_V1", "PolynomialInputs", "DBL", nargout=1
    )


def test_build_simulation_input_rejects_non_1d() -> None:
    eng = MagicMock()
    with pytest.raises(ValueError, match="1-D"):
        build_simulation_input(eng, model_name="M", coeffs=np.zeros((2, 7)), n_joints=2)


def test_build_simulation_input_rejects_wrong_size() -> None:
    eng = MagicMock()
    with pytest.raises(ValueError, match="n_joints"):
        build_simulation_input(eng, model_name="M", coeffs=np.zeros(13), n_joints=2)


def test_build_simulation_input_rejects_non_finite() -> None:
    eng = MagicMock()
    coeffs = np.zeros(14)
    coeffs[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_simulation_input(eng, model_name="M", coeffs=coeffs, n_joints=2)


def test_build_simulation_input_wraps_matlab_failure() -> None:
    eng = MagicMock()
    eng.eval.side_effect = RuntimeError("simulink boom")
    fake_matlab = MagicMock()
    fake_matlab.double = MagicMock(return_value="DBL")
    with (
        patch.dict(sys.modules, {"matlab": fake_matlab}),
        pytest.raises(SimscapeSimulationError, match="simulink boom"),
    ):
        build_simulation_input(
            eng,
            model_name="M",
            coeffs=np.zeros(7, dtype=np.float64),
            n_joints=1,
        )


def test_logsout_to_output_happy_path() -> None:
    out = logsout_to_simscape_output(_valid_logsout())
    assert isinstance(out, SimscapeOutput)
    assert out.n_samples == 3
    assert out.n_joints == 2


def test_logsout_missing_field_raises() -> None:
    bad = _valid_logsout()
    del bad["tau"]
    with pytest.raises(SimscapeSimulationError, match="missing required field"):
        logsout_to_simscape_output(bad)


def test_logsout_invariant_failure_wrapped() -> None:
    bad = _valid_logsout()
    bad["q_club"] = np.tile([2.0, 0.0, 0.0, 0.0], (3, 1))  # not unit-norm
    with pytest.raises(SimscapeSimulationError, match="SimscapeOutput validation"):
        logsout_to_simscape_output(bad)


def test_logsout_promotes_1d_joint_arrays() -> None:
    # 1-D joint columns get reshaped to (N,1)
    n = 3
    raw = _valid_logsout(n=n, n_joints=1)
    raw["q"] = np.zeros(n)  # 1-D
    raw["qd"] = np.zeros(n)
    raw["qdd"] = np.zeros(n)
    raw["tau"] = np.zeros(n)
    raw["omega"] = np.zeros(n)
    out = logsout_to_simscape_output(raw)
    assert out.n_joints == 1
