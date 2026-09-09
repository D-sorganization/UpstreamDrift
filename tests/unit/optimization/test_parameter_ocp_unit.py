"""Unit tests for Phase 4 parameter OCP contracts and degradation paths.

Executes under `tests/unit` without requiring `bioptim` or `casadi`.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.shared.python.estimation.identifiability import (
    IdentifiabilityGateOptions,
    UnidentifiableParametersError,
)
from src.shared.python.estimation.map_estimator import (
    MapEstimatorResult,
    SharedParameterBlock,
    SharedParameterSpec,
)
from src.shared.python.optimization._swing_models import ClubModel, GolferModel
from src.shared.python.optimization.ocp._compat import (
    BIOPTIM_INSTALL_HINT,
    BioptimNotAvailableError,
    bioptim_available,
)
from src.shared.python.optimization.ocp.parameter_ocp import (
    ParameterBlockBundle,
    ParameterOcpOptions,
    add_parameter_block,
    build_tracking_parameter_ocp,
    solve_tracking_parameter_ocp,
)
from src.shared.python.optimization.ocp.result import tracking_to_map_estimator_result
from src.shared.python.optimization.ocp.symbolic_model import PARAMETER_NAMES
from src.shared.python.optimization.ocp.tracking_ocp import (
    MarkerTargets,
    TrackingResult,
)

pytestmark = pytest.mark.unit


def test_parameter_ocp_options_contracts() -> None:
    opts = ParameterOcpOptions(max_iterations=150, n_integration_steps=3, n_threads=2)
    assert opts.max_iterations == 150
    assert opts.n_integration_steps == 3
    assert opts.n_threads == 2

    with pytest.raises(ValueError):
        ParameterOcpOptions(max_iterations=0)
    with pytest.raises(ValueError):
        ParameterOcpOptions(n_integration_steps=0)
    with pytest.raises(ValueError):
        ParameterOcpOptions(n_threads=-1)


def test_add_parameter_block_rejects_unknown_parameters() -> None:
    specs = [
        SharedParameterSpec(name="invalid_param", initial=1.0),
    ]
    with pytest.raises(ValueError, match="unknown parameters"):
        add_parameter_block(MagicMock(), specs)


def test_add_parameter_block_degradation_without_bioptim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.shared.python.optimization.ocp._compat as compat

    monkeypatch.setattr(compat, "bioptim_available", lambda: False)

    def raise_not_available():
        raise BioptimNotAvailableError(BIOPTIM_INSTALL_HINT)

    monkeypatch.setattr(compat, "require_bioptim", raise_not_available)

    specs = [
        SharedParameterSpec(name="arm_length", initial=0.6),
    ]
    with pytest.raises(BioptimNotAvailableError, match="bioptim"):
        add_parameter_block(MagicMock(), specs)


def test_add_parameter_block_wires_bioptim_structures() -> None:
    mock_biopt = MagicMock()
    mock_param_list = MagicMock()
    mock_bounds_list = {}
    mock_init_list = MagicMock()
    mock_obj_list = MagicMock()

    mock_biopt.ParameterList.return_value = mock_param_list
    mock_biopt.BoundsList.return_value = mock_bounds_list
    mock_biopt.InitialGuessList.return_value = mock_init_list
    mock_biopt.ParameterObjectiveList.return_value = mock_obj_list
    mock_biopt.VariableScaling = lambda key, arr: f"scaling_{key}"

    mock_model = MagicMock()
    mock_model.set_parameter = "mock_setter"

    specs = [
        SharedParameterSpec(
            name="arm_length",
            initial=0.6,
            lower=0.4,
            upper=0.8,
            prior=0.58,
            prior_scale=0.05,
            locked=False,
        ),
        SharedParameterSpec(
            name="trunk_length",
            initial=0.45,
            lower=0.3,
            upper=0.6,
            locked=True,
        ),
    ]

    bundle = add_parameter_block(mock_model, specs, bioptim=mock_biopt)

    assert isinstance(bundle, ParameterBlockBundle)
    assert bundle.free_names == ("arm_length",)
    assert bundle.locked_names == ("trunk_length",)

    # 4-tuple unpacking
    params, bounds, init, objs = bundle
    assert params == mock_param_list
    assert bounds == mock_bounds_list
    assert init == mock_init_list
    assert objs == mock_obj_list

    # ParameterList.add called for free parameter only
    mock_param_list.add.assert_called_once_with(
        name="arm_length",
        function="mock_setter",
        size=1,
        scaling="scaling_arm_length",
    )

    # Bounds set
    np.testing.assert_allclose(mock_bounds_list["arm_length"][0], [0.4])
    np.testing.assert_allclose(mock_bounds_list["arm_length"][1], [0.8])

    # Initial guess set
    mock_init_list.add.assert_called_once()
    call_args = mock_init_list.add.call_args[0]
    assert call_args[0] == "arm_length"
    np.testing.assert_allclose(call_args[1], [0.6])

    # Prior objective added
    mock_obj_list.add.assert_called_once()
    obj_call = mock_obj_list.add.call_args
    assert obj_call.kwargs["key"] == "arm_length"
    assert obj_call.kwargs["quadratic"] is True
    assert obj_call.kwargs["weight"] == pytest.approx(1.0 / (0.05**2))
    np.testing.assert_allclose(obj_call.kwargs["target"], [[0.58]])


def test_tracking_to_map_estimator_result() -> None:
    res = TrackingResult(
        time=np.array([0.0, 0.1, 0.2]),
        q=np.zeros((7, 3)),
        qdot=np.zeros((7, 3)),
        tau=np.zeros((7, 2)),
        parameters={"arm_length": 0.62, "trunk_length": 0.45},
        marker_rms_m={"shoulder": 0.004, "wrist": 0.005},
        status=0,
        iterations=42,
        cost=1.23,
        wall_time_s=0.456,
        locked_by_gate=("trunk_length",),
    )
    map_res = tracking_to_map_estimator_result(res)
    assert isinstance(map_res, MapEstimatorResult)
    assert map_res.success is True
    assert map_res.parameters == {"arm_length": 0.62, "trunk_length": 0.45}
    assert map_res.objective == pytest.approx(1.23)
    assert map_res.n_iterations == 42
    assert map_res.locked_by_gate == ("trunk_length",)
    assert "status 0" in map_res.message
    assert map_res.n_non_finite_evaluations == 0
