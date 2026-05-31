"""Tests for src.shared.python.engine_core.sub_protocols (Issues #1949, #1744)."""

from __future__ import annotations

from typing import Any

import numpy as np
from src.shared.python.engine_core.sub_protocols import (
    CounterfactualComputable,
    DynamicsComputable,
    Loadable,
    Queryable,
    Recordable,
    Steppable,
    SupportsParameterGradients,
)

# ---------------------------------------------------------------------------
# Protocol structural checks
# ---------------------------------------------------------------------------


class TestProtocolsRuntimeCheckable:
    """All sub-protocols must be @runtime_checkable (plain object does not match)."""

    def test_loadable_plain_object_false(self) -> None:
        assert isinstance(object(), Loadable) is False

    def test_steppable_plain_object_false(self) -> None:
        assert isinstance(object(), Steppable) is False

    def test_queryable_plain_object_false(self) -> None:
        assert isinstance(object(), Queryable) is False

    def test_dynamics_computable_plain_object_false(self) -> None:
        assert isinstance(object(), DynamicsComputable) is False

    def test_counterfactual_computable_plain_object_false(self) -> None:
        assert isinstance(object(), CounterfactualComputable) is False

    def test_recordable_plain_object_false(self) -> None:
        assert isinstance(object(), Recordable) is False

    def test_parameter_gradients_plain_object_false(self) -> None:
        assert isinstance(object(), SupportsParameterGradients) is False


# ---------------------------------------------------------------------------
# Mock implementations satisfy their protocols
# ---------------------------------------------------------------------------


class _MockLoadable:
    @property
    def model_name(self) -> str:
        return "test_model"

    def load_from_path(self, path: str) -> None:
        pass

    def load_from_string(self, content: str, fmt: str) -> None:
        pass


class _MockSteppable:
    def step(self, dt: float | None = None) -> None:
        pass

    def reset(self) -> None:
        pass

    def forward(self) -> None:
        pass


class _MockQueryable:
    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.zeros(3)

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        pass

    def set_control(self, u: np.ndarray) -> None:
        pass

    def get_time(self) -> float:
        return 0.0


class _MockRecordable:
    def get_time_series(self, key: str) -> np.ndarray:
        return np.zeros(10)

    def get_induced_acceleration_series(self, body: str) -> np.ndarray:
        return np.zeros(10)

    def set_analysis_config(self, config: dict[str, Any]) -> None:
        pass


class _MockParameterGradients:
    def parameter_jacobian(
        self,
        parameter_vector: np.ndarray,
        q: np.ndarray,
        v: np.ndarray,
        *,
        mode: str = "forward",
    ) -> np.ndarray:
        return np.zeros((2, 5))

    def evaluate_ztcf_parameter_sensitivity_along_trajectory(
        self,
        parameter_vector: np.ndarray,
        q_traj: np.ndarray,
        v_traj: np.ndarray,
        *,
        mode: str = "forward",
    ) -> np.ndarray:
        return np.zeros((len(q_traj), 2, 5))


class TestMockImplementations:
    def test_mock_loadable_isinstance(self) -> None:
        assert isinstance(_MockLoadable(), Loadable)

    def test_mock_steppable_isinstance(self) -> None:
        assert isinstance(_MockSteppable(), Steppable)

    def test_mock_queryable_isinstance(self) -> None:
        assert isinstance(_MockQueryable(), Queryable)

    def test_mock_recordable_isinstance(self) -> None:
        assert isinstance(_MockRecordable(), Recordable)

    def test_mock_parameter_gradients_isinstance(self) -> None:
        assert isinstance(_MockParameterGradients(), SupportsParameterGradients)

    def test_plain_object_not_loadable(self) -> None:
        assert not isinstance(object(), Loadable)

    def test_plain_object_not_steppable(self) -> None:
        assert not isinstance(object(), Steppable)

    def test_plain_object_not_queryable(self) -> None:
        assert not isinstance(object(), Queryable)

    def test_plain_object_not_recordable(self) -> None:
        assert not isinstance(object(), Recordable)

    def test_plain_object_not_parameter_gradients(self) -> None:
        assert not isinstance(object(), SupportsParameterGradients)
