"""Vectorization contract tests for JaxSim parameter-gradient helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.jaxsim import parameter_gradients

pytestmark = pytest.mark.unit


class _FakeJnp:
    @staticmethod
    def asarray(value: Any) -> np.ndarray:
        return np.asarray(value, dtype=np.float64)


class _FakeJax:
    def __init__(self) -> None:
        self.jacfwd_calls = 0
        self.vmap_calls = 0

    def jacfwd(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        del function, args, kwargs
        self.jacfwd_calls += 1

        def jacobian(*call_args: Any) -> np.ndarray:
            del call_args
            return np.ones((2, len(parameter_gradients.PARAMETER_NAMES)))

        return jacobian

    def jacrev(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        del function, args, kwargs
        raise AssertionError("reverse-mode transform should not be built")

    def vmap(self, function: Any) -> Any:
        self.vmap_calls += 1

        def mapped(q_batch: np.ndarray, v_batch: np.ndarray) -> np.ndarray:
            return np.stack(
                [
                    function(q_row, v_row)
                    for q_row, v_row in zip(q_batch, v_batch, strict=True)
                ]
            )

        return mapped


def test_trajectory_sensitivity_constructs_one_forward_transform(
    monkeypatch: Any,
) -> None:
    fake_jax = _FakeJax()
    monkeypatch.setattr(
        parameter_gradients,
        "_require_jax",
        lambda: (_FakeJnp, fake_jax),
    )

    sensitivity = (
        parameter_gradients.evaluate_ztcf_parameter_sensitivity_along_trajectory(
            parameter_gradients.DEFAULT_PARAMETER_VECTOR,
            np.array([[0.1, -0.2], [0.2, -0.15], [0.3, -0.1]], dtype=np.float64),
            np.array([[0.3, -0.4], [0.2, -0.1], [0.1, 0.05]], dtype=np.float64),
            mode="forward",
        )
    )

    assert fake_jax.jacfwd_calls == 1
    assert fake_jax.vmap_calls == 1
    assert sensitivity.shape == (3, 2, len(parameter_gradients.PARAMETER_NAMES))
