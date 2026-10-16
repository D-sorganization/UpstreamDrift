"""Characterization tests for BatchSwingOptimizer construction (#12844)."""

from __future__ import annotations

import pytest

from src.shared.python.optimization.batch_swing_optimizer import (
    BatchSwingObjectiveWeights,
    BatchSwingOptimizer,
)

_ATTRS = (
    "method",
    "n_candidates",
    "n_iterations",
    "elite_fraction",
    "temperature",
    "tau_max",
    "seed",
    "max_batch",
)


@pytest.mark.unit
def test_defaults_are_pinned() -> None:
    opt = BatchSwingOptimizer()
    assert opt.backend_name == "ode"
    assert [getattr(opt, a) for a in _ATTRS] == [
        "cem",
        64,
        10,
        0.2,
        1.0,
        100.0,
        0,
        None,
    ]
    assert opt.weights == BatchSwingObjectiveWeights()
    assert opt.params is not None


@pytest.mark.unit
def test_keyword_overrides_are_stored() -> None:
    w = BatchSwingObjectiveWeights(effort=0.5)
    opt = BatchSwingOptimizer(
        "mjx",
        method="mppi",
        n_candidates=8,
        n_iterations=2,
        elite_fraction=0.5,
        temperature=2.0,
        tau_max=10.0,
        seed=9,
        weights=w,
        max_batch=4,
    )
    assert opt.backend_name == "mjx"
    assert [getattr(opt, a) for a in _ATTRS] == [
        "mppi",
        8,
        2,
        0.5,
        2.0,
        10.0,
        9,
        4,
    ]
    assert opt.weights is w


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_iterations": 0}, "n_iterations must be positive"),
        ({"temperature": 0.0}, "temperature must be positive"),
        ({"elite_fraction": 1.5}, "elite_fraction"),
        ({"n_candidates": 1}, "n_candidates must be at least 2"),
    ],
)
def test_invalid_hyperparameters_raise_value_error(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        BatchSwingOptimizer(**kwargs)


@pytest.mark.unit
def test_unknown_keyword_raises_type_error() -> None:
    with pytest.raises(TypeError):
        BatchSwingOptimizer(not_an_option=1)  # type: ignore[call-arg]
