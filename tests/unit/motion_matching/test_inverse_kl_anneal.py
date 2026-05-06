"""Unit tests for the KL annealing schedule (issue #033 / GH #4002)."""

from __future__ import annotations

import pytest
from src.shared.python.motion_matching.inverse._kl import (
    default_warmup_epochs,
    linear_kl_beta,
)


@pytest.mark.unit
def test_kl_annealing_monotonic_in_first_warmup_epochs() -> None:
    warmup = 10
    betas = [
        linear_kl_beta(e, total_epochs=50, warmup_epochs=warmup, max_beta=1.0)
        for e in range(warmup + 5)
    ]
    # Strictly non-decreasing; reaches the plateau at exactly warmup_epochs.
    for prev, curr in zip(betas[:-1], betas[1:], strict=True):
        assert curr >= prev
    assert betas[0] == pytest.approx(0.1)  # (0+1)/10
    assert betas[warmup - 1] == pytest.approx(1.0)
    assert betas[warmup] == pytest.approx(1.0)
    assert betas[-1] == pytest.approx(1.0)


@pytest.mark.unit
def test_zero_warmup_returns_max_beta_immediately() -> None:
    assert linear_kl_beta(0, total_epochs=10, warmup_epochs=0, max_beta=1.0) == 1.0
    assert linear_kl_beta(7, total_epochs=10, warmup_epochs=0, max_beta=0.5) == 0.5


@pytest.mark.unit
def test_default_warmup_is_twenty_percent_of_total() -> None:
    assert default_warmup_epochs(100) == 20
    assert default_warmup_epochs(50) == 10
    assert default_warmup_epochs(1) == 1  # never zero


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"epoch": -1, "total_epochs": 10, "warmup_epochs": 2},
        {"epoch": 0, "total_epochs": 0, "warmup_epochs": 2},
        {"epoch": 0, "total_epochs": 10, "warmup_epochs": -1},
        {"epoch": 0, "total_epochs": 10, "warmup_epochs": 2, "max_beta": -0.1},
    ],
)
def test_invalid_inputs_raise(kwargs: dict[str, float]) -> None:
    from src.shared.python.core.contracts.exceptions import PreconditionError

    with pytest.raises(PreconditionError):
        linear_kl_beta(**kwargs)
