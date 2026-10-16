"""Oracle tests for the mixing diagnostics (NSX-70, #12588).

Oracles with closed forms: an AR(1) process with coefficient ``phi`` has
autocorrelation ``phi ** k`` and integrated autocorrelation time
``(1 + phi) / (1 - phi)``; independent normals have R-hat near one and an
effective sample size near the draw count; chains with different means do not
mix and must be flagged.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.mixing_diagnostics import (
    autocorrelation,
    effective_sample_size,
    split_rhat,
)

pytestmark = pytest.mark.unit


def _ar1(phi: float, n: int, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((m, n))
    out[:, 0] = rng.standard_normal(m)
    noise = rng.standard_normal((m, n)) * np.sqrt(1.0 - phi**2)
    for i in range(1, n):
        out[:, i] = phi * out[:, i - 1] + noise[:, i]
    return out


def test_autocorrelation_of_ar1_is_geometric() -> None:
    x = _ar1(0.8, 200_000, 1, seed=1)[0]
    rho = autocorrelation(x, max_lag=5)
    assert rho[0] == pytest.approx(1.0, abs=1e-12)
    for k in range(1, 6):
        assert rho[k] == pytest.approx(0.8**k, abs=0.01)


def test_ess_of_independent_draws_is_near_the_draw_count() -> None:
    chains = np.random.default_rng(2).standard_normal((4, 5000))
    ess = effective_sample_size(chains)
    assert 0.85 * chains.size <= ess <= 1.15 * chains.size


def test_ess_of_ar1_matches_the_integrated_autocorrelation_time() -> None:
    phi, n, m = 0.9, 20_000, 4
    ess = effective_sample_size(_ar1(phi, n, m, seed=3))
    expected = n * m * (1.0 - phi) / (1.0 + phi)
    assert ess == pytest.approx(expected, rel=0.2)


def test_split_rhat_is_one_for_iid_and_large_for_unmixed_chains() -> None:
    iid = np.random.default_rng(4).standard_normal((4, 4000))
    assert split_rhat(iid) == pytest.approx(1.0, abs=0.01)
    stuck = iid + np.array([[0.0], [3.0], [6.0], [9.0]])
    assert split_rhat(stuck) > 1.5


def test_split_rhat_flags_a_drifting_chain_that_plain_rhat_would_miss() -> None:
    drift = np.linspace(0.0, 5.0, 2000)
    chains = np.random.default_rng(5).standard_normal((2, 2000)) + drift
    assert split_rhat(chains) > 1.1


def test_preconditions_raise() -> None:
    with pytest.raises(ValueError, match="2-D"):
        effective_sample_size(np.zeros(10))
    with pytest.raises(ValueError, match="at least 8"):
        split_rhat(np.zeros((2, 4)))
    with pytest.raises(ValueError, match="finite"):
        effective_sample_size(np.full((2, 20), np.nan))
    with pytest.raises(ValueError, match="max_lag"):
        autocorrelation(np.zeros(10), max_lag=10)


def test_constant_chains_have_no_defined_rhat() -> None:
    assert np.isnan(split_rhat(np.ones((2, 20))))
