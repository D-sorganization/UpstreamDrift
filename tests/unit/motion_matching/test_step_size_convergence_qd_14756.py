"""#14756 P7: step-size convergence must not ignore qd or accept non-finite data."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.cross_engine_replay import (
    compute_step_size_convergence,
)

pytestmark = pytest.mark.unit

_N, _DOF = 5, 3


def _run(q_ref=None, qd_ref=None, **kwargs):
    t = np.linspace(0.0, 1.0, _N)
    q = np.zeros((_N, _DOF))
    qd = np.zeros((_N, _DOF))
    return compute_step_size_convergence(
        t,
        q,
        qd,
        q if q_ref is None else q_ref,
        qd if qd_ref is None else qd_ref,
        1e-3,
        5e-4,
        **kwargs,
    )


def test_identical_runs_converge() -> None:
    assert _run().is_converged is True


def test_qd_only_divergence_passes_by_default_but_is_reported() -> None:
    """Default behaviour is preserved: no qd threshold unless one is declared."""
    qd_ref = np.full((_N, _DOF), 50.0)
    res = _run(qd_ref=qd_ref)
    assert res.max_qd_difference == pytest.approx(50.0)
    assert res.is_converged is True


def test_qd_only_divergence_fails_with_declared_qd_tolerance() -> None:
    qd_ref = np.zeros((_N, _DOF))
    qd_ref[2, 1] = 50.0
    res = _run(qd_ref=qd_ref, tolerance_rad_s=1.0)
    assert res.is_converged is False


def test_qd_within_declared_tolerance_converges() -> None:
    qd_ref = np.full((_N, _DOF), 0.5)
    assert _run(qd_ref=qd_ref, tolerance_rad_s=1.0).is_converged is True


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_qd_never_converges(bad: float) -> None:
    qd_ref = np.zeros((_N, _DOF))
    qd_ref[1, 0] = bad
    assert _run(qd_ref=qd_ref).is_converged is False


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_q_never_converges(bad: float) -> None:
    q_ref = np.zeros((_N, _DOF))
    q_ref[3, 2] = bad
    assert _run(q_ref=q_ref).is_converged is False


def test_non_positive_qd_tolerance_rejected() -> None:
    with pytest.raises(ValueError, match="tolerance_rad_s"):
        _run(tolerance_rad_s=0.0)
