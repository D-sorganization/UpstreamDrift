"""Finite-difference minimum-frame contract + accuracy (issue #7146).

The estimators silently returned all-zero qdot/qddot for short trajectories,
turning inverse dynamics into statics. These tests pin the explicit contract
(``ValueError`` for too-few frames unless overrides are supplied) at the
``pose_interchange`` reference-adapter entry point and verify the shared helper,
plus a property-based check that finite differencing reproduces the analytic
derivative of a known polynomial for trajectories with enough samples.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.shared.python.engine_core.finite_difference import (
    MIN_FRAMES_FOR_QDDOT,
    MIN_FRAMES_FOR_QDOT,
    require_enough_frames_for_finite_diff,
)
from src.shared.python.pose_interchange.adapters.pinocchio_reference import (
    PinocchioReferenceAdapter,
)

pytestmark = pytest.mark.unit


class _IdentityBackend:
    def rnea(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        return a.copy()

    def aba(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return tau.copy()

    def fk(self, q, frame):  # pragma: no cover
        return np.eye(4)

    def jacobian(self, q, frame):  # pragma: no cover
        return np.zeros((6, q.shape[0] - 1))


def _q_row(scale: float) -> np.ndarray:
    """Free-flyer q with base-x translation set to ``scale`` (identity quat)."""

    return np.array([scale, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


# --- Shared helper contract -------------------------------------------------


def test_helper_rejects_single_frame_qdot() -> None:
    with pytest.raises(ValueError, match="qdot requires at least"):
        require_enough_frames_for_finite_diff(
            n_frames=1, need_qdot=True, need_qddot=False
        )


def test_helper_rejects_two_frame_qddot() -> None:
    with pytest.raises(ValueError, match="qddot requires at least"):
        require_enough_frames_for_finite_diff(
            n_frames=2, need_qdot=True, need_qddot=True
        )


def test_helper_allows_overrides_for_short_trajectories() -> None:
    # Caller supplies derivatives -> no estimation needed -> no error.
    require_enough_frames_for_finite_diff(n_frames=1, need_qdot=False, need_qddot=False)


def test_helper_constants_are_consistent() -> None:
    assert MIN_FRAMES_FOR_QDOT == 2
    assert MIN_FRAMES_FOR_QDDOT == 3


# --- Adapter entry-point contract ------------------------------------------


def test_adapter_single_frame_trajectory_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "upstream_pinocchio_id", raising=False)
    adapter = PinocchioReferenceAdapter(_IdentityBackend())
    q = np.vstack([_q_row(0.0)])
    with pytest.raises(ValueError, match="requires at least"):
        adapter.inverse_dynamics_trajectory(q, np.array([0.0]))


def test_adapter_two_frame_trajectory_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "upstream_pinocchio_id", raising=False)
    adapter = PinocchioReferenceAdapter(_IdentityBackend())
    q = np.vstack([_q_row(0.0), _q_row(0.1)])
    with pytest.raises(ValueError, match="qddot requires at least"):
        adapter.inverse_dynamics_trajectory(q, np.array([0.0, 0.1]))


def test_adapter_short_trajectory_with_overrides_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "upstream_pinocchio_id", raising=False)
    adapter = PinocchioReferenceAdapter(_IdentityBackend())
    q = np.vstack([_q_row(0.0)])
    qdot = np.zeros((1, 7), dtype=np.float64)
    qddot = np.zeros((1, 7), dtype=np.float64)
    result = adapter.inverse_dynamics_trajectory(
        q, np.array([0.0]), qdot=qdot, qddot=qddot
    )
    assert result.qdot.shape == (1, 7)


# --- Property-based accuracy on a known polynomial -------------------------


@settings(max_examples=30, deadline=None)
@given(
    accel=st.floats(min_value=-5.0, max_value=5.0),
    vel0=st.floats(min_value=-5.0, max_value=5.0),
)
def test_finite_diff_matches_analytic_derivative(accel: float, vel0: float) -> None:
    monkeypatch_modules = sys.modules.pop("upstream_pinocchio_id", None)
    try:
        adapter = PinocchioReferenceAdapter(_IdentityBackend())
        # base-x(t) = 0.5*a*t^2 + v0*t (pure translation; naive finite diff is
        # exact for a quadratic). Base linear vx is qdot column 0.
        times = np.linspace(0.0, 1.0, 11)
        q = np.vstack([_q_row(0.5 * accel * t**2 + vel0 * t) for t in times])
        result = adapter.inverse_dynamics_trajectory(q, times)
        # Interior base vx should approximate a*t + v0; ax ~ a.
        interior = slice(1, -1)
        analytic_qdot = accel * times[interior] + vel0
        np.testing.assert_allclose(result.qdot[interior, 0], analytic_qdot, atol=1e-6)
        np.testing.assert_allclose(
            result.qddot[interior, 0], np.full(times[interior].shape, accel), atol=1e-6
        )
    finally:
        if monkeypatch_modules is not None:
            sys.modules["upstream_pinocchio_id"] = monkeypatch_modules
