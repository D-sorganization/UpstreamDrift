"""Tests for the Pendulum motion-matching provider."""

from __future__ import annotations

import pytest
import numpy as np

from src.shared.python.motion_matching.club_target import ClubTarget, SourceProvenance
from src.shared.python.motion_matching.provider import (
    FitOptions,
    MultiSourceTarget,
    resolve_club_target,
)
from src.shared.python.motion_matching.provider_registry import (
    clear_registry,
    get_provider,
    register_provider,
)
from src.engines.physics_engines.pendulum.python.motion_matching.provider import (
    PendulumFitSwingProvider,
)


@pytest.fixture
def dummy_club_target() -> ClubTarget:
    """Return a minimal ClubTarget for testing."""
    return ClubTarget(
        time=np.array([0.0, 0.1]),
        butt=np.zeros((2, 3)),
        clubhead=np.zeros((2, 3)),
        club_quat=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        impact_idx=1,
        source=SourceProvenance("test.c3d", "c3d", "test", "test", "dummy"),
    )


def test_provider_registers() -> None:
    """Test that the pendulum provider can be successfully registered."""
    clear_registry()
    provider = PendulumFitSwingProvider()
    register_provider(provider)

    retrieved = get_provider("pendulum")
    assert retrieved is provider


def test_fit_swing_returns_baseline(dummy_club_target: ClubTarget) -> None:
    """fit_swing returns a well-formed CanonicalFitResult from the SLSQP fit.

    The pendulum provider drives ``scipy.optimize.minimize(SLSQP)`` over the
    polynomial-torque coefficients; it does not produce a zero-cost analytic
    baseline. Assert the real solver contract: a finite non-negative cost, a
    matching RMSE, the SLSQP method tag, and a populated git-commit stamp
    (#6935 / #6939 wired the shared provenance probe).
    """
    import math

    provider = PendulumFitSwingProvider()
    opts = FitOptions(maxiter=10)

    result = provider.fit_swing(dummy_club_target, opts)

    assert result.solver_status in {"success", "failure"}
    assert result.final_cost >= 0.0
    assert math.isfinite(result.final_cost)
    assert result.final_rmse_m == pytest.approx(math.sqrt(result.final_cost))
    assert result.method == "scipy SLSQP"
    assert isinstance(result.git_commit, str) and result.git_commit


def test_extract_club_from_multisource(dummy_club_target: ClubTarget) -> None:
    """Test extracting the club target from a MultiSourceTarget."""
    provider = PendulumFitSwingProvider()
    multi = MultiSourceTarget(club=dummy_club_target, body=None)

    extracted = provider._extract_club(multi)
    assert extracted is dummy_club_target


def test_extract_club_rejects_invalid() -> None:
    """Test that the shared unwrap raises errors on bad input (#6935)."""
    with pytest.raises(TypeError, match="MultiSourceTarget"):
        resolve_club_target("not a target")  # type: ignore

    with pytest.raises(ValueError, match="at least one of \\(club, body\\) set"):
        MultiSourceTarget(club=None, body=None)


def test_provider_capabilities() -> None:
    """Test the static capability flags of the provider."""
    provider = PendulumFitSwingProvider()

    assert not provider.supports_body_target()
    assert not provider.supports_ball_target()
    assert provider.engine_version() == "1.0.0"
    assert provider.engine_name == "pendulum"
