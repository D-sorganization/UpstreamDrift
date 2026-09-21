"""Tests for the Pendulum motion-matching provider."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.engines.physics_engines.pendulum.python.motion_matching.provider import (
    PendulumFitSwingProvider,
)
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

pytestmark = pytest.mark.unit


@pytest.fixture
def dummy_club_target() -> ClubTarget:
    """Return a minimal realistic ClubTarget for testing."""
    times = np.linspace(0.0, 0.2, 5)
    angles = np.linspace(0.1, 0.5, 5)
    # 2D arc projected into 3D
    butt = np.column_stack([0.6 * np.sin(angles), np.zeros(5), -0.6 * np.cos(angles)])
    head = np.column_stack([1.5 * np.sin(angles), np.zeros(5), -1.5 * np.cos(angles)])
    return ClubTarget(
        time=times,
        butt=butt,
        clubhead=head,
        club_quat=np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)),
        impact_idx=4,
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
    """fit_swing returns a well-formed CanonicalFitResult from the SLSQP fit."""
    provider = PendulumFitSwingProvider()
    opts = FitOptions(maxiter=10)

    result = provider.fit_swing(dummy_club_target, opts)

    assert result.solver_status in {"success", "failure"}
    assert result.final_cost >= 0.0
    assert math.isfinite(result.final_cost)
    assert result.final_rmse_m == pytest.approx(math.sqrt(result.final_cost))
    assert result.method == "scipy SLSQP"
    assert isinstance(result.git_commit, str) and result.git_commit
    assert result.target_hash != "dummy"
    assert len(result.target_hash) == 64


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
