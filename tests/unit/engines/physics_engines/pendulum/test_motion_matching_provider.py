"""Tests for the Pendulum motion-matching provider."""

from __future__ import annotations

import pytest
from types import SimpleNamespace
import numpy as np

pytestmark = pytest.mark.unit

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


def test_planar_floor_prevents_futile_fitting() -> None:
    """A spatial target whose irreducible swing plane RMSE exceeds tolerance rejects early before optimization."""
    provider = PendulumFitSwingProvider()

    # Create target with significant non-planar out-of-plane variation
    n = 20
    times = np.linspace(0.0, 0.2, n)
    butt = np.zeros((n, 3))
    butt[:, 0] = np.linspace(0.0, 0.5, n)
    butt[:, 1] = 0.0
    butt[:, 2] = 0.0

    clubhead = np.zeros((n, 3))
    clubhead[:, 0] = 0.0
    clubhead[:, 1] = np.linspace(0.5, 1.2, n)
    # Inject large z-axis (out-of-plane) excursion that cannot be fit by a 2D plane
    clubhead[:, 2] = 0.15 * np.sin(np.pi * np.linspace(0.0, 1.0, n))

    target = ClubTarget(
        time=times,
        butt=butt,
        clubhead=clubhead,
        club_quat=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        impact_idx=n // 2,
        source=SourceProvenance("nonplanar.c3d", "c3d", "test", "test", "hash123"),
    )

    # Ceiling is 10 mm (0.010 m), but irreducible normal residual is ~40-50 mm
    opts = FitOptions(maxiter=100, max_marker_rmse_m=0.010)

    result = provider.fit_swing(target, opts)

    assert result.solver_status == "failure"
    assert result.iterations == 0
    assert result.n_evaluations == 0
    assert "Planar floor" in result.message or "irreducible" in result.message.lower()


def _identity_plane(residual_rmse: float):
    """Identity-plane calibration whose recorded residual is fabricated."""
    from src.shared.python.motion_matching.projection_2d import (
        CalibratedSwingPlane,
        GeometricProjectionResidual,
    )

    return CalibratedSwingPlane(
        origin=np.zeros(3),
        basis=np.eye(3),
        transform_world_to_plane=np.eye(4),
        transform_plane_to_world=np.eye(4),
        inclination_deg=0.0,
        azimuth_deg=0.0,
        residual=GeometricProjectionResidual(
            rmse=residual_rmse,
            max_deviation=residual_rmse * 1.5,
            signed_deviations=np.zeros(4),
        ),
    )


def _planar_target(nonplanar_z_amplitude: float) -> ClubTarget:
    n = 20
    times = np.linspace(0.0, 0.2, n)
    butt = np.zeros((n, 3))
    butt[:, 0] = np.linspace(0.0, 0.5, n)
    clubhead = np.zeros((n, 3))
    clubhead[:, 0] = 0.0
    clubhead[:, 1] = np.linspace(0.5, 1.2, n)
    clubhead[:, 2] = nonplanar_z_amplitude * np.sin(np.pi * np.linspace(0.0, 1.0, n))
    return ClubTarget(
        time=times,
        butt=butt,
        clubhead=clubhead,
        club_quat=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        impact_idx=n // 2,
        source=SourceProvenance("plane.c3d", "c3d", "test", "test", "hash123"),
    )


def test_planar_floor_gates_calibrated_plane_on_current_target() -> None:
    """P1: a clean calibration residual must not pass a far-off target."""
    provider = PendulumFitSwingProvider()
    plane = _identity_plane(residual_rmse=0.0005)  # "perfect" calibration
    target = _planar_target(nonplanar_z_amplitude=0.15)  # ~45 mm off-plane target
    engine_opts = SimpleNamespace(calibrated_plane=plane)
    opts = FitOptions(maxiter=100, engine_options=engine_opts, max_marker_rmse_m=0.010)

    result = provider.fit_swing(target, opts)

    assert result.solver_status == "failure"
    assert result.iterations == 0
    assert "planar floor" in result.message.lower()


def test_planar_floor_does_not_reject_target_from_stale_calibration_residual() -> None:
    """P1: a stale/large calibration residual must not reject a planar target."""
    provider = PendulumFitSwingProvider()
    plane = _identity_plane(residual_rmse=0.500)  # stale, noisy calibration
    target = _planar_target(nonplanar_z_amplitude=0.0)  # perfectly planar target
    engine_opts = SimpleNamespace(calibrated_plane=plane)
    opts = FitOptions(maxiter=100, engine_options=engine_opts, max_marker_rmse_m=0.010)

    result = provider.fit_swing(target, opts)

    assert "planar floor" not in result.message.lower()


def test_planar_floor_nan_ceiling_via_engine_options_fails_closed() -> None:
    """A non-finite ceiling must not silently disable the planar-floor gate."""
    provider = PendulumFitSwingProvider()
    target = _planar_target(nonplanar_z_amplitude=0.0)
    engine_opts = SimpleNamespace(max_marker_rmse_m=float("nan"))
    opts = FitOptions(maxiter=100, engine_options=engine_opts)

    result = provider.fit_swing(target, opts)

    assert result.solver_status == "failure"
    assert "max_marker_rmse_m must be finite and positive" in result.message
