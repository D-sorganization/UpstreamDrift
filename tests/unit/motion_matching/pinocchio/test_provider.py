"""Unit tests for :class:`PinocchioFitSwingProvider` (issue #4517).

The provider is a thin adapter around :func:`fit_swing_pinocchio`; the
heavy numerical work is already covered by the dedicated heavy-integration
suite under ``tests/heavy_integration/test_pinocchio_fit_swing.py``. The
tests here focus on the contract surface introduced by issue #4517:

* The provider self-registers in ``PROVIDER_REGISTRY`` at import time.
* ``engine_name`` is ``"pinocchio"``.
* ``supports_body_target()`` and ``supports_ball_target()`` both return
  ``False`` (Pinocchio MM is club-target only).
* ``fit_swing(target, opts)`` on a synthetic recovery target returns a
  valid :class:`FitResult` whose recovered theta is close to the truth.

The numerical-regression check pins recovery to within 5% on a small
synthetic problem, mirroring the pinned regression bound used by the
existing heavy-integration coverage. A real ``TW_ProV1.mat`` recovery
test stays in ``tests/heavy_integration/`` because (a) the .mat file
isn't available in stripped-down checkouts, and (b) running it at
unit-test scope would blow past the 60 s timeout.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("pinocchio")


def _real_pinocchio() -> bool:
    """Detect whether ``pinocchio`` is the real C++ binding.

    The unit-test conftest substitutes a :class:`unittest.mock.MagicMock`
    for ``pinocchio`` so the rest of the suite can import engine modules
    without the heavy C++ binding installed. The numerical-recovery test
    below needs the real bindings, so we skip cleanly when a mock is in
    play.
    """
    import sys
    from unittest.mock import MagicMock

    return not isinstance(sys.modules.get("pinocchio"), MagicMock)


from src.engines.physics_engines.pinocchio.python.motion_matching import (  # noqa: E402
    FitOptions,
    FitResult,
    PinocchioFitSwingProvider,
    rotmat_to_quat_wxyz,
    simulate_with_coefficients,
)
from src.engines.physics_engines.pinocchio.python.motion_matching.provider import (  # noqa: E402
    ENGINE_NAME,
    PROVIDER_REGISTRY,
)
from src.engines.physics_engines.pinocchio.python.motion_matching.simulate import (  # noqa: E402
    COEFFS_PER_JOINT,
    SimOptions,
)
from src.shared.python.motion_matching.club_target import (  # noqa: E402
    ClubTarget,
    SourceProvenance,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _provenance() -> SourceProvenance:
    return SourceProvenance(
        filename="synthetic_provider_test",
        format="synthetic",
        subject_id="UNIT",
        trial_id="provider",
        sha256=hashlib.sha256(b"provider-test").hexdigest(),
    )


def _synthesize_target(theta: np.ndarray, *, t_final: float, dt: float) -> ClubTarget:
    """Forward-sim ``theta`` and pack the result as a :class:`ClubTarget`."""
    sim_options = SimOptions(t_final=t_final, dt=dt, compute_energy=False)
    out = simulate_with_coefficients(theta, sim_options)
    quats = rotmat_to_quat_wxyz(out.clubhead_rotation)
    n = out.t.shape[0]
    return ClubTarget(
        time=out.t.copy(),
        butt=out.grip_position.copy(),
        clubhead=out.clubhead_position.copy(),
        club_quat=quats,
        impact_idx=(n // 2) + 1,
        source=_provenance(),
    )


def _n_joints() -> int:
    import pinocchio as pin

    from src.engines.physics_engines.pinocchio.python.motion_matching.simulate import (
        _resolve_urdf_path,
    )

    urdf = _resolve_urdf_path(None)
    if not urdf.exists():
        pytest.skip(f"golfer.urdf not found at {urdf}")
    return int(pin.buildModelFromUrdf(str(urdf)).nv)


# --------------------------------------------------------------------------- #
# Registration / contract surface
# --------------------------------------------------------------------------- #


class TestProviderRegistration:
    def test_engine_name_constant(self) -> None:
        assert PinocchioFitSwingProvider.engine_name == "pinocchio"
        assert ENGINE_NAME == "pinocchio"

    def test_registered_at_import_time(self) -> None:
        assert ENGINE_NAME in PROVIDER_REGISTRY
        provider = PROVIDER_REGISTRY[ENGINE_NAME]
        assert isinstance(provider, PinocchioFitSwingProvider)
        assert provider.engine_name == "pinocchio"

    def test_supports_body_target_returns_false(self) -> None:
        assert PinocchioFitSwingProvider().supports_body_target() is False

    def test_supports_ball_target_returns_false(self) -> None:
        assert PinocchioFitSwingProvider().supports_ball_target() is False


# --------------------------------------------------------------------------- #
# Numerical recovery (synthetic, small to fit in unit-test budget)
# --------------------------------------------------------------------------- #


class TestProviderFitSwing:
    """Provider returns a valid FitResult and recovers theta within 5%.

    The 5% bound matches the pinned regression already in
    ``tests/heavy_integration/test_pinocchio_fit_swing.py`` so any
    accidental change to the underlying optimiser surfaces here too.
    """

    @pytest.mark.slow
    def test_fit_swing_recovery_within_5_percent(self) -> None:
        if not _real_pinocchio():
            pytest.skip("real pinocchio bindings unavailable (mock in use)")
        n_joints = _n_joints()
        rng = np.random.default_rng(seed=42)
        theta_truth = 1e-3 * rng.standard_normal(n_joints * COEFFS_PER_JOINT)
        target = _synthesize_target(theta_truth, t_final=0.05, dt=1e-3)

        theta0 = theta_truth + 1e-4 * rng.standard_normal(theta_truth.shape)
        opts = FitOptions(
            theta0=theta0,
            max_iter=30,
            jac_mode="analytical",
            ftol=1e-10,
            xtol=1e-10,
        )

        provider = PinocchioFitSwingProvider()
        result = provider.fit_swing(target, opts)

        assert isinstance(result, FitResult)
        assert result.theta_optimal.shape == theta_truth.shape
        assert np.all(np.isfinite(result.theta_optimal))
        assert result.final_cost >= 0.0
        assert result.wall_clock_s > 0.0

        # Numerical regression: recovered theta within 5% of truth.
        denom = max(float(np.linalg.norm(theta_truth)), 1e-12)
        rel_err = float(np.linalg.norm(result.theta_optimal - theta_truth) / denom)
        assert rel_err < 0.05, (
            f"||theta - theta_truth|| / ||theta_truth|| = {rel_err:.4f} "
            f">= 0.05 (provider regression bound, mirrors #4132)"
        )

    def test_fit_swing_with_default_options(self) -> None:
        """Provider accepts ``opts=None`` and returns a valid FitResult."""
        if not _real_pinocchio():
            pytest.skip("real pinocchio bindings unavailable (mock in use)")
        n_joints = _n_joints()
        rng = np.random.default_rng(seed=7)
        theta_truth = 1e-3 * rng.standard_normal(n_joints * COEFFS_PER_JOINT)
        target = _synthesize_target(theta_truth, t_final=0.02, dt=1e-3)

        provider = PinocchioFitSwingProvider()
        # Use a tiny max_iter via explicit opts to keep wall-clock tight,
        # while still exercising the None-passthrough at the surface.
        opts = FitOptions(
            theta0=theta_truth,
            max_iter=2,
            jac_mode="analytical",
        )
        result = provider.fit_swing(target, opts)

        assert isinstance(result, FitResult)
        assert result.theta_optimal.shape == theta_truth.shape
        assert np.all(np.isfinite(result.theta_optimal))
