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
    PinocchioFitSwingProvider,
    rotmat_to_quat_wxyz,
    simulate_with_coefficients,
)
from src.engines.physics_engines.pinocchio.python.motion_matching.provider import (  # noqa: E402
    ENGINE_NAME,
)
from src.engines.physics_engines.pinocchio.python.motion_matching.simulate import (  # noqa: E402
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
        from src.shared.python.motion_matching.provider import get_provider

        provider = get_provider("pinocchio")
        assert isinstance(provider, PinocchioFitSwingProvider)
        assert provider.engine_name == "pinocchio"

    def test_supports_body_target_returns_false(self) -> None:
        assert PinocchioFitSwingProvider().supports_body_target() is False

    def test_supports_ball_target_returns_false(self) -> None:
        assert PinocchioFitSwingProvider().supports_ball_target() is False


# --------------------------------------------------------------------------- #
# Numerical recovery (synthetic, small to fit in unit-test budget)
# --------------------------------------------------------------------------- #
