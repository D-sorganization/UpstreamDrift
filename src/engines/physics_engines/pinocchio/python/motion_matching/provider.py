"""Pinocchio :class:`PinocchioFitSwingProvider` (issue #4517).

Wraps the existing engine-local optimiser in the canonical fit-swing
provider contract introduced by issue #4514. The provider is a thin
adapter: all numerical work is delegated to
:func:`fit_swing_pinocchio` (LM + analytical Jacobians, issue #4132)
and :func:`simulate_with_coefficients` (RK4 + ABA, issue #4118), which
already consume the canonical :class:`ClubTarget` produced by
:func:`load_robneal_target` (issue #4127).

The provider auto-registers in :data:`PROVIDER_REGISTRY` at import time
so callers can do::

    from src.engines.physics_engines.pinocchio.python.motion_matching import (
        PinocchioFitSwingProvider,
    )
    provider = PinocchioFitSwingProvider()
    result = provider.fit_swing(target, opts)

When :mod:`pinocchio` (or its compiled bindings) is unavailable the
provider class still imports cleanly; the actual ``fit_swing`` call
will surface :class:`ImportError` from the underlying simulator. This
matches the behaviour of every other engine adapter and keeps the
provider registry usable in environments where one engine is missing.

The provider intentionally does NOT support body or ball targets --
the Pinocchio motion-matching pipeline is club-target only, mirroring
what ``club_target_adapter.load_robneal_target`` produces.
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import TYPE_CHECKING, Final

from .fit_swing import FitOptions, FitResult, fit_swing_pinocchio

if TYPE_CHECKING:  # pragma: no cover -- type-only import
    from src.shared.python.motion_matching.club_target import ClubTarget

logger = logging.getLogger(__name__)

__all__ = [
    "PROVIDER_REGISTRY",
    "PinocchioFitSwingProvider",
]


# Engine-name string used as the registry key. Kept as a module-level
# constant so the test suite (and downstream cross-engine drivers) can
# look it up without instantiating the provider.
ENGINE_NAME: Final[str] = "pinocchio"


class PinocchioFitSwingProvider:
    """Canonical motion-matching provider for the Pinocchio engine.

    Implements the protocol-shape called out in issue #4517:

    * ``engine_name`` -- string identifier, ``"pinocchio"``.
    * ``fit_swing(target, opts)`` -- delegates to
      :func:`fit_swing_pinocchio`.
    * ``supports_body_target() -> bool`` -- ``False``.
    * ``supports_ball_target() -> bool`` -- ``False``.

    The class itself is intentionally tiny; every numerical behaviour
    is preserved by deferring to the underlying optimiser rather than
    re-implementing it. This keeps the killer-feature analytical-Jacobian
    path the single source of truth.
    """

    engine_name: Final[str] = ENGINE_NAME

    def fit_swing(
        self,
        target: ClubTarget,
        opts: FitOptions | None = None,
    ) -> FitResult:
        """Fit polynomial-torque coefficients for ``target``.

        Args:
            target: Validated :class:`ClubTarget` (e.g. produced by
                :func:`load_robneal_target`).
            opts: Optional :class:`FitOptions`. ``None`` -> defaults.

        Returns:
            Canonical :class:`FitResult` from :func:`fit_swing_pinocchio`.

        Raises:
            ValueError: If ``target`` shapes are inconsistent.
            ImportError: If the ``pinocchio`` bindings are unavailable.
        """
        result = fit_swing_pinocchio(target, opts)
        # Issue #4713: opt-in CI publication of the cross-engine leaderboard.
        from src.shared.python.motion_matching.leaderboard import maybe_append_row

        maybe_append_row(
            ENGINE_NAME,
            result,
            self.engine_version(),
            target_id=getattr(result, "trial_id", None),
        )
        return result

    def supports_body_target(self) -> bool:
        """Return ``False`` -- Pinocchio MM is club-target only."""
        return False

    def supports_ball_target(self) -> bool:
        """Return ``False`` -- Pinocchio MM is club-target only."""
        return False

    def engine_version(self) -> str:
        """Return the installed ``pinocchio`` version, or ``"unknown"``.

        Stamped into leaderboard rows (issue #4705) so two runs against
        different Pinocchio wheels stay distinguishable. Returns
        ``"unknown"`` when the bindings are not installed; the provider
        class itself imports cleanly even without them.
        """
        try:
            import pinocchio  # type: ignore[import-not-found]
        except ImportError:
            return "unknown"
        if not isinstance(pinocchio, ModuleType):
            return "unknown"
        version = getattr(pinocchio, "__version__", None)
        if isinstance(version, str) and version:
            return version
        try:
            from importlib.metadata import PackageNotFoundError
            from importlib.metadata import version as _v
        except ImportError:  # pragma: no cover -- stdlib >=3.8
            return "unknown"
        try:
            return _v("pin")
        except PackageNotFoundError:
            try:
                return _v("pinocchio")
            except PackageNotFoundError:
                return "unknown"


# --------------------------------------------------------------------------- #
# Auto-registration
# --------------------------------------------------------------------------- #

# Engine-local provider registry. When the canonical cross-engine
# registry from #4514 lands, this module should re-export from there
# instead. Until then, the dict keyed by ``engine_name`` is sufficient
# for the symmetric pattern requested by #4517 ("provider registered at
# import time"). Kept module-private (re-exported below) so that import
# of this module is the registration step.
PROVIDER_REGISTRY: dict[str, PinocchioFitSwingProvider] = {}


def _register() -> None:
    """Idempotently register the Pinocchio provider in the local registry."""
    if ENGINE_NAME in PROVIDER_REGISTRY:
        return
    PROVIDER_REGISTRY[ENGINE_NAME] = PinocchioFitSwingProvider()
    logger.debug(
        "PinocchioFitSwingProvider registered under engine_name=%r", ENGINE_NAME
    )


_register()
