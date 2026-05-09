"""``MujocoFitSwingProvider`` -- canonical motion-matching adapter for MuJoCo.

Per #4519 (and the cross-engine parity spec #4513), every engine ships a
provider that adapts the canonical ``MultiSourceTarget`` / ``FitOptions``
inputs to the engine's existing fit driver. This module is a thin
adapter on top of :func:`fit_swing_mujoco`; it does NOT touch the
optimiser, the polynomial-torque driver, or the einsum-optimised cost
gradient.

The provider auto-registers at import time so::

    import src.engines.physics_engines.mujoco.python.motion_matching  # noqa
    from src.shared.python.motion_matching.provider import get_provider
    provider = get_provider("mujoco")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.provider import (
    FitOptions,
    MultiSourceTarget,
    register_provider,
)

from .fit_swing import FitOptions as MujocoFitOptions
from .fit_swing import fit_swing_mujoco

if TYPE_CHECKING:
    from src.shared.python.motion_matching.fit_result import (
        CanonicalFitResult,
    )

logger = logging.getLogger(__name__)

__all__ = ["MujocoFitSwingProvider"]


class MujocoFitSwingProvider:
    """Canonical-API adapter wrapping :func:`fit_swing_mujoco`.

    The provider:

    1. accepts either a :class:`MultiSourceTarget` (reading
       ``target.club``) or a raw :class:`ClubTarget` (back-compat);
    2. unwraps the canonical :class:`FitOptions` to the engine's native
       :class:`MujocoFitOptions`, preserving any user-supplied
       ``engine_options``;
    3. delegates to :func:`fit_swing_mujoco` and returns the engine's
       :class:`CanonicalFitResult` unchanged.
    """

    engine_name: str = "mujoco"

    def fit_swing(
        self,
        target: MultiSourceTarget | ClubTarget,
        opts: FitOptions,
    ) -> CanonicalFitResult:
        """Adapt canonical inputs and call :func:`fit_swing_mujoco`.

        Args:
            target: Either a :class:`MultiSourceTarget` (whose ``.club``
                slot MUST be a :class:`ClubTarget`) or a bare
                :class:`ClubTarget`.
            opts: Canonical :class:`FitOptions`. ``opts.engine_options``,
                if supplied, MUST be a :class:`MujocoFitOptions`; when
                absent, defaults are used and ``opts.maxiter`` /
                ``opts.rng_seed`` are projected onto the native options.

        Returns:
            :class:`CanonicalFitResult` directly from
            :func:`fit_swing_mujoco`.

        Raises:
            ValueError: when the canonical target has no ``.club`` slot
                or carries a non-:class:`ClubTarget` payload.
            TypeError: when ``opts.engine_options`` is supplied but is not
                a :class:`MujocoFitOptions`.
        """
        club = self._extract_club(target)
        native = self._build_native_options(opts)
        return fit_swing_mujoco(club, native)

    def supports_body_target(self) -> bool:
        """MuJoCo's swing fitter consumes only the club trajectory."""
        return False

    def supports_ball_target(self) -> bool:
        """MuJoCo's swing fitter does not consume ball targets."""
        return False

    def engine_version(self) -> str:
        """Return the installed ``mujoco`` version, or ``"unknown"``.

        Lets leaderboard rows distinguish runs across MuJoCo wheel
        upgrades (issue #4705). Returns ``"unknown"`` when the
        ``mujoco`` package is not installed so the provider stays
        constructible in MuJoCo-less environments.
        """
        try:
            import mujoco  # type: ignore[import-not-found]
        except ImportError:
            return "unknown"
        version = getattr(mujoco, "__version__", None)
        if isinstance(version, str) and version:
            return version
        try:
            from importlib.metadata import PackageNotFoundError
            from importlib.metadata import version as _v
        except ImportError:  # pragma: no cover -- stdlib >=3.8
            return "unknown"
        try:
            return _v("mujoco")
        except PackageNotFoundError:
            return "unknown"

    # --- Internal helpers ----------------------------------------------

    @staticmethod
    def _extract_club(target: MultiSourceTarget | ClubTarget) -> ClubTarget:
        """Return the :class:`ClubTarget` payload from ``target``.

        Accepts both wrapped and unwrapped inputs to keep call sites
        portable across the canonical-API rollout.
        """
        if isinstance(target, ClubTarget):
            return target
        if isinstance(target, MultiSourceTarget):
            if target.club is None:
                raise ValueError(
                    "MujocoFitSwingProvider requires target.club to be set; "
                    "got MultiSourceTarget with club=None"
                )
            if not isinstance(target.club, ClubTarget):
                raise ValueError(
                    f"target.club must be a ClubTarget, "
                    f"got {type(target.club).__name__}"
                )
            return target.club
        raise TypeError(
            f"target must be MultiSourceTarget or ClubTarget, "
            f"got {type(target).__name__}"
        )

    @staticmethod
    def _build_native_options(opts: FitOptions) -> MujocoFitOptions:
        """Convert canonical :class:`FitOptions` -> :class:`MujocoFitOptions`.

        Honors ``opts.engine_options`` when it carries a
        :class:`MujocoFitOptions`; otherwise builds a default options
        graph and projects ``maxiter`` / ``rng_seed`` onto it.
        """
        if opts.engine_options is None:
            from dataclasses import replace

            base = MujocoFitOptions()
            return replace(
                base,
                minimizer=replace(base.minimizer, maxiter=int(opts.maxiter)),
                rng_seed=int(opts.rng_seed),
            )
        if isinstance(opts.engine_options, MujocoFitOptions):
            return opts.engine_options
        raise TypeError(
            f"opts.engine_options must be MujocoFitOptions or None, "
            f"got {type(opts.engine_options).__name__}"
        )


# Auto-register at import time so a single
# ``import src.engines.physics_engines.mujoco.python.motion_matching``
# is sufficient to populate the canonical registry.
register_provider(MujocoFitSwingProvider())
