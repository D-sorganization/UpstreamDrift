"""``OpenSimFitSwingProvider`` -- canonical motion-matching adapter for OpenSim.

Per #4708 (and the cross-engine parity spec #4513), every engine ships a
provider that adapts the canonical ``MultiSourceTarget`` / ``FitOptions``
inputs to the engine's existing fit driver. This module is a thin
adapter on top of :func:`fit_swing_opensim` (issue #4128); it does NOT
touch the optimiser, the polynomial-torque controller, or the forward
simulator.

The provider auto-registers at import time so::

    import src.engines.physics_engines.opensim.python.motion_matching  # noqa
    from src.shared.python.motion_matching.provider_registry import (
        get_provider,
    )
    provider = get_provider("opensim")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.shared.python.motion_matching.club_ball_target import ClubBallTarget
from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.provider import (
    FitOptions,
    MultiSourceTarget,
    publish_leaderboard_row,
    resolve_club_target,
)

from .fit_swing import FitOptions as OpenSimFitOptions
from .fit_swing import fit_swing_opensim

if TYPE_CHECKING:  # pragma: no cover - type-only import
    from src.shared.python.motion_matching.fit_result import (
        CanonicalFitResult,
    )

logger = logging.getLogger(__name__)

__all__ = ["OpenSimFitSwingProvider"]


class OpenSimFitSwingProvider:
    """Canonical-API adapter wrapping :func:`fit_swing_opensim`.

    The provider:

    1. accepts a :class:`MultiSourceTarget` (reading ``target.club``), a
       bare :class:`ClubTarget` (back-compat), or a :class:`ClubBallTarget`
       (the club portion is consumed; the ball boundary is ignored until
       OpenSim grows ball-target support);
    2. unwraps the canonical :class:`FitOptions` to the engine's native
       :class:`OpenSimFitOptions`, preserving any user-supplied
       ``engine_options`` and projecting ``opts.maxiter`` /
       ``opts.rng_seed`` onto a fresh native options object when none is
       supplied;
    3. delegates to :func:`fit_swing_opensim` and returns the engine's
       :class:`CanonicalFitResult` unchanged.
    """

    engine_name: str = "opensim"

    def fit_swing(
        self,
        target: MultiSourceTarget | ClubTarget | ClubBallTarget,
        opts: FitOptions,
    ) -> CanonicalFitResult:
        """Adapt canonical inputs and call :func:`fit_swing_opensim`.

        Args:
            target: One of:
                * a :class:`MultiSourceTarget` (whose ``.club`` slot MUST
                  be a :class:`ClubTarget`),
                * a bare :class:`ClubTarget`, or
                * a :class:`ClubBallTarget` (the wrapped ``.club`` is
                  used; the ball boundary is ignored).
            opts: Canonical :class:`FitOptions`. ``opts.engine_options``,
                if supplied, MUST be an :class:`OpenSimFitOptions`; when
                absent, defaults are used and ``opts.maxiter`` /
                ``opts.rng_seed`` are projected onto the native options.

        Returns:
            :class:`CanonicalFitResult` directly from
            :func:`fit_swing_opensim`.

        Raises:
            ValueError: when the canonical target has no ``.club`` slot
                or carries a non-:class:`ClubTarget` payload.
            TypeError: when ``target`` is not one of the supported types,
                or when ``opts.engine_options`` is supplied but is not an
                :class:`OpenSimFitOptions`.
        """
        club = resolve_club_target(target)
        native = self._build_native_options(opts)
        result = fit_swing_opensim(club, native)
        # Issue #4713 / #6935: opt-in CI publication via the shared helper.
        version = (
            self.engine_version()
            if hasattr(self, "engine_version")
            else getattr(result, "engine_version", "unknown")
        )
        publish_leaderboard_row(self.engine_name, result, version)
        return result

    def supports_body_target(self) -> bool:
        """OpenSim's swing fitter consumes only the club trajectory."""
        return False

    def supports_ball_target(self) -> bool:
        """OpenSim's swing fitter does not consume ball targets yet."""
        return False

    # --- Internal helpers ----------------------------------------------

    @staticmethod
    def _extract_club(
        target: MultiSourceTarget | ClubTarget | ClubBallTarget,
    ) -> ClubTarget:
        """Return the :class:`ClubTarget` payload from ``target``.

        Thin delegate to the shared :func:`resolve_club_target` (issue
        #6935) so the unwrap behaviour is identical across every engine.
        Retained as a public static method for back-compat with callers
        and tests that reference it directly.
        """
        return resolve_club_target(target)

    @staticmethod
    def _build_native_options(opts: FitOptions) -> OpenSimFitOptions:
        """Convert canonical :class:`FitOptions` -> :class:`OpenSimFitOptions`.

        Honors ``opts.engine_options`` when it carries an
        :class:`OpenSimFitOptions`; otherwise builds a default options
        object and projects ``maxiter`` / ``rng_seed`` onto it.
        """
        if opts.engine_options is None:
            from dataclasses import replace

            base = OpenSimFitOptions()
            return replace(
                base,
                max_iter=int(opts.maxiter),
                rng_seed=int(opts.rng_seed),
            )
        if isinstance(opts.engine_options, OpenSimFitOptions):
            return opts.engine_options
        raise TypeError(
            f"opts.engine_options must be OpenSimFitOptions or None, "
            f"got {type(opts.engine_options).__name__}"
        )
