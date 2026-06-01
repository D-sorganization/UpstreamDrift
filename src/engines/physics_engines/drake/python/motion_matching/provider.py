"""Drake adapter for the canonical ``fit_swing`` provider surface.

Issue #4516. Wires the existing :func:`fit_swing_drake` driver — the
SLSQP / scipy.optimize gradient-free path that already returns a
:class:`CanonicalFitResult` — into the cross-engine provider registry
defined in :mod:`src.shared.python.motion_matching.provider_registry`.

The provider does **not** rewrite the optimiser. Its job is purely
input/output adaptation:

* Accept the still-evolving ``MultiSourceTarget`` shape from #4509 by
  duck-typing on ``target.club``; fall back to a bare :class:`ClubTarget`
  / ``ClubBallTarget`` when those types are passed directly.
* Forward to :func:`fit_swing_drake`, which already returns
  :class:`CanonicalFitResult`.

Body- and ball-target cost terms are explicitly **not** supported in
this initial pass (see #4520). The provider advertises that via
``supports_body_target`` / ``supports_ball_target`` returning ``False``.
"""

from __future__ import annotations

import logging
from typing import Any

from src.shared.python.motion_matching.fit_result import CanonicalFitResult
from src.shared.python.motion_matching.provenance import engine_package_version
from src.shared.python.motion_matching.provider import (
    publish_leaderboard_row,
    resolve_club_target,
)

from .fit_swing import FitOptions, fit_swing_drake

logger = logging.getLogger(__name__)

__all__ = ["DrakeFitSwingProvider"]


class DrakeFitSwingProvider:
    """Drake-side ``fit_swing`` provider.

    See module docstring for the full contract. The optimiser is the
    canonical SLSQP driver from :mod:`.fit_swing`; this class is a thin
    I/O wrapper that lets the cross-engine matcher discover Drake
    through the shared registry.
    """

    engine_name: str = "drake"

    def fit_swing(
        self,
        target: Any,
        opts: FitOptions | None = None,
    ) -> CanonicalFitResult:
        """Run the canonical Drake fit on ``target.club``.

        Args:
            target: A :class:`ClubTarget`, ``ClubBallTarget``, or
                ``MultiSourceTarget``. Only the club component is used
                in this initial pass.
            opts: Optional :class:`FitOptions`. ``None`` means
                ``FitOptions()`` (canonical 23-joint Drake humanoid,
                SLSQP, 200 iters, ftol=1e-6).

        Returns:
            :class:`CanonicalFitResult` from
            :func:`fit_swing_drake`.

        Raises:
            TypeError: If ``target`` lacks a usable ``.club`` /
                :class:`ClubTarget` shape.
        """
        club_target = resolve_club_target(target)
        result = fit_swing_drake(club_target, opts)
        # Issue #4713 / #6935: opt-in CI publication via the shared helper.
        publish_leaderboard_row(self.engine_name, result, self.engine_version())
        return result

    def supports_body_target(self) -> bool:
        """Drake body-target cost terms are out-of-scope for #4516."""
        return False

    def supports_ball_target(self) -> bool:
        """Ball-boundary-condition cost terms land separately (see #4488)."""
        return False

    def engine_version(self) -> str:
        """Return the installed ``pydrake`` version, or ``"unknown"``.

        Stamps the resolved value into leaderboard rows so two runs
        against different Drake wheels are distinguishable (issue #4705).
        Delegates to the shared :func:`engine_package_version` cascade
        (issue #6939): ``pydrake.__version__`` first, then the ``drake`` /
        ``pydrake`` distributions, then ``"unknown"``.
        """
        try:
            import pydrake  # type: ignore[import-not-found]
        except ImportError:
            return "unknown"
        return engine_package_version(pydrake, "drake", "pydrake")
