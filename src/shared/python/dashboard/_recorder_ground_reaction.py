"""Ground-reaction breakdown recording (GCV-5, #11711).

Collects the engine's per-step ``GroundReactionBreakdown`` (when the engine
implements ``get_ground_reaction_breakdown``) into a stacked
:class:`GroundReactionSeries`, read by
``ground_reaction_service.resolve_ground_reaction_series`` the same way it
reads an engine's own ``get_ground_reaction_series()``. Engines without the
method are unaffected; a ``None`` breakdown at any sample makes the whole
series unavailable (never partial, never zero).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from src.shared.python.biomechanics.ground_reaction import (
        GroundReactionBreakdown,
        GroundReactionSeries,
    )

logger = get_logger(__name__)


class _GroundReactionMixin:
    engine: Any
    _ground_reaction_times: list[float]
    _ground_reaction_breakdowns: list[GroundReactionBreakdown]
    _ground_reaction_unavailable: bool

    def _reset_ground_reaction(self) -> None:
        """(Re)start collection; call from ``_reset_buffers``/``reset``."""
        self._ground_reaction_times = []
        self._ground_reaction_breakdowns = []
        self._ground_reaction_unavailable = False

    def _record_ground_reaction_breakdown(self, t: float) -> None:
        """Append the engine's breakdown at time ``t``, if it provides one.

        A ``None`` breakdown (the engine cannot report at this instant) marks
        the series unavailable for the rest of the recording: a gap would be
        indistinguishable from zero reaction.
        """
        if self._ground_reaction_unavailable:
            return
        provider = getattr(self.engine, "get_ground_reaction_breakdown", None)
        if not callable(provider):
            return
        try:
            breakdown = provider()
        except (ValueError, RuntimeError) as e:
            logger.warning("Failed to compute ground reaction breakdown: %s", e)
            self._ground_reaction_unavailable = True
            return
        if breakdown is None:
            self._ground_reaction_unavailable = True
            return
        self._ground_reaction_times.append(float(t))
        self._ground_reaction_breakdowns.append(breakdown)

    def get_ground_reaction_series(self) -> GroundReactionSeries | None:
        """The recorded ground-reaction series, or ``None`` when unavailable.

        ``None`` when the engine never provided a breakdown, reported one as
        ``None`` at some sample, or fewer than one sample was recorded.
        """
        from src.shared.python.biomechanics.ground_reaction import (
            GroundReactionSeries,
        )

        if self._ground_reaction_unavailable or not self._ground_reaction_breakdowns:
            return None
        try:
            return GroundReactionSeries.from_breakdowns(
                self._ground_reaction_times, self._ground_reaction_breakdowns
            )
        except ValueError as e:
            logger.warning("Ground reaction series unavailable: %s", e)
            return None
