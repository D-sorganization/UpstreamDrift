"""How a replay is driven: the policy enum shared by engines and null-space code.

Lives under ``shared`` so the null-space replay can name a policy without a
``shared -> engines`` import; :mod:`src.engines.feedback_comparison` re-exports
it unchanged, so ``feedback_comparison.ReplayPolicy is ReplayPolicy``.
"""

from __future__ import annotations

import enum

__all__ = ["ReplayPolicy"]


class ReplayPolicy(str, enum.Enum):
    """Whether a replay applies inputs on a fixed clock or closes a state loop."""

    INDEPENDENT_TIME_ONLY = "independent_time_only"
    STATE_FEEDBACK = "state_feedback"
