"""Full TreeModel for tree/forest penalty calculation.

Extends the Phase-1 stub in ``course.conditions.TreeModel`` with a proper
recovery-distribution model: when penalization is high the ball is behind
trees and must be punched out sideways rather than played normally.

Phase 2 (#6271).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.shared.python.contracts import require

if TYPE_CHECKING:  # pragma: no cover
    from src.shared.python.sg_optimizer.course.features import StateFeatures

# Threshold above which the shot is fully forced into punch-out (matches the
# Phase-1 stub ``TreeModel.is_forced_punch_out`` in course.conditions).
_FULL_PUNCH_OUT_THRESHOLD = 0.85


class TreeModel:
    """Tree/forest penalty model with recovery-distribution support.

    Parameters
    ----------
    penalization : float
        Tree density / occlusion severity ∈ [0, 1].
        0 = no trees; 1 = fully surrounded with no direct escape route.

    This class provides:
    - ``forced_punch_out_probability`` — probability that a normal swing is
      impossible and a sideways punch-out is mandatory.
    - ``apply_to_transition`` — modifies a transition probability dict so
      that some weight is re-routed from the intended landing zone to a
      punch-out landing zone that is immediately to the side.
    """

    def __init__(self, penalization: float) -> None:
        require(
            0.0 <= penalization <= 1.0,
            "penalization must lie in [0, 1]",
            penalization,
        )
        self.penalization = penalization

    # --- Public API -------------------------------------------------------

    def forced_punch_out_probability(
        self, state_features: StateFeatures | None = None
    ) -> float:
        """Return the probability that the ball must be punched out.

        When ``penalization > 0.85`` (full jail) this is **always** 1.0.
        Below that threshold the probability scales with penalization and
        with how close the ball is to trees (proxied via
        ``state_features.lie`` when provided).

        Parameters
        ----------
        state_features :
            Optional state features; when supplied and lie is not "trees"
            the probability is zero (ball is not actually in trees).
        """
        if state_features is not None and state_features.lie != "trees":
            return 0.0

        if self.penalization > _FULL_PUNCH_OUT_THRESHOLD:
            return 1.0

        # Logistic-style ramp: slow growth until ~0.7, then steep.
        p = self.penalization
        # P(punch_out | trees) = p³ / (p³ + (1-p)³) is a smooth S-curve.
        p3 = p**3
        q3 = (1.0 - p) ** 3
        denom = p3 + q3
        if denom < 1e-12:
            return 0.5
        return p3 / denom

    def apply_to_transition(
        self,
        transition_probs: dict[str, float],
        state_features: StateFeatures | None = None,
    ) -> dict[str, float]:
        """Modify a transition-probability dict to account for tree penalty.

        A fraction ``forced_punch_out_probability(state_features)`` of the
        total probability mass is re-allocated from every outcome to a
        ``"punch_out"`` outcome representing lateral escape.  The remaining
        probability mass is rescaled proportionally so the dict still sums
        to 1.0.

        Parameters
        ----------
        transition_probs :
            Dict mapping outcome label → probability.  Must be non-empty and
            values must sum to approximately 1.0 (within 1 %).
        state_features :
            Passed through to ``forced_punch_out_probability``.

        Returns
        -------
        dict[str, float]
            Modified probability dict (new dict, input unchanged).
            Always contains a ``"punch_out"`` key (possibly 0.0).
        """
        require(len(transition_probs) > 0, "transition_probs must be non-empty")
        total = sum(transition_probs.values())
        require(
            abs(total - 1.0) < 0.01,
            f"transition_probs must sum to ~1.0, got {total:.4f}",
            total,
        )

        punch_p = self.forced_punch_out_probability(state_features)
        keep = 1.0 - punch_p

        result: dict[str, float] = {k: v * keep for k, v in transition_probs.items()}
        result["punch_out"] = result.get("punch_out", 0.0) + punch_p
        return result

    # --- Convenience properties -------------------------------------------

    @property
    def is_forced_punch_out(self) -> bool:
        """True when penalization exceeds the full-jail threshold."""
        return self.penalization > _FULL_PUNCH_OUT_THRESHOLD

    def distance_multiplier(self) -> float:
        """Effective advance-distance scaling when in trees."""
        return max(0.05, 1.0 - 0.9 * self.penalization)

    def dispersion_multiplier(self) -> float:
        """Dispersion scaling factor when in trees."""
        return 1.0 + 0.6 * self.penalization

    def __repr__(self) -> str:
        return f"TreeModel(penalization={self.penalization!r})"
