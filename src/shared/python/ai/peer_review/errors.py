"""Error hierarchy for the ``ai.peer_review`` package (Tools #2738)."""

from __future__ import annotations


class PeerReviewError(Exception):
    """Base class for all peer-review errors."""


class NoReviewersError(PeerReviewError):
    """Raised when the registry has zero reviewers registered."""


class InsufficientPanelError(PeerReviewError):
    """Raised when the panel built from the registry is smaller than the
    configured minimum panel size for a coordinator."""


class ReviewerTimeoutError(PeerReviewError, TimeoutError):
    """Raised when the coordinator's deadline elapses before all panel
    members have returned a verdict."""


__all__ = [
    "InsufficientPanelError",
    "NoReviewersError",
    "PeerReviewError",
    "ReviewerTimeoutError",
]
