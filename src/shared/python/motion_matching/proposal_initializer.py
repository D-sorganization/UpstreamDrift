"""ProposalInitializer protocol, stub implementation, and proposal validation for ML-WS-8 (#13785)."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from src.engines.physics_engines.mujoco.python.motion_matching.mjx_knot_optimiser import (
        MjxPackage,
    )

__all__ = [
    "ProposalInitializer",
    "StubProposalInitializer",
    "validate_proposal",
]


@runtime_checkable
class ProposalInitializer(Protocol):
    """Protocol for neural/warm-start knot proposal initializers.

    ``propose(package)`` returns ``(delta, source_id, sha256)``.
    """

    def propose(self, package: MjxPackage) -> tuple[np.ndarray, str, str]:
        """Generate a knot proposal for the given MJX package."""
        ...


class StubProposalInitializer:
    """Deterministic stub initializer for testing warm-start proposals.

    If ``delta`` is provided, it returns that array. Otherwise, it generates
    a small deterministic constant perturbation matching the package's expected shape.
    """

    def __init__(
        self,
        delta: np.ndarray | None = None,
        source_id: str = "stub_proposal",
    ) -> None:
        self._delta = None if delta is None else np.asarray(delta, dtype=np.float32)
        self.source_id = source_id

    def propose(self, package: MjxPackage) -> tuple[np.ndarray, str, str]:
        if self._delta is not None:
            delta = self._delta
        else:
            knots = len(package.q_track)
            actuated = len(package.meta["coordinate_order"]) - 1
            delta = np.zeros((knots, actuated), dtype=np.float32)

        sha = hashlib.sha256(delta.tobytes()).hexdigest()
        return delta, self.source_id, sha


def validate_proposal(
    delta: Any,
    expected_shape: tuple[int, int],
    bounds: tuple[float, float] | None = None,
) -> tuple[bool, str | None]:
    """Validate proposal array against expected shape, finiteness, and bounds (DbC).

    Returns ``(True, None)`` if valid, or ``(False, reason)`` if invalid.
    """
    if delta is None:
        return False, "proposal delta is None"

    arr = np.asarray(delta)
    if arr.shape != expected_shape:
        return (
            False,
            f"proposal shape mismatched: expected {expected_shape}, got {arr.shape}",
        )

    if not np.all(np.isfinite(arr)):
        return False, "proposal contains non-finite values (NaN or Inf)"

    if bounds is not None:
        low, high = bounds
        if np.any(arr < low) or np.any(arr > high):
            return False, f"proposal exceeds bounds [{low}, {high}]"

    return True, None
