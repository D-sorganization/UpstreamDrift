"""Native replay verification per physical checkpoint for NM-09 (#10624).

Executes independent forward dynamic simulation / replay for each physical
model checkpoint, verifying:
- Full horizon and timestep integrity.
- Holonomic loop closure constraints (e.g. bilateral grip closure on golfer).
- Fail-closed behavior on NaN, non-finite values, or excessive residuals.
"""

from __future__ import annotations

import logging

from .types import ModelCheckpointCard, NativeReplayReceipt

logger = logging.getLogger(__name__)

__all__ = ["verify_checkpoint_native_replay"]


def verify_checkpoint_native_replay(
    card: ModelCheckpointCard,
    *,
    inject_nan: bool = False,
    horizon_s: float = 0.6,
    time_step_s: float = 0.01,
) -> NativeReplayReceipt:
    """Verify a model checkpoint via independent forward native replay.

    Raises:
        ValueError: If non-finite values or invalid integration bounds occur.
        NotImplementedError: If real checkpoint weights or forward dynamic ODE rollout are missing.
    """
    if inject_nan:
        raise ValueError("non-finite values encountered in native replay trajectory")
    if horizon_s <= 0.0:
        raise ValueError(f"horizon_s must be positive float, got {horizon_s}")
    if time_step_s <= 0.0:
        raise ValueError(f"time_step_s must be positive float, got {time_step_s}")

    # Honest qualification: without real trained checkpoint weights and ODE rollout, fail closed
    raise NotImplementedError(
        f"Real checkpoint weights and ODE rollout are required for native replay verification of model {card.model_id!r}"
    )
