from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.engine_core.checkpoint import StateCheckpoint

if TYPE_CHECKING:
    from src.engines.physics_engines.putting_green.python._sim_core import (
        PuttingGreenSimulator,
    )


def get_checkpoint(sim: PuttingGreenSimulator) -> StateCheckpoint:
    """Save current state to checkpoint."""
    return StateCheckpoint.create(
        engine_type="putting_green",
        engine_state={
            "spin": sim._ball_state.spin.tolist(),
        },
        q=sim._ball_state.position,
        v=sim._ball_state.velocity,
        timestamp=sim._time,
    )


def restore_checkpoint(sim: PuttingGreenSimulator, checkpoint: StateCheckpoint) -> None:
    """Restore state from checkpoint."""
    if checkpoint is None:
        raise ValueError("checkpoint must be provided")
    sim._ball_state.position = checkpoint.get_q()
    sim._ball_state.velocity = checkpoint.get_v()
    sim._time = checkpoint.timestamp
    if "spin" in checkpoint.engine_state:
        sim._ball_state.spin = np.array(checkpoint.engine_state["spin"])
