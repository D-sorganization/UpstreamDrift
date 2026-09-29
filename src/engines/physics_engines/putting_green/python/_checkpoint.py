"""Checkpoint save/restore for the putting-green engine.

Roll-model provenance (ADR-0045 F1, issue #9343):
    Checkpoints are a *persisted* format — archives written before this change
    still exist on disk (``CheckpointManager.save_to_disk``) — so they are the
    one putting-green payload that is archive-tolerant rather than fail-closed.
    The engine-owned payload is versioned by ``schema_version`` inside
    ``engine_state``: version 2 carries ``roll_model`` and is read fail-closed,
    while an unversioned version-1 archive still restores but reads back as
    :class:`CheckpointProvenance` with ``roll_model=None`` and
    ``is_archive=True``. A legacy archive is never silently relabelled with the
    current model name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
    ROLL_MODEL_FIELD,
    require_roll_model,
)
from src.shared.python.engine_core.checkpoint import StateCheckpoint

if TYPE_CHECKING:
    from src.engines.physics_engines.putting_green.python._sim_core import (
        PuttingGreenSimulator,
    )

#: Engine type recorded in every putting-green checkpoint.
PUTTING_CHECKPOINT_ENGINE_TYPE = "putting_green"

#: Version of the putting-green ``engine_state`` payload written today.
#: Version 1 (implicit, pre-ADR-0045) carried only ``spin``; version 2 adds
#: ``roll_model``.
PUTTING_CHECKPOINT_SCHEMA_VERSION = 2

#: Version assumed for an archive that predates ``schema_version``.
PUTTING_CHECKPOINT_ARCHIVE_VERSION = 1

#: Key holding the payload version inside ``engine_state``.
SCHEMA_VERSION_FIELD = "schema_version"


@dataclass(frozen=True)
class CheckpointProvenance:
    """Roll-model provenance read back from a putting-green checkpoint.

    Attributes:
        roll_model: Model name recorded in the checkpoint, or ``None`` for a
            pre-ADR-0045 archive that never recorded one.
        schema_version: Version of the ``engine_state`` payload.
        is_archive: True when the payload predates roll-model provenance and
            therefore cannot be compared against a named result.
    """

    roll_model: str | None
    schema_version: int
    is_archive: bool


def get_checkpoint(sim: PuttingGreenSimulator) -> StateCheckpoint:
    """Save current state to a checkpoint that names its roll model.

    Postcondition: ``engine_state`` carries ``schema_version`` and
    ``roll_model`` (ADR-0045 F1).
    """
    return StateCheckpoint.create(
        engine_type=PUTTING_CHECKPOINT_ENGINE_TYPE,
        engine_state={
            SCHEMA_VERSION_FIELD: PUTTING_CHECKPOINT_SCHEMA_VERSION,
            ROLL_MODEL_FIELD: sim.roll_model,
            "spin": sim._ball_state.spin.tolist(),
        },
        q=sim._ball_state.position,
        v=sim._ball_state.velocity,
        timestamp=sim._time,
    )


def read_checkpoint_provenance(checkpoint: StateCheckpoint) -> CheckpointProvenance:
    """Classify a putting-green checkpoint's roll-model provenance.

    Archive-tolerant by design: a version-1 payload (written before ADR-0045)
    is reported as an archive with ``roll_model=None`` instead of being
    relabelled with the model in use today. Version 2 and later are read
    fail-closed — a current-version payload that omits the model is a bug, not
    an archive.

    Args:
        checkpoint: Checkpoint produced by this engine.

    Returns:
        The checkpoint's provenance classification.

    Raises:
        ValueError: If ``checkpoint`` is ``None`` or was written by another
            engine.
        RollModelProvenanceError: If a current-version payload omits or
            misnames its roll model.
    """
    if checkpoint is None:
        raise ValueError("checkpoint must be provided")
    if checkpoint.engine_type != PUTTING_CHECKPOINT_ENGINE_TYPE:
        raise ValueError(
            "checkpoint must come from the putting_green engine, got "
            f"{checkpoint.engine_type!r}"
        )

    engine_state: dict[str, Any] = dict(checkpoint.engine_state or {})
    version = int(
        engine_state.get(SCHEMA_VERSION_FIELD, PUTTING_CHECKPOINT_ARCHIVE_VERSION)
    )
    if version < PUTTING_CHECKPOINT_SCHEMA_VERSION:
        return CheckpointProvenance(
            roll_model=None,
            schema_version=version,
            is_archive=True,
        )

    roll_model = require_roll_model(
        engine_state,
        source=f"putting_green checkpoint {checkpoint.id!r} (v{version})",
    )
    return CheckpointProvenance(
        roll_model=roll_model,
        schema_version=version,
        is_archive=False,
    )


def restore_checkpoint(sim: PuttingGreenSimulator, checkpoint: StateCheckpoint) -> None:
    """Restore state from a checkpoint, including pre-ADR-0045 archives.

    Restoring is deliberately archive-tolerant: refusing an old checkpoint
    would delete the ability to read existing archives, which ADR-0045
    forbids. Callers that need the provenance ask
    :func:`read_checkpoint_provenance` and handle ``roll_model=None``.
    """
    if checkpoint is None:
        raise ValueError("checkpoint must be provided")
    sim._ball_state.position = checkpoint.get_q()
    sim._ball_state.velocity = checkpoint.get_v()
    sim._time = checkpoint.timestamp
    if "spin" in checkpoint.engine_state:
        sim._ball_state.spin = np.array(checkpoint.engine_state["spin"])


__all__ = [
    "PUTTING_CHECKPOINT_ARCHIVE_VERSION",
    "PUTTING_CHECKPOINT_ENGINE_TYPE",
    "PUTTING_CHECKPOINT_SCHEMA_VERSION",
    "SCHEMA_VERSION_FIELD",
    "CheckpointProvenance",
    "get_checkpoint",
    "read_checkpoint_provenance",
    "restore_checkpoint",
]
