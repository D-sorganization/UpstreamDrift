"""Per-model checkpoint matrix and qualification package for NM-09 (#10624)."""

from __future__ import annotations

from .adapters import (
    ConstrainedUpperBodyGeneratorAdapter,
    FullBodyEngineGeneratorAdapter,
    GeneratorAdapterBase,
    KinematicReconstructionGeneratorAdapter,
    PlanarDoublePendulumGeneratorAdapter,
    PlanarTriplePendulumGeneratorAdapter,
    get_generator_adapter_for_model,
    is_runtime_available_for_model,
)
from .builder import (
    MATRIX_SCHEMA,
    NeuralCheckpointMatrix,
    build_checkpoint_matrix,
)
from .replay import verify_checkpoint_native_replay
from .types import (
    CHECKPOINT_SCHEMA,
    BenefitResult,
    ModelCheckpointCard,
    ModelCheckpointStatus,
    NativeReplayReceipt,
    ThreeSeedEvidence,
    assert_model_checkpoint_compatible,
)

__all__ = [
    "CHECKPOINT_SCHEMA",
    "MATRIX_SCHEMA",
    "BenefitResult",
    "ConstrainedUpperBodyGeneratorAdapter",
    "FullBodyEngineGeneratorAdapter",
    "GeneratorAdapterBase",
    "KinematicReconstructionGeneratorAdapter",
    "ModelCheckpointCard",
    "ModelCheckpointStatus",
    "NativeReplayReceipt",
    "NeuralCheckpointMatrix",
    "PlanarDoublePendulumGeneratorAdapter",
    "PlanarTriplePendulumGeneratorAdapter",
    "ThreeSeedEvidence",
    "assert_model_checkpoint_compatible",
    "build_checkpoint_matrix",
    "get_generator_adapter_for_model",
    "is_runtime_available_for_model",
    "verify_checkpoint_native_replay",
]
