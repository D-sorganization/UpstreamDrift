"""Training-job configuration — what to train, how, and with what budget.

A :class:`TrainingConfig` is an immutable, fully-validated description
of *one* training run. It is the single object that crosses every layer
boundary: GUI builds it, the scheduler queues it, the worker process
deserializes it and hands it to a framework adapter.

The schema is versioned (:data:`CURRENT_SCHEMA_VERSION`) so we can
evolve the format without breaking persisted jobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any
from collections.abc import Mapping

from .errors import TrainingConfigError
from .resources import ResourceRequest

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "TrainingConfig",
    "TrainingFramework",
]


CURRENT_SCHEMA_VERSION = 1
"""Wire-format version for :class:`TrainingConfig`. Bump on breaking changes."""


class TrainingFramework(Enum):
    """ML/RL framework that owns the training inner loop.

    v1 ships with PyTorch (supervised) and Gymnasium (RL envs). Future
    frameworks (TensorFlow, JAX, stable-baselines3, RLlib) plug in by
    adding an enum member here plus a :class:`TrainingJobRunner`
    adapter — they do not require changes to :class:`TrainingConfig`.
    """

    PYTORCH = "pytorch"
    GYMNASIUM = "gymnasium"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Immutable description of a training job.

    Attributes:
        framework: Which adapter should execute this config.
        entry_point: ``"module.path:callable"`` or path to a Python
            script the worker will invoke. Validated as non-empty here;
            the worker (PR2) is responsible for resolving it.
        output_dir: Directory where artifacts (checkpoints, metrics,
            logs) will be written. The directory does not need to exist
            at construction time — the worker creates it.
        schema_version: Wire-format version. Defaults to
            :data:`CURRENT_SCHEMA_VERSION`. Older configs are migrated
            on load (PR2 territory).
        hyperparameters: Free-form mapping passed verbatim to the
            framework adapter. Frozen via :class:`MappingProxyType`.
        dataset_id: Identifier of a registered dataset, or ``None`` for
            self-generating workloads (e.g. RL envs).
        resources: Resource ceiling. Defaults to a single-CPU job with
            1024 MiB RAM.
        max_epochs: Optional hard cap on epochs (supervised). Mutually
            non-exclusive with ``max_steps``.
        max_steps: Optional hard cap on steps (RL or per-iteration).
        seed: Optional RNG seed for reproducibility.
        tags: Free-form string metadata for filtering / grouping in
            the dashboard. Frozen view.

    Invariants (enforced in :meth:`__post_init__`):
        - ``framework`` is a :class:`TrainingFramework` member.
        - ``entry_point`` is a non-empty string.
        - ``output_dir`` is a :class:`Path`.
        - ``schema_version >= 1``.
        - ``max_epochs`` and ``max_steps`` are positive ints when set.
        - ``seed`` is a non-negative int when set.
        - ``dataset_id`` (when set) and tag keys / values are non-empty
          strings.
    """

    framework: TrainingFramework
    entry_point: str
    output_dir: Path
    schema_version: int = CURRENT_SCHEMA_VERSION
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    dataset_id: str | None = None
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    max_epochs: int | None = None
    max_steps: int | None = None
    seed: int | None = None
    tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.framework, TrainingFramework):
            raise TrainingConfigError(
                f"framework must be a TrainingFramework, got {self.framework!r}"
            )
        if not isinstance(self.entry_point, str) or not self.entry_point.strip():
            raise TrainingConfigError("entry_point must be a non-empty string")
        if not isinstance(self.output_dir, Path):
            raise TrainingConfigError(
                f"output_dir must be a pathlib.Path (got {type(self.output_dir).__name__})"
            )
        if not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise TrainingConfigError(
                f"schema_version must be a positive int (got {self.schema_version!r})"
            )
        if not isinstance(self.resources, ResourceRequest):
            raise TrainingConfigError("resources must be a ResourceRequest instance")
        self._validate_optional_caps()
        self._validate_dataset_id()
        self._validate_tags()
        object.__setattr__(
            self,
            "hyperparameters",
            MappingProxyType(dict(self.hyperparameters)),
        )
        object.__setattr__(self, "tags", MappingProxyType(dict(self.tags)))

    def _validate_optional_caps(self) -> None:
        for name in ("max_epochs", "max_steps"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or value <= 0:
                raise TrainingConfigError(
                    f"{name} must be a positive int when set (got {value!r})"
                )
        if self.seed is not None and (not isinstance(self.seed, int) or self.seed < 0):
            raise TrainingConfigError(
                f"seed must be a non-negative int when set (got {self.seed!r})"
            )

    def _validate_dataset_id(self) -> None:
        if self.dataset_id is None:
            return
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise TrainingConfigError("dataset_id must be a non-empty string when set")

    def _validate_tags(self) -> None:
        if not isinstance(self.tags, Mapping):
            raise TrainingConfigError("tags must be a Mapping")
        for key, value in self.tags.items():
            if not isinstance(key, str) or not key:
                raise TrainingConfigError(
                    f"tag keys must be non-empty strings (got {key!r})"
                )
            if not isinstance(value, str):
                raise TrainingConfigError(
                    f"tag values must be strings (got {value!r} for {key!r})"
                )
