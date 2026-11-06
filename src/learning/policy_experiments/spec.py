"""Typed experiment spec for neural swing policy studies (NP-01, #14476).

An :class:`ExperimentSpec` pins model, revision, task, demonstration identity,
policy architecture, seeds, intervention (localized edit, constrained
fine-tune, or ``None`` for the baseline) and budget. Each part validates itself.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass
from typing import Literal

from src.sim_run.manifest import REVISION_PATTERN
from src.sim_run.manifest_emitter import sha256_json

ACTIVATIONS = frozenset({"tanh", "relu", "elu", "gelu"})
_DATA_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class ExperimentSpecError(ValueError):
    """Raised when an experiment spec or one of its parts is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentSpecError(message)


def _require_text(value: str, field: str) -> None:
    _require(isinstance(value, str) and bool(value.strip()), f"{field} is blank")


def _require_positive(value: float, field: str) -> None:
    _require(math.isfinite(value) and value > 0, f"{field} must be finite and > 0")


@dataclass(frozen=True)
class PolicyArch:
    """MLP policy shape: hidden layer widths and the hidden activation."""

    hidden_sizes: tuple[int, ...]
    activation: str

    def __post_init__(self) -> None:
        _require(bool(self.hidden_sizes), "hidden_sizes is empty")
        _require(all(n > 0 for n in self.hidden_sizes), "hidden sizes must be > 0")
        _require(
            self.activation in ACTIVATIONS,
            f"activation {self.activation!r} not in {sorted(ACTIVATIONS)}",
        )


@dataclass(frozen=True)
class EditRecipe:
    """A localized low-rank weight edit: ``W += scale * u v^T`` (rank ``rank``)."""

    layer: str
    rank: int
    scale: float
    kind: Literal["edit"] = "edit"

    def __post_init__(self) -> None:
        _require_text(self.layer, "edit layer")
        _require(self.rank > 0, "edit rank must be > 0")
        _require(math.isfinite(self.scale), "edit scale must be finite")


@dataclass(frozen=True)
class FineTuneConfig:
    """A constrained fine-tune; ``lora_rank`` ``None`` means full weights."""

    steps: int
    lr: float
    lora_rank: int | None = None
    kind: Literal["fine_tune"] = "fine_tune"

    def __post_init__(self) -> None:
        _require(self.steps > 0, "fine-tune steps must be > 0")
        _require_positive(self.lr, "fine-tune lr")
        _require(self.lora_rank is None or self.lora_rank > 0, "lora_rank must be > 0")


@dataclass(frozen=True)
class ComputeBudget:
    """Upper bounds the study may spend: forward passes and wall time."""

    max_forward_passes: int
    max_wall_s: float

    def __post_init__(self) -> None:
        _require(self.max_forward_passes > 0, "max_forward_passes must be > 0")
        _require_positive(self.max_wall_s, "max_wall_s")


@dataclass(frozen=True)
class ExperimentSpec:
    """Everything that identifies one policy study (see module doc).

    Preconditions (:class:`ExperimentSpecError`): non-blank text, an immutable
    ``model_revision`` (#14436 manifest rule), a ``sha256:`` demonstration
    digest, and distinct seeds >= 0.
    """

    model_id: str
    model_revision: str
    task: str
    demonstration_source: str
    policy: PolicyArch
    seeds: tuple[int, ...]
    intervention: EditRecipe | FineTuneConfig | None
    budget: ComputeBudget

    def __post_init__(self) -> None:
        _require_text(self.model_id, "model_id")
        _require_text(self.task, "task")
        _require(
            re.fullmatch(REVISION_PATTERN, self.model_revision or "") is not None,
            f"model_revision {self.model_revision!r} is not an immutable revision",
        )
        _require(
            _DATA_DIGEST.fullmatch(self.demonstration_source or "") is not None,
            "demonstration_source must be a sha256: digest of the data",
        )
        distinct = len(set(self.seeds)) == len(self.seeds)
        ok = bool(self.seeds) and distinct and min(self.seeds, default=-1) >= 0
        _require(ok, f"seeds {self.seeds} must be non-empty, distinct and >= 0")

    def as_dict(self) -> dict[str, object]:
        """The spec as plain JSON-ready data (tuples become lists)."""
        return dataclasses.asdict(self)

    def digest(self) -> str:
        """SHA-256 of the spec's canonical JSON; equal specs share a digest."""
        return sha256_json(self.as_dict())

    def require_seed(self, seed: int) -> None:
        """DbC: ``seed`` is one this spec declares."""
        _require(seed in self.seeds, f"seed {seed} is not in the spec {self.seeds}")
