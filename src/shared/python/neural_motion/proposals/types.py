"""Typed contracts for NM-06 masked trajectory-to-control proposals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.shared.python.neural_motion.tasks import (
    MaskedTrajectoryTask,
    dimensions_from_model,
)

__all__ = [
    "PROPOSAL_SCHEMA",
    "ProposalConfig",
    "ProposalFromTaskSpec",
    "ProposalMode",
]

PROPOSAL_SCHEMA = "neural-masked-proposals/1.0.0"

_LEGACY_COEFFICIENT_WIDTH = 189


class ProposalMode(str, Enum):
    """Deterministic selection path versus bounded mixture ablation."""

    SELECTION_OBJECTIVE = "selection_objective"
    MIXTURE_ABLATION = "mixture_ablation"


@dataclass(frozen=True, slots=True)
class ProposalFromTaskSpec:
    """Optional architecture overrides for :meth:`ProposalConfig.from_task`."""

    n_proposals: int = 1
    mode: ProposalMode = ProposalMode.SELECTION_OBJECTIVE
    include_solver_hints: bool = False
    seed: int = 0
    embed_dim: int = 32
    mlp_hidden: int = 64
    n_blocks: int = 2
    selection_objective: str = "min_effort"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ProposalMode):
            raise TypeError("mode must be a ProposalMode")
        for name, value in (
            ("n_proposals", self.n_proposals),
            ("embed_dim", self.embed_dim),
            ("mlp_hidden", self.mlp_hidden),
            ("n_blocks", self.n_blocks),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if not self.selection_objective.strip():
            raise ValueError("selection_objective must be non-empty")


@dataclass(frozen=True, slots=True)
class ProposalConfig:
    """Architecture and identity for a masked proposal model."""

    model_id: str
    u_dim: int
    control_basis: str
    seq_len: int
    obs_channels: int
    q_dim: int
    embed_dim: int = 32
    mlp_hidden: int = 64
    n_blocks: int = 2
    n_proposals: int = 1
    mode: ProposalMode = ProposalMode.SELECTION_OBJECTIVE
    selection_objective: str = "min_effort"
    include_solver_hints: bool = False
    seed: int = 0
    schema: str = PROPOSAL_SCHEMA

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not isinstance(self.control_basis, str) or not self.control_basis.strip():
            raise ValueError("control_basis must be a non-empty string")
        if not isinstance(self.mode, ProposalMode):
            raise TypeError("mode must be a ProposalMode")
        registered_u = int(dimensions_from_model(self.model_id).u_dim)
        if self.u_dim != registered_u:
            raise ValueError(
                f"u_dim={self.u_dim} does not match registered independent_dof="
                f"{registered_u} for {self.model_id}; refuse hard-coded "
                f"{_LEGACY_COEFFICIENT_WIDTH} (or mismatched) primary control dim"
            )
        for name, value in (
            ("u_dim", self.u_dim),
            ("seq_len", self.seq_len),
            ("obs_channels", self.obs_channels),
            ("q_dim", self.q_dim),
            ("embed_dim", self.embed_dim),
            ("mlp_hidden", self.mlp_hidden),
            ("n_blocks", self.n_blocks),
            ("n_proposals", self.n_proposals),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.mode is ProposalMode.SELECTION_OBJECTIVE and self.n_proposals != 1:
            raise ValueError("selection_objective mode requires n_proposals == 1")
        if self.mode is ProposalMode.MIXTURE_ABLATION and self.n_proposals < 2:
            raise ValueError("mixture_ablation requires n_proposals >= 2")
        if not self.selection_objective.strip():
            raise ValueError("selection_objective must be non-empty")

    @classmethod
    def from_task(
        cls,
        task: MaskedTrajectoryTask,
        *,
        control_basis: str,
        seq_len: int,
        overrides: ProposalFromTaskSpec | None = None,
    ) -> ProposalConfig:
        """Build a config from a task plus required wire fields.

        Optional architecture knobs travel through :class:`ProposalFromTaskSpec`
        so this constructor stays within the parameter-count budget.
        """
        if not isinstance(task, MaskedTrajectoryTask):
            raise TypeError("task must be a MaskedTrajectoryTask")
        spec = overrides or ProposalFromTaskSpec()
        dims = task.dimensions
        return cls(
            model_id=dims.model_id,
            u_dim=dims.u_dim,
            control_basis=control_basis,
            seq_len=int(seq_len),
            obs_channels=dims.q_dim,
            q_dim=dims.q_dim,
            embed_dim=spec.embed_dim,
            mlp_hidden=spec.mlp_hidden,
            n_blocks=spec.n_blocks,
            n_proposals=spec.n_proposals,
            mode=spec.mode,
            selection_objective=spec.selection_objective,
            include_solver_hints=spec.include_solver_hints,
            seed=int(spec.seed),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_id": self.model_id,
            "u_dim": self.u_dim,
            "control_basis": self.control_basis,
            "seq_len": self.seq_len,
            "obs_channels": self.obs_channels,
            "q_dim": self.q_dim,
            "embed_dim": self.embed_dim,
            "mlp_hidden": self.mlp_hidden,
            "n_blocks": self.n_blocks,
            "n_proposals": self.n_proposals,
            "mode": self.mode.value,
            "selection_objective": self.selection_objective,
            "include_solver_hints": self.include_solver_hints,
            "seed": self.seed,
        }
