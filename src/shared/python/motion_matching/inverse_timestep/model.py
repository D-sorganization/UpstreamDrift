"""Per-timestep inverse-dynamics MLP: ``(q, qd, qdd) -> tau``.

Each timestep is treated as an independent sample. State vector is
``concat(q, qd, qdd)`` of size 3*27 = 81. Output is the 27-dim torque
vector. NaN inputs (e.g. unmapped DOFs like ``LSAngularAccelerationZ``)
are zero-filled with a per-DOF "missing" indicator channel appended,
yielding ``81 + 81 = 162`` effective input features.

Architecture mirrors the surrogate's ``_ResidualBlock`` style: a
LayerNorm-input projection followed by ``n_blocks`` pre-norm residual
blocks of width ``hidden``, then a final norm + linear head to 27 dims.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.shared.python.motion_matching._checkpoint_artifacts import (
    load_checkpoint_dict,
    require_schema_version,
)

DEFAULT_INPUT_DIM = 81  # 3 * 27 (q, qd, qdd)
DEFAULT_OUTPUT_DIM = 27
DEFAULT_HIDDEN = 256
DEFAULT_N_BLOCKS = 4
DEFAULT_DROPOUT = 0.1


@dataclass(frozen=True)
class TimestepInverseConfig:
    """Architectural hyperparameters for :class:`TimestepInverseDynamics`.

    Defaults yield ~280k parameters with a 162-feature effective input
    (81 raw + 81 missing-indicator) and ``hidden=256``, ``n_blocks=4``.
    """

    input_dim: int = DEFAULT_INPUT_DIM
    output_dim: int = DEFAULT_OUTPUT_DIM
    hidden: int = DEFAULT_HIDDEN
    n_blocks: int = DEFAULT_N_BLOCKS
    dropout: float = DEFAULT_DROPOUT
    use_missing_indicator: bool = True

    def validate(self) -> None:
        """Raise ``ValueError`` for any field that breaks the architecture."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if self.hidden <= 0:
            raise ValueError(f"hidden must be positive, got {self.hidden}")
        if self.n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {self.n_blocks}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def effective_input_dim(self) -> int:
        if self.use_missing_indicator:
            return self.input_dim * 2
        return self.input_dim


class _ResidualBlock(nn.Module):
    """Pre-norm residual MLP block: ``x + Linear(GELU(Linear(LayerNorm(x))))``."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        h = self.norm(x)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.fc2(h)
        h = self.drop(h)
        return residual + h


class TimestepInverseDynamics(nn.Module):
    """MLP mapping per-timestep ``(q, qd, qdd)`` to applied torque.

    Input contract (``state``):
        * ``state.shape == (B, cfg.input_dim)`` (default 81).
        * ``state.dtype == torch.float32``.
        * NaN entries are tolerated and replaced with 0.0; if
          ``cfg.use_missing_indicator`` is True (default), a per-DOF
          binary indicator channel is appended internally so the MLP can
          distinguish "value 0.0" from "missing".

    Output contract:
        * ``out.shape == (B, cfg.output_dim)`` (default 27).
        * ``out.dtype == torch.float32``.
        * In the standardised space the model is trained on; callers must
          de-standardise using the per-DOF stats stored in the checkpoint
          payload to recover Newton-metres.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __init__(self, cfg: TimestepInverseConfig | None = None) -> None:
        super().__init__()
        cfg = cfg if cfg is not None else TimestepInverseConfig()
        cfg.validate()
        self.cfg = cfg

        eff_in = cfg.effective_input_dim
        self.input_proj = nn.Linear(eff_in, cfg.hidden)
        self.input_norm = nn.LayerNorm(cfg.hidden)
        self.blocks = nn.ModuleList(
            [_ResidualBlock(cfg.hidden, cfg.dropout) for _ in range(cfg.n_blocks)]
        )
        self.head_norm = nn.LayerNorm(cfg.hidden)
        self.head = nn.Linear(cfg.hidden, cfg.output_dim)

    # ----- public ----- #

    def forward(self, state: Tensor) -> Tensor:
        """Predict the standardised torque vector for a batch of states.

        Args:
            state: ``(B, cfg.input_dim)`` float32 tensor of standardised
                ``concat(q, qd, qdd)`` values, possibly containing NaN
                for unmapped DOFs.

        Returns:
            ``(B, cfg.output_dim)`` float32 tensor of standardised torques.

        Raises:
            TypeError: If ``state`` is not a float32 tensor.
            ValueError: If the shape/rank is wrong or values are non-finite
                (Inf rejected; NaN tolerated and zero-filled).
        """
        self._validate_input(state)
        x = self._apply_missing_indicator(state)
        h = self.input_proj(x)
        h = self.input_norm(h)
        for block in self.blocks:
            h = block(h)
        h = self.head_norm(h)
        return self.head(h)

    def state_payload(self) -> dict:
        """Return a checkpoint-ready dict bundling weights and config."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "config": _config_to_dict(self.cfg),
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> TimestepInverseDynamics:
        """Re-instantiate a model from a payload produced by ``state_payload``.

        The returned model is in eval mode and has its weights loaded.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the payload is missing required keys.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        payload = load_checkpoint_dict(
            ckpt_path,
            map_location=map_location,
            required_keys=("state_dict", "config", "schema_version"),
            artifact_name="TimestepInverseDynamics checkpoint",
        )
        require_schema_version(
            payload,
            cls.SCHEMA_VERSION,
            artifact_name="TimestepInverseDynamics checkpoint",
        )
        cfg_dict = payload.get("config")
        if cfg_dict is None:
            raise ValueError("checkpoint missing 'config' entry")
        cfg = _config_from_dict(cfg_dict)
        model = cls(cfg)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    # ----- private ----- #

    def _apply_missing_indicator(self, state: Tensor) -> Tensor:
        """Replace NaN with 0 and (optionally) append a missing-mask channel."""
        nan_mask = torch.isnan(state)
        clean = torch.where(
            nan_mask,
            torch.zeros_like(state),
            state,
        )
        if not self.cfg.use_missing_indicator:
            return clean
        indicator = nan_mask.to(state.dtype)
        return torch.cat([clean, indicator], dim=-1)

    def _validate_input(self, state: Tensor) -> None:
        if not isinstance(state, Tensor):
            raise TypeError(f"state must be torch.Tensor, got {type(state).__name__}")
        if state.dim() != 2:
            raise ValueError(
                f"state must be 2-D (B, input_dim); got shape {tuple(state.shape)}"
            )
        if state.shape[-1] != self.cfg.input_dim:
            raise ValueError(
                f"state last-dim must be {self.cfg.input_dim}; got {state.shape[-1]}"
            )
        if state.dtype != torch.float32:
            raise TypeError(f"state must be float32; got {state.dtype}")
        # Inf is always rejected (a real numerical bug). NaN is tolerated
        # because unmapped DOFs legitimately carry NaN per the schema.
        if torch.isinf(state).any():
            raise ValueError("state contains Inf values (NaN is tolerated)")


# ---------------------------------------------------------------------------
# Config (de)serialisation helpers
# ---------------------------------------------------------------------------


def _config_to_dict(cfg: TimestepInverseConfig) -> dict:
    return {
        "input_dim": cfg.input_dim,
        "output_dim": cfg.output_dim,
        "hidden": cfg.hidden,
        "n_blocks": cfg.n_blocks,
        "dropout": cfg.dropout,
        "use_missing_indicator": cfg.use_missing_indicator,
    }


def _config_from_dict(d: dict) -> TimestepInverseConfig:
    return TimestepInverseConfig(
        input_dim=int(d.get("input_dim", DEFAULT_INPUT_DIM)),
        output_dim=int(d.get("output_dim", DEFAULT_OUTPUT_DIM)),
        hidden=int(d.get("hidden", DEFAULT_HIDDEN)),
        n_blocks=int(d.get("n_blocks", DEFAULT_N_BLOCKS)),
        dropout=float(d.get("dropout", DEFAULT_DROPOUT)),
        use_missing_indicator=bool(d.get("use_missing_indicator", True)),
    )
