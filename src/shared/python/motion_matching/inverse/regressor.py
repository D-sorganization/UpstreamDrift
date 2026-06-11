"""Deterministic inverse regressor: trajectory -> 189 coefficient vector.

Production alternative to the cVAE in :mod:`.cvae`. The cVAE exhibited a
hard reconstruction plateau on the real compact dataset (val_recon stuck
at the mean-prediction baseline for 18+ CPU epochs even after the bug-2
[-1, 1] standardisation + free-bits fixes). This module provides a much
simpler deterministic regressor: a 1-D conv stem embeds each timestep,
temporal pooling collapses the time axis, and a residual MLP maps the
flat feature to the 189-dim coefficient vector. Output is hard-clamped
to per-letter physical bounds via ``tanh(x) * bound`` (identical to the
cVAE decoder).

Architecture (~1-3 M params at the documented defaults):

* **Conv stem.** Two 1-D convolutions over the time axis with GELU +
  LayerNorm, embedding each of the 31 timesteps into ``embed_dim``
  features.
* **Temporal pool.** Concatenated mean + max pool over the T axis,
  yielding ``2 * embed_dim`` global features.
* **MLP body.** Pre-norm residual blocks of width ``mlp_hidden`` (same
  shape as :class:`SwingSurrogate._ResidualBlock` in
  :mod:`...surrogate.compact.model`).
* **Bounded head.** Linear to 189 raw values, then ``tanh * bounds``
  to keep every coefficient in its physical range.

DbC: :meth:`forward` validates input shape/dtype/rank and raises
``TypeError``/``ValueError`` with descriptive messages.
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

from .cvae import (
    COEFFICIENT_LETTER_BOUNDS,
    COEFFICIENTS_PER_JOINT,
    DEFAULT_COEFFICIENT_DIM,
    DEFAULT_N_JOINTS,
    DEFAULT_TRAJECTORY_CHANNELS,
    build_coefficient_bound_vector,
    parameter_count,
)

__all__ = [
    "COEFFICIENT_LETTER_BOUNDS",
    "DEFAULT_COEFFICIENT_DIM",
    "DEFAULT_N_JOINTS",
    "DEFAULT_TRAJECTORY_CHANNELS",
    "InverseRegressor",
    "RegressorConfig",
    "build_coefficient_bound_vector",
    "parameter_count",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressorConfig:
    """Architectural hyperparameters for :class:`InverseRegressor`.

    Defaults yield ~1.5 M parameters with ``embed_dim=64``,
    ``mlp_hidden=512``, ``n_blocks=4`` and ``temporal_aggregation='flatten'``
    on a fixed ``seq_len=31`` — comparable to the cVAE budget.

    The temporal-aggregation choice is critical:

    * ``'meanmax'`` — global mean+max over T. Lossy for time-localised
      signal; tends to plateau at the dataset variance baseline on the
      compact dataset.
    * ``'flatten'`` — concatenate per-timestep conv-stem features into a
      ``(seq_len * embed_dim)`` flat vector. Preserves time-localised
      information; recommended default per issue investigation.
    * ``'flatten_raw'`` — skip the conv stem and concatenate the raw
      ``(seq_len * trajectory_channels)`` trajectory directly into the
      MLP. The most aggressive preservation of per-timestep info; falls
      back to a pure-MLP architecture.

    The ``coefficient_scale_factor`` controls the symmetric range of the
    ``tanh * scale`` output head: ``scale = factor * coefficient_bounds``.
    The compact dataset's coefficients reach ~40x the nominal per-letter
    bounds, so the default of 1.0 (matching the cVAE) clamps the model
    to a range it cannot escape. Set to e.g. ``50.0`` to give the model
    enough output range to fit the empirical coefficient distribution.
    """

    n_joints: int = DEFAULT_N_JOINTS
    coefficients_per_joint: int = COEFFICIENTS_PER_JOINT
    trajectory_channels: int = DEFAULT_TRAJECTORY_CHANNELS
    seq_len: int = 31
    embed_dim: int = 64
    conv_kernel: int = 5
    mlp_hidden: int = 512
    n_blocks: int = 4
    dropout: float = 0.1
    temporal_aggregation: str = "flatten"
    coefficient_scale_factor: float = 50.0

    @property
    def coefficient_dim(self) -> int:
        return self.n_joints * self.coefficients_per_joint

    def validate(self) -> None:
        """Raise ``ValueError`` for any field that breaks the architecture."""
        self._validate_dims()
        self._validate_arch()
        self._validate_aggregation()

    def _validate_dims(self) -> None:
        if self.n_joints <= 0:
            raise ValueError(f"n_joints must be positive, got {self.n_joints}")
        if self.coefficients_per_joint <= 0:
            raise ValueError(
                "coefficients_per_joint must be positive, "
                f"got {self.coefficients_per_joint}"
            )
        if self.trajectory_channels <= 0:
            raise ValueError(
                f"trajectory_channels must be positive, got {self.trajectory_channels}"
            )
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")

    def _validate_arch(self) -> None:
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.mlp_hidden <= 0:
            raise ValueError(f"mlp_hidden must be positive, got {self.mlp_hidden}")
        if self.n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {self.n_blocks}")
        if self.conv_kernel < 1:
            raise ValueError(f"conv_kernel must be >= 1, got {self.conv_kernel}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def _validate_aggregation(self) -> None:
        if self.temporal_aggregation not in ("meanmax", "flatten", "flatten_raw"):
            raise ValueError(
                "temporal_aggregation must be 'meanmax', 'flatten', or "
                f"'flatten_raw'; got {self.temporal_aggregation!r}"
            )
        if self.coefficient_scale_factor <= 0:
            raise ValueError(
                "coefficient_scale_factor must be positive, "
                f"got {self.coefficient_scale_factor}"
            )
        # ``_ConvStem`` uses ``padding = kernel // 2`` for SAME padding,
        # which only preserves the time-axis length when the kernel is
        # odd. The ``flatten`` aggregation routes through ``_ConvStem``
        # and then sizes ``input_proj`` for ``seq_len * embed_dim``, so
        # an even kernel would crash at the first forward pass. Reject
        # even kernels for ``flatten`` up-front. ``flatten_raw`` skips
        # the conv stem entirely (see issue #4294 — codex review on PR
        # #4292) so kernel parity has no effect there.
        if self.temporal_aggregation == "flatten" and (self.conv_kernel % 2 == 0):
            raise ValueError(
                "conv_kernel must be odd when temporal_aggregation is "
                "'flatten' (SAME padding requires an odd kernel to "
                f"preserve seq_len); got conv_kernel={self.conv_kernel}"
            )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _ResidualBlock(nn.Module):
    """Pre-norm residual MLP block: ``x + Linear(GELU(Linear(LayerNorm(x))))``.

    Shape-preserving; mirrors the surrogate's ``_ResidualBlock``.
    """

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


class _ConvStem(nn.Module):
    """Two-layer 1-D conv stem ``(B, T, C) -> (B, T, embed_dim)``.

    Same kernel size on both layers with stride 1 and SAME padding so the
    time axis length is preserved. LayerNorm operates over the channel
    dimension (after transposing back to ``(B, T, C)``).
    """

    def __init__(
        self, in_channels: int, embed_dim: int, kernel: int, dropout: float
    ) -> None:
        super().__init__()
        padding = kernel // 2
        self.conv1 = nn.Conv1d(in_channels, embed_dim, kernel, padding=padding)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel, padding=padding)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, traj: Tensor) -> Tensor:
        # traj: (B, T, C) -> (B, C, T) for Conv1d
        h = traj.transpose(1, 2)
        h = self.conv1(h)
        # Back to (B, T, embed) for LayerNorm-over-channels
        h = h.transpose(1, 2)
        h = self.norm1(h)
        h = F.gelu(h)
        h = h.transpose(1, 2)
        h = self.conv2(h)
        h = h.transpose(1, 2)
        h = self.norm2(h)
        h = F.gelu(h)
        return self.drop(h)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class InverseRegressor(nn.Module):
    """Deterministic regressor mapping trajectory -> 189-dim coefficients.

    Input contract:
        * ``trajectory.shape == (B, T, cfg.trajectory_channels)``
        * ``trajectory.dtype == torch.float32``
        * Trajectory is in physical units (the trained model handles
          internal feature normalisation via LayerNorm).

    Output contract:
        * ``coeffs.shape == (B, cfg.coefficient_dim)``
        * ``coeffs.dtype == torch.float32``
        * Each coefficient in ``[-bound_i, +bound_i]`` for the per-letter
          physical bounds (PROJECT_SPEC.md §4).
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __init__(self, cfg: RegressorConfig | None = None) -> None:
        super().__init__()
        cfg = cfg if cfg is not None else RegressorConfig()
        cfg.validate()
        self.cfg = cfg

        self.stem = _ConvStem(
            in_channels=cfg.trajectory_channels,
            embed_dim=cfg.embed_dim,
            kernel=cfg.conv_kernel,
            dropout=cfg.dropout,
        )

        if cfg.temporal_aggregation == "meanmax":
            feature_dim = 2 * cfg.embed_dim
        elif cfg.temporal_aggregation == "flatten":
            feature_dim = cfg.seq_len * cfg.embed_dim
        else:  # 'flatten_raw' — bypasses the conv stem
            feature_dim = cfg.seq_len * cfg.trajectory_channels
        self.input_proj = nn.Linear(feature_dim, cfg.mlp_hidden)
        self.blocks = nn.ModuleList(
            [_ResidualBlock(cfg.mlp_hidden, cfg.dropout) for _ in range(cfg.n_blocks)]
        )
        self.head_norm = nn.LayerNorm(cfg.mlp_hidden)
        self.head = nn.Linear(cfg.mlp_hidden, cfg.coefficient_dim)

        # Non-trainable buffers:
        #  * coefficient_bounds: per-letter physical bounds (PROJECT_SPEC.md §4)
        #    — kept for downstream consumers that gate on the nominal range.
        #  * coefficient_scale: scale applied to ``tanh(x)`` in the output
        #    head. ``= bounds * scale_factor`` to give the model enough output
        #    range to fit empirical coefficients which exceed the nominal
        #    bounds in the compact dataset.
        bounds = build_coefficient_bound_vector(cfg.n_joints)
        self.register_buffer("coefficient_bounds", bounds, persistent=False)
        self.register_buffer(
            "coefficient_scale",
            bounds * cfg.coefficient_scale_factor,
            persistent=False,
        )

    # ----- public ----- #

    def forward(self, trajectory: Tensor) -> Tensor:
        """Map a trajectory to a 189-dim coefficient vector in physical units.

        Args:
            trajectory: ``(B, T, trajectory_channels)`` float32 tensor.

        Returns:
            ``(B, coefficient_dim)`` float32 tensor in physical units, with
            every entry in ``[-bound_i, +bound_i]``.

        Raises:
            TypeError: If ``trajectory`` is not a float32 tensor.
            ValueError: If the trajectory shape/rank is wrong.
        """
        self._validate_input(trajectory)
        if self.cfg.temporal_aggregation == "flatten_raw":
            # Skip the conv stem entirely — flatten raw (B, T, C) -> (B, T*C).
            pooled = trajectory.reshape(trajectory.shape[0], -1)
        else:
            embedded = self.stem(trajectory)
            if self.cfg.temporal_aggregation == "meanmax":
                pooled_mean = embedded.mean(dim=1)
                pooled_max = embedded.amax(dim=1)
                pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
            else:  # 'flatten' — preserve per-timestep conv features.
                pooled = embedded.reshape(embedded.shape[0], -1)
        h = self.input_proj(pooled)
        for block in self.blocks:
            h = block(h)
        h = self.head_norm(h)
        raw = self.head(h)
        scale = self.coefficient_scale
        assert isinstance(scale, Tensor)
        return torch.tanh(raw) * scale

    # ----- (de)serialisation ----- #

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
    ) -> InverseRegressor:
        """Re-instantiate a model from a payload produced by ``state_payload``.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the payload is missing keys.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        payload = load_checkpoint_dict(
            ckpt_path,
            map_location=map_location,
            required_keys=("state_dict", "config", "schema_version"),
            artifact_name="InverseRegressor checkpoint",
        )
        require_schema_version(
            payload,
            cls.SCHEMA_VERSION,
            artifact_name="InverseRegressor checkpoint",
        )
        cfg_dict = payload.get("config")
        if cfg_dict is None:
            raise ValueError("checkpoint missing 'config' entry")
        cfg = _config_from_dict(cfg_dict)
        model = cls(cfg)
        model.load_state_dict(payload["state_dict"])
        return model

    # ----- private ----- #

    def _validate_input(self, trajectory: Tensor) -> None:
        if not isinstance(trajectory, Tensor):
            raise TypeError(
                f"trajectory must be torch.Tensor, got {type(trajectory).__name__}"
            )
        if trajectory.dim() != 3:
            raise ValueError(
                f"trajectory must be 3-D (B, T, C); got shape {tuple(trajectory.shape)}"
            )
        if trajectory.shape[-1] != self.cfg.trajectory_channels:
            raise ValueError(
                f"trajectory last-dim must be {self.cfg.trajectory_channels}; "
                f"got {trajectory.shape[-1]}"
            )
        if trajectory.dtype != torch.float32:
            raise TypeError(f"trajectory must be float32; got {trajectory.dtype}")
        # 'flatten' / 'flatten_raw' aggregations have a fixed input length
        # set at construction; 'meanmax' is sequence-length-agnostic.
        if self.cfg.temporal_aggregation in ("flatten", "flatten_raw"):
            t_actual = trajectory.shape[1]
            if t_actual != self.cfg.seq_len:
                raise ValueError(
                    "trajectory T must equal cfg.seq_len when "
                    f"temporal_aggregation={self.cfg.temporal_aggregation!r}; "
                    f"got T={t_actual}, cfg.seq_len={self.cfg.seq_len}"
                )


# ---------------------------------------------------------------------------
# Config (de)serialisation helpers
# ---------------------------------------------------------------------------


def _config_to_dict(cfg: RegressorConfig) -> dict:
    return {
        "n_joints": cfg.n_joints,
        "coefficients_per_joint": cfg.coefficients_per_joint,
        "trajectory_channels": cfg.trajectory_channels,
        "seq_len": cfg.seq_len,
        "embed_dim": cfg.embed_dim,
        "conv_kernel": cfg.conv_kernel,
        "mlp_hidden": cfg.mlp_hidden,
        "n_blocks": cfg.n_blocks,
        "dropout": cfg.dropout,
        "temporal_aggregation": cfg.temporal_aggregation,
        "coefficient_scale_factor": cfg.coefficient_scale_factor,
    }


def _config_from_dict(d: dict) -> RegressorConfig:
    return RegressorConfig(
        n_joints=int(d["n_joints"]),
        coefficients_per_joint=int(d["coefficients_per_joint"]),
        trajectory_channels=int(d["trajectory_channels"]),
        seq_len=int(d.get("seq_len", 31)),
        embed_dim=int(d["embed_dim"]),
        conv_kernel=int(d["conv_kernel"]),
        mlp_hidden=int(d["mlp_hidden"]),
        n_blocks=int(d["n_blocks"]),
        dropout=float(d["dropout"]),
        temporal_aggregation=str(d.get("temporal_aggregation", "flatten")),
        coefficient_scale_factor=float(d.get("coefficient_scale_factor", 50.0)),
    )
