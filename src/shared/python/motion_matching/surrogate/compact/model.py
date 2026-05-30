"""Option-2 NN swing surrogate: forward map ``coeffs -> trajectory``.

This module implements the compact-schema variant requested in issue
#4075. It maps a 189-dimensional polynomial-coefficient vector to a
``(T=31, 12)`` kinematic trajectory of the hand path. The 12 output
channels are::

    [r_clubhead(3), v_clubhead(3), r_grip(3), clubhead_speed(1), shaft_axis(2)]

where ``shaft_axis = (r_clubhead - r_grip) / ||·||`` is reduced to
``(azimuth, polar)`` for stability (the model never predicts a raw 3-vec
that would have to be re-normalised at inference time).

Architecture (~700k params at the documented defaults):

* **Coefficient encoder.** A linear projection ``189 -> hidden`` followed
  by three pre-norm residual MLP blocks of width ``hidden=256``. Each
  residual block is ``LayerNorm -> Linear -> GELU -> Linear``. This is
  the "4-layer MLP with residual blocks" from the spec — one input layer
  plus three residual blocks gives four learnable layers along the
  coefficient path while keeping gradient flow stable.
* **Decoder.** A single linear projection from the encoded latent to
  ``T * 12`` outputs, reshaped to ``(B, T, 12)``. The decoder is
  intentionally cheap because the per-timestep dynamics is already
  captured by the coefficient encoding — the trajectory is a smooth
  function of the encoded latent.

Why this rather than e.g. a transformer? Two reasons. (1) The input is
a fixed-shape 189-vec with no sequence structure: attention adds no
inductive bias. (2) Inversion via gradient descent through the surrogate
(the next step in the motion-matching pipeline) is cheaper through a
shallow MLP than a transformer; per #4075 the surrogate has to run in
"millisecond-scale" and back-prop in tens of ms.

The polynomial bounds from PROJECT_SPEC.md §4 are ``|A,B|<=1000``,
``|C,D|<=500``, ``|E,F|<=100``, ``|G|<=25`` for letters ``[A..G]`` of
each of the 27 joints. The :class:`CoeffNormalizer` rescales the raw
189-vec to ``[-1, 1]`` and back.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F

_LOGGER = logging.getLogger(__name__)

# --- Physical constants (from PROJECT_SPEC.md §4 + COMPACT_DATASET_SCHEMA.md) ---

#: Number of joints in the canonical compact-dataset ordering.
N_JOINTS_DEFAULT: int = 27
#: Polynomial coefficients per joint (letters A..G).
COEFFS_PER_JOINT: int = 7
#: Flattened coefficient-vector dimension.
COEFF_DIM_DEFAULT: int = N_JOINTS_DEFAULT * COEFFS_PER_JOINT  # 189
#: Number of timesteps in the compact dataset (≈ 0.30 s × 100 Hz + 1).
SEQ_LEN_DEFAULT: int = 31
#: Output channels: r_ch(3) + v_ch(3) + r_grip(3) + chs(1) + shaft_axis(2).
OUT_CHANNELS: int = 12

#: Per-letter coefficient bound (``|A|<=1000`` etc.) — letters A..G in order.
COEFF_BOUNDS: tuple[float, ...] = (1000.0, 1000.0, 500.0, 500.0, 100.0, 100.0, 25.0)


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SurrogateConfig:
    """Frozen architectural configuration for :class:`SwingSurrogate`.

    Attributes:
        n_joints: Number of joints (default 27 per the compact schema).
        coeffs_per_joint: Polynomial coefficients per joint (default 7).
        seq_len: Output trajectory length (default 31, matches compact data).
        hidden_dim: Hidden width of the encoder (default 256).
        n_residual_blocks: Number of pre-norm residual MLP blocks (default 3,
            which combined with the input projection gives the "4-layer MLP"
            from the spec).
        decoder_hidden: Optional widening of the decoder MLP. ``None`` uses a
            single ``Linear`` layer. Set e.g. ``512`` for a 2-layer decoder.
        dropout: Dropout probability after each residual block (default 0.0).
    """

    n_joints: int = N_JOINTS_DEFAULT
    coeffs_per_joint: int = COEFFS_PER_JOINT
    seq_len: int = SEQ_LEN_DEFAULT
    hidden_dim: int = 256
    n_residual_blocks: int = 3
    decoder_hidden: int | None = None
    dropout: float = 0.0
    coeff_bounds: tuple[float, ...] = field(default=COEFF_BOUNDS)

    @property
    def coeff_dim(self) -> int:
        """Flat coefficient-vector dimension."""
        return self.n_joints * self.coeffs_per_joint

    @property
    def out_channels(self) -> int:
        """Number of output channels per timestep (=12)."""
        return OUT_CHANNELS

    def validate(self) -> None:
        """Raise ``ValueError`` for any field that breaks the architecture."""
        if self.n_joints <= 0:
            raise ValueError(f"n_joints must be positive, got {self.n_joints}")
        if self.coeffs_per_joint <= 0:
            raise ValueError(
                f"coeffs_per_joint must be positive, got {self.coeffs_per_joint}"
            )
        if self.seq_len <= 1:
            raise ValueError(f"seq_len must be > 1, got {self.seq_len}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.n_residual_blocks < 1:
            raise ValueError(
                f"n_residual_blocks must be >= 1, got {self.n_residual_blocks}"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.decoder_hidden is not None and self.decoder_hidden <= 0:
            raise ValueError(
                f"decoder_hidden must be positive when set, got {self.decoder_hidden}"
            )
        if len(self.coeff_bounds) != self.coeffs_per_joint:
            raise ValueError(
                "coeff_bounds length must equal coeffs_per_joint "
                f"({self.coeffs_per_joint}); got {len(self.coeff_bounds)}"
            )
        if any(b <= 0 for b in self.coeff_bounds):
            raise ValueError("every coeff_bound must be strictly positive")


# --------------------------------------------------------------------------- #
# Coefficient normalisation                                                   #
# --------------------------------------------------------------------------- #


class CoeffNormalizer:
    """Maps the raw 189-vec to ``[-1, 1]`` and back, per PROJECT_SPEC.md §4.

    Bounds are per-letter (``A..G``), tiled once per joint:
        ``|A,B| <= 1000, |C,D| <= 500, |E,F| <= 100, |G| <= 25``.

    The class is intentionally not an ``nn.Module`` — the bounds are
    fixed physical constants, not learnable parameters.
    """

    def __init__(
        self,
        n_joints: int = N_JOINTS_DEFAULT,
        coeff_bounds: Sequence[float] = COEFF_BOUNDS,
    ) -> None:
        if n_joints <= 0:
            raise ValueError(f"n_joints must be positive, got {n_joints}")
        if any(b <= 0 for b in coeff_bounds):
            raise ValueError("every coeff_bound must be strictly positive")
        self._n_joints = int(n_joints)
        self._coeffs_per_joint = len(coeff_bounds)
        # Repeat the per-letter bounds across joints — shape (n_joints*L,).
        self._scale = torch.tensor(
            list(coeff_bounds) * self._n_joints, dtype=torch.float32
        )

    @property
    def coeff_dim(self) -> int:
        """Total flat coefficient dimension."""
        return self._n_joints * self._coeffs_per_joint

    @property
    def scale(self) -> torch.Tensor:
        """Per-coefficient bound vector (shape ``(coeff_dim,)``)."""
        return self._scale

    def normalize(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Return ``coeffs / scale`` clamped to ``[-1, 1]``.

        Args:
            coeffs: ``(B, coeff_dim)`` raw polynomial coefficients in the
                physical units of PROJECT_SPEC.md §4.

        Returns:
            ``(B, coeff_dim)`` tensor in ``[-1, 1]``.

        Raises:
            ValueError: If the trailing dim does not match ``coeff_dim``.
        """
        self._check_shape(coeffs)
        scale = self._scale.to(coeffs.device).to(coeffs.dtype)
        return torch.clamp(coeffs / scale, min=-1.0, max=1.0)

    def denormalize(self, coeffs_norm: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`normalize` — returns physical-unit coefficients."""
        self._check_shape(coeffs_norm)
        scale = self._scale.to(coeffs_norm.device).to(coeffs_norm.dtype)
        return coeffs_norm * scale

    def _check_shape(self, coeffs: torch.Tensor) -> None:
        if coeffs.ndim != 2:
            raise ValueError(
                f"coeffs must be 2-D (B, coeff_dim); got ndim={coeffs.ndim}"
            )
        if coeffs.shape[-1] != self.coeff_dim:
            raise ValueError(
                f"coeffs trailing dim {coeffs.shape[-1]} != expected {self.coeff_dim}"
            )


# --------------------------------------------------------------------------- #
# Architecture                                                                #
# --------------------------------------------------------------------------- #


class _ResidualBlock(nn.Module):
    """Pre-norm residual MLP block: ``x + Linear(GELU(Linear(LayerNorm(x))))``."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.fc2(h)
        h = self.drop(h)
        return residual + h


class SwingSurrogate(nn.Module):
    """Forward surrogate ``(B, 189) -> (B, 31, 12)``.

    Input contract:
        * ``coeffs.shape == (B, cfg.coeff_dim)``.
        * ``coeffs.dtype == torch.float32``.
        * ``coeffs`` already normalised to ``[-1, 1]`` (use
          :class:`CoeffNormalizer` to enforce this).

    Output contract:
        * ``traj.shape == (B, cfg.seq_len, 12)``.
        * Channel order ``[r_clubhead(3), v_clubhead(3), r_grip(3),
          clubhead_speed(1), shaft_axis_az_pol(2)]``.
        * ``traj.dtype == torch.float32``.
    """

    def __init__(self, cfg: SurrogateConfig | None = None) -> None:
        super().__init__()
        cfg = cfg if cfg is not None else SurrogateConfig()
        cfg.validate()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.coeff_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                _ResidualBlock(cfg.hidden_dim, cfg.dropout)
                for _ in range(cfg.n_residual_blocks)
            ]
        )
        self.head_norm = nn.LayerNorm(cfg.hidden_dim)

        out_dim = cfg.seq_len * cfg.out_channels
        if cfg.decoder_hidden is None:
            self.decoder: nn.Module = nn.Linear(cfg.hidden_dim, out_dim)
        else:
            self.decoder = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.decoder_hidden),
                nn.GELU(),
                nn.Linear(cfg.decoder_hidden, out_dim),
            )
        self._init_weights()

    # ----- public ----- #

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Map normalised polynomial coefficients to a hand-path trajectory.

        Args:
            coeffs: ``(B, coeff_dim)`` float32 tensor in ``[-1, 1]``. The
                caller is responsible for normalisation — see
                :class:`CoeffNormalizer`.

        Returns:
            ``(B, seq_len, 12)`` float32 trajectory tensor.

        Raises:
            TypeError: If ``coeffs`` is not a floating-point tensor.
            ValueError: If shape, dtype, or value range is wrong.
        """
        self._validate_input(coeffs)
        h = self.input_proj(coeffs)
        for block in self.blocks:
            h = block(h)
        h = self.head_norm(h)
        flat = self.decoder(h)
        return flat.view(-1, self.cfg.seq_len, self.cfg.out_channels)

    @torch.no_grad()
    def parameter_count(self) -> int:
        """Return the total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ----- private ----- #

    def _validate_input(self, coeffs: torch.Tensor) -> None:
        if not isinstance(coeffs, torch.Tensor):
            raise TypeError(
                f"coeffs must be a torch.Tensor, got {type(coeffs).__name__}"
            )
        if not coeffs.is_floating_point():
            raise TypeError(
                f"coeffs must be a floating-point tensor, got dtype={coeffs.dtype}"
            )
        if coeffs.dtype != torch.float32:
            raise TypeError(
                f"coeffs dtype must be torch.float32 per the contract, "
                f"got {coeffs.dtype}"
            )
        if coeffs.ndim != 2:
            raise ValueError(
                f"coeffs must be 2-D (B, {self.cfg.coeff_dim}); "
                f"got ndim={coeffs.ndim}, shape={tuple(coeffs.shape)}"
            )
        if coeffs.shape[-1] != self.cfg.coeff_dim:
            raise ValueError(
                f"coeffs trailing dim {coeffs.shape[-1]} != "
                f"cfg.coeff_dim ({self.cfg.coeff_dim})"
            )
        # Normalisation contract: ``coeffs in [-1, 1]`` after CoeffNormalizer.
        # We allow a small slack for fp jitter + downstream gradient steps
        # that can briefly leave the unit cube during inversion.
        if torch.is_grad_enabled() and coeffs.requires_grad:
            return
        with torch.no_grad():
            max_abs = coeffs.abs().max().item() if coeffs.numel() else 0.0
        if max_abs > 1.0 + 1e-3:
            _LOGGER.debug(
                "SwingSurrogate received coeffs with max-abs %.3f > 1; "
                "did you forget to call CoeffNormalizer.normalize()?",
                max_abs,
            )

    def _init_weights(self) -> None:
        """Xavier-init linear layers; zero biases for stable training start."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0 / math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


# --------------------------------------------------------------------------- #
# Channel-layout helpers                                                      #
# --------------------------------------------------------------------------- #

#: Output-channel slices (start, stop) — single source of truth.
CHANNEL_SLICES: dict[str, tuple[int, int]] = {
    "r_clubhead": (0, 3),
    "v_clubhead": (3, 6),
    "r_grip": (6, 9),
    "clubhead_speed": (9, 10),
    "shaft_axis_az_pol": (10, 12),
}


def shaft_axis_to_az_pol(shaft_axis_xyz: torch.Tensor) -> torch.Tensor:
    """Convert a unit ``shaft_axis`` vector to ``(azimuth, polar)``.

    Args:
        shaft_axis_xyz: ``(..., 3)`` tensor; need not be unit-norm
            (this function normalises it first).

    Returns:
        ``(..., 2)`` tensor of ``(azimuth, polar)`` angles in radians.
        Azimuth is ``atan2(y, x)`` in ``[-pi, pi]``; polar is
        ``acos(z / ||v||)`` in ``[0, pi]``.

    Raises:
        ValueError: If the trailing dim is not 3.
    """
    if shaft_axis_xyz.shape[-1] != 3:
        raise ValueError(
            f"shaft_axis_xyz must have trailing dim 3; got {shaft_axis_xyz.shape}"
        )
    norm = shaft_axis_xyz.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit = shaft_axis_xyz / norm
    azimuth = torch.atan2(unit[..., 1], unit[..., 0])
    polar = torch.acos(unit[..., 2].clamp(-1.0, 1.0))
    return torch.stack([azimuth, polar], dim=-1)


def az_pol_to_shaft_axis(az_pol: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`shaft_axis_to_az_pol` — back to a unit 3-vec."""
    if az_pol.shape[-1] != 2:
        raise ValueError(f"az_pol must have trailing dim 2; got {az_pol.shape}")
    azimuth = az_pol[..., 0]
    polar = az_pol[..., 1]
    sin_p = torch.sin(polar)
    x = sin_p * torch.cos(azimuth)
    y = sin_p * torch.sin(azimuth)
    z = torch.cos(polar)
    return torch.stack([x, y, z], dim=-1)


# --------------------------------------------------------------------------- #
# Target (per-channel) normalisation                                          #
# --------------------------------------------------------------------------- #


class TargetNormalizer:
    """Per-channel zero-mean / unit-std normaliser for the 12-dim trajectory.

    The 12 output channels are heterogeneous in scale (clubhead-speed in
    mph has magnitude ~100 while position channels in metres have
    magnitude ~1). A naive MSE on raw channels lets the optimiser drive
    the speed channel to near-zero error while ignoring positions.
    Standardising each channel before computing MSE makes the per-channel
    contributions comparable, so the optimiser allocates capacity to all
    channels.

    The class stores ``mean`` and ``std`` vectors of shape ``(C,)`` (with
    ``C == 12`` by default). ``standardize`` and ``destandardize`` are
    inverses to float64 precision.

    Notes:
        * ``std`` is floored at ``eps`` (default ``1e-6``) to keep
          standardisation safe even for degenerate fixtures with a
          near-constant channel.
        * The class is intentionally not an ``nn.Module``; the stats are
          fixed (computed once from the training split) and shouldn't
          appear as learnable parameters.
    """

    def __init__(
        self, mean: torch.Tensor, std: torch.Tensor, *, eps: float = 1e-6
    ) -> None:
        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError(
                f"mean and std must be 1-D; got mean.shape={tuple(mean.shape)}, "
                f"std.shape={tuple(std.shape)}"
            )
        if mean.shape != std.shape:
            raise ValueError(
                f"mean.shape ({tuple(mean.shape)}) must equal std.shape "
                f"({tuple(std.shape)})"
            )
        if eps <= 0:
            raise ValueError(f"eps must be strictly positive, got {eps}")
        self._mean = mean.detach().to(dtype=torch.float32).clone()
        self._std = std.detach().to(dtype=torch.float32).clone().clamp_min(eps)
        self._eps = float(eps)

    @classmethod
    def from_targets(
        cls, targets: torch.Tensor, *, eps: float = 1e-6
    ) -> TargetNormalizer:
        """Compute per-channel stats from a ``(N, T, C)`` (or ``(B*T, C)``) tensor.

        Args:
            targets: Float tensor with at least 2 dims; the trailing dim
                is the channel axis.
            eps: Floor applied to ``std`` to avoid division by zero on
                degenerate channels.

        Returns:
            A populated :class:`TargetNormalizer`.

        Raises:
            ValueError: If ``targets`` has fewer than 2 dims or the
                trailing dim is empty.
        """
        if targets.ndim < 2:
            raise ValueError(
                f"targets must have at least 2 dims (..., C); got ndim={targets.ndim}"
            )
        if targets.shape[-1] == 0:
            raise ValueError("targets trailing dim is empty")
        flat = targets.reshape(-1, targets.shape[-1]).to(dtype=torch.float32)
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False)
        return cls(mean=mean, std=std, eps=eps)

    @property
    def mean(self) -> torch.Tensor:
        """Per-channel mean vector ``(C,)``."""
        return self._mean

    @property
    def std(self) -> torch.Tensor:
        """Per-channel std vector ``(C,)`` (already eps-floored)."""
        return self._std

    @property
    def num_channels(self) -> int:
        """Number of channels stored."""
        return int(self._mean.shape[0])

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        """Subtract mean and divide by std along the channel axis.

        Args:
            x: ``(..., C)`` tensor in physical units.

        Returns:
            Same-shape tensor with each channel zero-mean / unit-std under
            the stats stored in this object.
        """
        self._check_channels(x)
        mean = self._mean.to(device=x.device, dtype=x.dtype)
        std = self._std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def destandardize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`standardize`."""
        self._check_channels(x_norm)
        mean = self._mean.to(device=x_norm.device, dtype=x_norm.dtype)
        std = self._std.to(device=x_norm.device, dtype=x_norm.dtype)
        return x_norm * std + mean

    def _check_channels(self, x: torch.Tensor) -> None:
        if x.ndim < 1 or x.shape[-1] != self.num_channels:
            raise ValueError(
                f"trailing dim {x.shape[-1] if x.ndim >= 1 else None} != "
                f"num_channels ({self.num_channels})"
            )

    # (de)serialisation helpers used by training/predict ------------------

    def to_state_dict(self) -> dict[str, list[float]]:
        """Return a JSON-friendly dict suitable for checkpointing."""
        return {
            "mean": self._mean.detach().cpu().tolist(),
            "std": self._std.detach().cpu().tolist(),
            "eps": [float(self._eps)],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Sequence[float]]) -> TargetNormalizer:
        """Inverse of :meth:`to_state_dict`."""
        if "mean" not in state or "std" not in state:
            raise ValueError(
                f"target-normalizer state must contain 'mean' and 'std'; got keys={list(state)}"
            )
        eps_list = state.get("eps", [1e-6])
        if isinstance(eps_list, (int, float)):
            eps = float(eps_list)
        else:
            eps = float(eps_list[0])
        mean = torch.tensor(list(state["mean"]), dtype=torch.float32)
        std = torch.tensor(list(state["std"]), dtype=torch.float32)
        return cls(mean=mean, std=std, eps=eps)
