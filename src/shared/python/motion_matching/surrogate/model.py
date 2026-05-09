"""FiLM-conditioned MLP forward surrogate.

Implements the architecture sketched in ``option2_nn_surrogate/APPROACH.md``:

  coeffs (B, D) -> coeff encoder MLP -> (gamma, beta) per layer
  fixed sinusoidal time embedding (T, T_emb) -> per-timestep MLP backbone,
      modulated layer-wise by the FiLM (gamma, beta) from the encoder.

Decoded heads emit (butt, clubhead, q_club, joint_q) with a unit-norm
post-hoc projection on the quaternion head and ``w >= 0`` canonicalization.

Output is a :class:`ClubTrajectory` dataclass of tensors (NOT the
``ClubTarget`` numpy dataclass from ``club_target.py`` -- that one is
batchless and validated as a measured swing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.shared.python.core.contracts import postcondition, precondition

_QUAT_NORM_EPS = 1.0e-8


@dataclass(frozen=True)
class SurrogateConfig:
    """Frozen architectural configuration for :class:`SwingSurrogate`.

    Attributes:
        n_joints: Number of robot joints (drives input/aux-output dims).
        coeffs_per_joint: Polynomial coefficients per joint (default 7
            for the A..G slot convention).
        seq_len: Number of timesteps the surrogate emits (default 300 =
            0.3 s at 1 kHz).
        hidden_dim: Hidden width of every MLP layer.
        n_layers: Number of FiLM-modulated backbone layers.
        time_embed_dim: Sinusoidal time-embedding dimension.
        encoder_layers: Depth of the coefficient encoder MLP.
        dropout: Dropout probability applied after each backbone GELU.
    """

    n_joints: int
    coeffs_per_joint: int = 7
    seq_len: int = 300
    hidden_dim: int = 256
    n_layers: int = 3
    time_embed_dim: int = 64
    encoder_layers: int = 3
    dropout: float = 0.0

    @property
    def coeff_dim(self) -> int:
        """Flat coefficient-vector dimension ``D = n_joints * coeffs_per_joint``."""
        return self.n_joints * self.coeffs_per_joint


@dataclass
class ClubTrajectory:
    """Surrogate output. Mirrors CLUB_IK_SPEC.md target schema, batch-first.

    Tensor shapes (``B`` = batch, ``T`` = ``cfg.seq_len``, ``J`` = ``cfg.n_joints``):

    Attributes:
        butt:     ``(B, T, 3)`` metres, world frame.
        clubhead: ``(B, T, 3)`` metres, world frame.
        club_quat: ``(B, T, 4)`` unit quaternion ``[w, x, y, z]`` with ``w >= 0``.
        joint_q:  ``(B, T, J)`` auxiliary joint-angle predictions.
    """

    butt: torch.Tensor
    clubhead: torch.Tensor
    club_quat: torch.Tensor
    joint_q: torch.Tensor


def _build_time_embedding(seq_len: int, dim: int) -> torch.Tensor:
    """Build a fixed sinusoidal time embedding of shape ``(seq_len, dim)``."""
    if dim % 2 != 0:
        raise ValueError(f"time_embed_dim must be even, got {dim}")
    t = torch.linspace(0.0, 1.0, seq_len).unsqueeze(1)
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=torch.float32)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = t * freqs.unsqueeze(0) * (2.0 * math.pi)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def _make_encoder(in_dim: int, hidden: int, depth: int) -> nn.Sequential:
    """Build a stack of ``Linear -> GELU`` blocks of width ``hidden``."""
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(prev, hidden))
        layers.append(nn.GELU())
        prev = hidden
    return nn.Sequential(*layers)


def _canonicalize_quaternion_sign(q: torch.Tensor) -> torch.Tensor:
    """Flip the sign of any quaternion whose ``w`` component is negative.

    Uses a smooth-but-non-differentiable sign branch via ``torch.where``;
    gradients flow through the selected branch, which is sufficient for
    surrogate inversion since the optimizer never crosses the ``w == 0``
    plane in practice (training data has ``w >= 0`` after canonicalization).
    """
    sign = torch.where(
        q[..., :1] < 0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1])
    )
    return q * sign


class _FiLMLayer(nn.Module):
    """Single FiLM-modulated MLP layer: ``y = gelu(gamma * Wx + beta)``."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        h: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Apply linear + FiLM + GELU + dropout. ``h: (B,T,H), gamma/beta: (B,H)``."""
        h = self.linear(h)
        h = gamma.unsqueeze(1) * h + beta.unsqueeze(1)
        h = F.gelu(h)
        return self.dropout(h)


class SwingSurrogate(nn.Module):
    """FiLM-conditioned MLP forward surrogate.

    See ``option2_nn_surrogate/APPROACH.md`` for the full algorithmic
    description. The forward pass is naively parallel over the time
    dimension and therefore trivially differentiable -- this is the
    load-bearing property for the inversion in #029.
    """

    def __init__(self, cfg: SurrogateConfig) -> None:
        """Initialise the FiLM-MLP. See :class:`SurrogateConfig` for fields."""
        self._validate_cfg(cfg)
        super().__init__()
        self.cfg = cfg

        # Sinusoidal time embedding (fixed, registered as buffer).
        self.register_buffer(
            "time_embed", _build_time_embedding(cfg.seq_len, cfg.time_embed_dim)
        )

        # Coefficient encoder -> latent z.
        self.coeff_encoder = _make_encoder(
            cfg.coeff_dim, cfg.hidden_dim, cfg.encoder_layers
        )

        # FiLM head: emits 2 * hidden_dim per layer.
        self.film_head = nn.Linear(cfg.hidden_dim, 2 * cfg.hidden_dim * cfg.n_layers)

        # Time-embedding -> hidden projection, then n_layers of FiLM.
        self.input_proj = nn.Linear(cfg.time_embed_dim, cfg.hidden_dim)
        self.backbone = nn.ModuleList(
            [_FiLMLayer(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.n_layers)]
        )

        # Output heads.
        self.butt_head = nn.Linear(cfg.hidden_dim, 3)
        self.clubhead_head = nn.Linear(cfg.hidden_dim, 3)
        self.quat_head = nn.Linear(cfg.hidden_dim, 4)
        self.joint_head = nn.Linear(cfg.hidden_dim, cfg.n_joints)

    @staticmethod
    def _validate_cfg(cfg: SurrogateConfig) -> None:
        """Raise ``ValueError`` for any field that breaks the architecture."""
        if cfg.n_joints <= 0:
            raise ValueError(f"n_joints must be positive, got {cfg.n_joints}")
        if cfg.seq_len <= 1:
            raise ValueError(f"seq_len must be > 1, got {cfg.seq_len}")
        if cfg.hidden_dim <= 0 or cfg.n_layers <= 0:
            raise ValueError("hidden_dim and n_layers must be positive")
        if cfg.time_embed_dim <= 0 or cfg.time_embed_dim % 2 != 0:
            raise ValueError(
                f"time_embed_dim must be a positive even integer, got {cfg.time_embed_dim}"
            )

    @precondition(
        lambda self, coeffs: coeffs.ndim == 2,
        "coeffs must be a 2-D tensor of shape (B, D)",
    )
    @precondition(
        lambda self, coeffs: (
            coeffs.shape[1] == self.cfg.n_joints * self.cfg.coeffs_per_joint
        ),
        "coeffs second dim must equal n_joints * coeffs_per_joint",
    )
    @postcondition(
        lambda result: bool(torch.isfinite(result.butt).all().item()),
        "butt must be finite",
    )
    @postcondition(
        lambda result: bool(torch.isfinite(result.clubhead).all().item()),
        "clubhead must be finite",
    )
    @postcondition(
        lambda result: torch.allclose(
            result.club_quat.norm(dim=-1),
            torch.ones_like(result.club_quat[..., 0]),
            atol=1e-5,
        ),
        "club_quat must be unit-norm",
    )
    def forward(self, coeffs: torch.Tensor) -> ClubTrajectory:
        """Run the FiLM-MLP forward pass.

        Args:
            coeffs: ``(B, n_joints * coeffs_per_joint)`` float tensor.

        Returns:
            A :class:`ClubTrajectory` of tensors on the same device/dtype
            as ``coeffs`` (modulo the unit-norm projection on quaternions).
        """
        z = self.coeff_encoder(coeffs)
        film = self.film_head(z).view(-1, self.cfg.n_layers, 2, self.cfg.hidden_dim)

        b = coeffs.shape[0]
        # time_embed is a buffer; broadcast across the batch.
        time_embed: torch.Tensor = self.time_embed  # type: ignore[assignment]
        t_emb = time_embed.unsqueeze(0).expand(b, -1, -1)
        h = self.input_proj(t_emb)
        for i, layer in enumerate(self.backbone):
            gamma = 1.0 + film[:, i, 0, :]
            beta = film[:, i, 1, :]
            h = layer(h, gamma, beta)

        return self._decode_heads(h)

    def _decode_heads(self, h: torch.Tensor) -> ClubTrajectory:
        """Project hidden states to the four output heads. ``h: (B,T,H)``."""
        butt = self.butt_head(h)
        clubhead = self.clubhead_head(h)
        q_raw = self.quat_head(h)
        q_norm = q_raw.norm(dim=-1, keepdim=True).clamp_min(_QUAT_NORM_EPS)
        q_unit = _canonicalize_quaternion_sign(q_raw / q_norm)
        joint_q = self.joint_head(h)
        return ClubTrajectory(
            butt=butt, clubhead=clubhead, club_quat=q_unit, joint_q=joint_q
        )
