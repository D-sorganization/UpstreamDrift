"""SwingSurrogate — differentiable forward model for Option 2.

Skeleton only. Bodies land in issue #028. See:
  - APPROACH.md  (architecture, loss, training procedure)
  - INTERFACES.md (full type signatures + dataclasses)
  - ASSUMPTIONS.md (validity bounds, differentiability, output schema)

Do NOT add production logic to this file outside the implementation PR.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.shared.python.core.contracts import postcondition, precondition

# NOTE: SurrogateConfig lives in config.py (issue #028) — this skeleton file
# uses a forward-declared local stub solely so the type signatures below are
# legible without needing config.py to exist yet. The real import lands at
# implementation time:
#     from .config import SurrogateConfig


@dataclass(frozen=True)
class _SurrogateConfigStub:
    """Placeholder. Replaced by `from .config import SurrogateConfig` at impl time."""

    architecture: str = "film_mlp"
    hidden_dim: int = 256
    n_layers: int = 4
    time_embed_dim: int = 64
    n_joints: int = 0
    seq_len: int = 300
    coeffs_per_joint: int = 7


@dataclass(frozen=True)
class ClubTrajectory:
    """Surrogate output. Mirrors shared/CLUB_IK_SPEC.md, batch-first.

    Tensor shapes (B = batch, N = seq_len):
        butt:     (B, N, 3)     metres, world frame
        clubhead: (B, N, 3)     metres, world frame
        q_club:   (B, N, 4)     unit quaternion [w, x, y, z], w >= 0
        q_joints: (B, N, n_j)   auxiliary joint angles (debug only)

    See ASSUMPTIONS.md § A3 — q_joints is NOT consumed by compute_cost.
    """

    butt: torch.Tensor
    clubhead: torch.Tensor
    q_club: torch.Tensor
    q_joints: torch.Tensor


class SwingSurrogate(nn.Module):
    """Differentiable forward surrogate: coefficients -> club kinematic trajectory.

    See APPROACH.md for the architecture choice (FiLM-MLP recommended for v1)
    and the full forward sketch. See INTERFACES.md for the surrounding API.
    """

    @precondition(
        lambda self, cfg: cfg.n_joints > 0,
        "SurrogateConfig.n_joints must be set (populate from dataset on load)",
    )
    @precondition(
        lambda self, cfg: cfg.architecture in {"film_mlp", "cnn1d"},
        "architecture must be film_mlp or cnn1d",
    )
    def __init__(self, cfg: _SurrogateConfigStub) -> None:
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError(
            "SwingSurrogate.__init__ — implementation lands in issue #028. "
            "See APPROACH.md § Architecture."
        )

    @precondition(
        lambda self, coeffs: coeffs.ndim == 2,
        "coeffs must be (B, D)",
    )
    @precondition(
        lambda self, coeffs: (
            coeffs.shape[1] == self.cfg.n_joints * self.cfg.coeffs_per_joint
        ),
        "coeffs second dim must equal n_joints * coeffs_per_joint",
    )
    @postcondition(
        lambda result: torch.isfinite(result.butt).all(),
        "butt must be finite",
    )
    @postcondition(
        lambda result: torch.allclose(
            result.q_club.norm(dim=-1),
            torch.ones_like(result.q_club[..., 0]),
            atol=1e-5,
        ),
        "q_club must be unit-norm",
    )
    def forward(self, coeffs: torch.Tensor) -> ClubTrajectory:
        raise NotImplementedError(
            "SwingSurrogate.forward — implementation lands in issue #028. "
            "See APPROACH.md § Architecture for the FiLM-MLP forward sketch."
        )
