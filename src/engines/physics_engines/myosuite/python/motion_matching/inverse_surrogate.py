import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass(frozen=True)
class InverseSurrogateConfig:
    """Architectural configuration for the MyoSuite inverse surrogate."""

    n_joints: int
    n_muscles: int
    seq_len: int
    hidden_dim: int = 512
    n_layers: int = 4
    dropout: float = 0.1


class MyoSuiteInverseSurrogate(nn.Module):
    """Inverse muscle surrogate mapping joint kinematics to muscle activations.

    Outputs muscle activations in the strictly bounded range [0, 1] suitable
    for MyoSuite actuation.
    """

    def __init__(self, cfg: InverseSurrogateConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # We take joint_q and joint_v (kinematic state) as input per timestep.
        in_dim = cfg.n_joints * 2

        layers = []
        prev = in_dim
        for _ in range(cfg.n_layers):
            layers.append(nn.Linear(prev, cfg.hidden_dim))
            layers.append(nn.LayerNorm(cfg.hidden_dim))
            layers.append(nn.GELU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            prev = cfg.hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(cfg.hidden_dim, cfg.n_muscles)

    def forward(self, joint_q: torch.Tensor, joint_v: torch.Tensor) -> torch.Tensor:
        """Predict muscle activations from joint kinematics.

        Args:
            joint_q: (B, T, n_joints) joint positions
            joint_v: (B, T, n_joints) joint velocities

        Returns:
            activations: (B, T, n_muscles) in the range [0, 1]
        """
        x = torch.cat([joint_q, joint_v], dim=-1)
        h = self.backbone(x)
        # Muscle activations are strictly bounded [0, 1]
        return torch.sigmoid(self.head(h))
