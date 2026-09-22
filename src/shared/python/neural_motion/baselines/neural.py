"""Optional small-MLP dynamics baseline (NM-05)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "SmallMlpResult",
    "optional_torch_available",
    "torch_is_available",
    "train_small_mlp",
]


def torch_is_available() -> bool:
    """Return True when a working ``torch`` import succeeds."""
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def optional_torch_available() -> bool:
    """Public alias used by acceptance tests."""
    return torch_is_available()


@dataclass(frozen=True, slots=True)
class SmallMlpResult:
    train_mse: float
    val_mse: float
    seed: int
    state: dict[str, Any] | None
    skipped_reason: str | None = None


def train_small_mlp(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    seed: int,
    epochs: int = 40,
    hidden: int = 32,
    lr: float = 1e-2,
) -> SmallMlpResult:
    """Train a tiny MLP; skip cleanly when torch is absent."""
    if not torch_is_available():
        return SmallMlpResult(
            train_mse=float("nan"),
            val_mse=float("nan"),
            seed=seed,
            state=None,
            skipped_reason="torch_unavailable",
        )

    import torch
    from torch import nn

    torch.manual_seed(int(seed))
    np_rng = np.random.default_rng(int(seed))
    order = np_rng.permutation(train_x.shape[0])
    tx = torch.tensor(train_x[order], dtype=torch.float32)
    ty = torch.tensor(train_y[order], dtype=torch.float32)
    vx = torch.tensor(val_x, dtype=torch.float32)
    vy = torch.tensor(val_y, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(tx.shape[1], hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, ty.shape[1]),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        pred = model(tx)
        loss = loss_fn(pred, ty)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_mse = float(loss_fn(model(tx), ty).item())
        val_mse = float(loss_fn(model(vx), vy).item())

    return SmallMlpResult(
        train_mse=train_mse,
        val_mse=val_mse,
        seed=int(seed),
        state={k: v.detach().cpu().numpy() for k, v in model.state_dict().items()},
        skipped_reason=None,
    )
