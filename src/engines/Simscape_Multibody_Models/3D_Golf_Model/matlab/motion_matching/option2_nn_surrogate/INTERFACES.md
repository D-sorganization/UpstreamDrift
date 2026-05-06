# Option 2 — Interfaces

Public Python and MATLAB function signatures for Option 2. Decorators are from `src.shared.python.core.contracts` per [shared/CODING_STANDARDS.md § DbC](../shared/CODING_STANDARDS.md#dbc-design-by-contract). Bodies are out of scope here — agents pick these up via issues #028–#031.

## Module layout

```
option2_nn_surrogate/
├── surrogate.py           # SwingSurrogate, ClubTrajectory, SurrogateConfig
├── train.py               # train_surrogate, TrainConfig, TrainedSurrogate
├── invert.py              # fit_swing_via_surrogate, FitResult, InvertOptions
├── validate.py            # validate_against_simscape, ValidationReport
├── dataset.py             # SweepDatasetTorch (PyTorch wrapper), normalization stats
├── config.py              # shared dataclasses
├── fit_swing_surrogate.m  # MATLAB shim → pyrunfile
└── tests/
```

A skeleton of `surrogate.py` ships with the docs (single short file with type signatures, docstrings, and `raise NotImplementedError`).

## Dataclasses (in `config.py`)

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class SurrogateConfig:
    architecture: str = "film_mlp"        # "film_mlp" | "cnn1d"
    hidden_dim: int = 256
    n_layers: int = 4
    time_embed_dim: int = 64
    n_joints: int = 0                     # populated from dataset on construction; 0 sentinel
    seq_len: int = 300                    # N timesteps
    coeffs_per_joint: int = 7             # A..G

@dataclass(frozen=True)
class TrainConfig:
    dataset_path: Path
    output_dir: Path
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    batch_size: int = 32
    max_steps: int = 50_000
    warmup_steps: int = 500
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    eval_every: int = 1000
    early_stop_patience: int = 5
    early_stop_rmse_m: float = 5e-3
    w_butt: float = 1.0
    w_clubhead: float = 1.0
    w_quat: float = 0.1
    w_aux: float = 0.1
    seed: int = 0xC0FFEE
    use_amp: bool = True

@dataclass(frozen=True)
class InvertOptions:
    n_restarts: int = 8
    max_iters: int = 200
    invert_lr: float = 1e-2
    early_stop_loss: float = 1e-6
    regularizer: str = "none"             # "none" | "coeff_l2"
    lambda_: float = 0.0
    seed: int | None = None
```

## `surrogate.py`

```python
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
from src.shared.python.core.contracts import precondition, postcondition

from .config import SurrogateConfig


@dataclass(frozen=True)
class ClubTrajectory:
    """Surrogate output. Mirrors shared/CLUB_IK_SPEC.md target schema, batch-first.

    Tensor shapes (B = batch, N = seq_len):
        butt:     (B, N, 3)     metres, world frame
        clubhead: (B, N, 3)     metres, world frame
        q_club:   (B, N, 4)     unit quaternion [w, x, y, z], w >= 0
        q_joints: (B, N, n_j)   auxiliary joint angles (debug only)
    """
    butt: torch.Tensor
    clubhead: torch.Tensor
    q_club: torch.Tensor
    q_joints: torch.Tensor


class SwingSurrogate(nn.Module):
    """Differentiable forward surrogate: coefficients -> club kinematic trajectory.

    See APPROACH.md for architecture details. v1 default is FiLM-MLP.
    """

    @precondition(
        lambda self, cfg: cfg.n_joints > 0,
        "SurrogateConfig.n_joints must be set (populate from dataset on load)",
    )
    @precondition(
        lambda self, cfg: cfg.architecture in {"film_mlp", "cnn1d"},
        "architecture must be film_mlp or cnn1d",
    )
    def __init__(self, cfg: SurrogateConfig) -> None: ...

    @precondition(
        lambda self, coeffs: coeffs.ndim == 2,
        "coeffs must be (B, D)",
    )
    @precondition(
        lambda self, coeffs: coeffs.shape[1] == self.cfg.n_joints * self.cfg.coeffs_per_joint,
        "coeffs second dim must equal n_joints * coeffs_per_joint",
    )
    @postcondition(
        lambda self_unused_args_kwargs, result: torch.isfinite(result.butt).all(),
        "butt must be finite",
    )
    @postcondition(
        lambda self_unused_args_kwargs, result: torch.allclose(
            result.q_club.norm(dim=-1), torch.ones_like(result.q_club[..., 0]), atol=1e-5
        ),
        "q_club must be unit-norm",
    )
    def forward(self, coeffs: torch.Tensor) -> ClubTrajectory: ...
```

## `train.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from src.shared.python.core.contracts import precondition, postcondition
from .config import TrainConfig
from .surrogate import SwingSurrogate
from .dataset import SweepDatasetTorch, NormalizationStats


@dataclass(frozen=True)
class TrainedSurrogate:
    """Bundle of a trained surrogate and everything needed to use it."""
    model: SwingSurrogate
    norm_stats: NormalizationStats
    train_config: TrainConfig
    final_val_rmse_m: float                # best held-out clubhead RMSE
    checkpoint_path: Path                  # absolute path to best.pt
    git_commit: str
    seed: int


@precondition(
    lambda dataset, config: dataset is not None,
    "dataset must be provided",
)
@precondition(
    lambda dataset, config: config.batch_size > 0,
    "batch_size must be positive",
)
@postcondition(
    lambda result: result.final_val_rmse_m >= 0,
    "final RMSE must be non-negative",
)
@postcondition(
    lambda result: result.checkpoint_path.exists(),
    "checkpoint must be persisted",
)
def train_surrogate(
    dataset: SweepDatasetTorch,
    config: TrainConfig,
) -> TrainedSurrogate:
    """Train SwingSurrogate end-to-end per APPROACH.md.

    Returns a TrainedSurrogate that downstream code (invert, validate, the
    MATLAB shim) loads via load_trained_surrogate(checkpoint_path).
    """
    raise NotImplementedError


def load_trained_surrogate(checkpoint_path: Path) -> TrainedSurrogate: ...
```

## `invert.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import torch

from src.shared.python.core.contracts import precondition, postcondition
from src.shared.python.motion_matching.targets import ClubTarget   # from CLUB_IK_SPEC.md
from .config import InvertOptions
from .train import TrainedSurrogate


@dataclass(frozen=True)
class FitResult:
    """Result of a surrogate-based fit. Round-trip RMSE is filled by validate.py."""
    coefficients: np.ndarray              # (n_joints * 7,) float64
    surrogate_loss: float                 # final loss against target, surrogate-predicted
    surrogate_rmse_m: float               # RMSE in metres on (butt, clubhead)
    simscape_rmse_m: float | None         # populated by validate_against_simscape
    n_restarts_used: int
    iters_per_restart: int
    duration_s: float
    target_hash: str
    surrogate_checkpoint_id: str
    git_commit: str
    timestamp_utc: datetime
    invert_options: InvertOptions


@precondition(
    lambda target, surrogate, opts: target.butt.shape[0] == surrogate.train_config.surrogate.seq_len,
    "target and surrogate must share seq_len",
)
@precondition(
    lambda target, surrogate, opts: opts.n_restarts >= 1,
    "n_restarts must be >= 1",
)
@postcondition(
    lambda result: np.isfinite(result.coefficients).all(),
    "fitted coefficients must be finite",
)
@postcondition(
    lambda result: result.surrogate_rmse_m >= 0,
    "surrogate RMSE must be non-negative",
)
def fit_swing_via_surrogate(
    target: ClubTarget,
    surrogate: TrainedSurrogate,
    opts: InvertOptions,
) -> FitResult:
    """Fit coefficients to a measured target via Adam on a frozen surrogate.

    See APPROACH.md § Inversion. K-restart, bound-projected, gradient-based.
    Bounds come from generateRandomCoefficients.m via the dataset metadata.
    """
    raise NotImplementedError
```

## `validate.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

from src.shared.python.core.contracts import precondition, postcondition
from .invert import FitResult


SimulateFn = Callable[[np.ndarray], dict]
"""sim_fn(coefficients) -> simscape output dict (matches simulate_with_coefficients.m, issue #018)."""


@dataclass(frozen=True)
class ValidationReport:
    fit_result: FitResult
    simscape_rmse_m: float
    surrogate_rmse_m: float
    extrapolation_ratio: float            # simscape_rmse / surrogate_rmse
    is_extrapolation: bool                # ratio > extrapolation_factor
    extrapolation_factor: float           # threshold used (default 2.0)
    flag: str                             # "ok" | "extrapolation" | "simscape_failed"


@precondition(
    lambda result, sim_fn, **kw: callable(sim_fn),
    "sim_fn must be callable",
)
@postcondition(
    lambda report: report.simscape_rmse_m >= 0,
    "simscape RMSE must be non-negative",
)
def validate_against_simscape(
    result: FitResult,
    sim_fn: SimulateFn,
    *,
    extrapolation_factor: float = 2.0,
) -> ValidationReport:
    """Round-trip the fitted coefficients through Simscape and compare.

    See APPROACH.md § Validation. The simscape RMSE — not the surrogate RMSE — is
    what gets reported to the leaderboard.
    """
    raise NotImplementedError
```

## `dataset.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.shared.python.core.contracts import precondition, postcondition


@dataclass(frozen=True)
class NormalizationStats:
    """Per-feature normalization stats. Computed on the train split only."""
    coeffs_mean: np.ndarray          # (D,)
    coeffs_std: np.ndarray           # (D,)
    butt_mean: np.ndarray            # (3,)
    butt_std: np.ndarray             # (3,)
    clubhead_mean: np.ndarray        # (3,)
    clubhead_std: np.ndarray         # (3,)
    # Quaternions are not normalized (already unit-norm).


class SweepDatasetTorch(Dataset):
    """PyTorch wrapper over the SweepDataset (issue #019).

    Each sample is one trial: (coeffs, club_trajectory_target).
    """
    @precondition(
        lambda self, sweep_path, split, norm_stats=None: split in {"train", "val", "test"},
        "split must be train|val|test",
    )
    def __init__(
        self,
        sweep_path: Path,
        split: str,
        norm_stats: NormalizationStats | None = None,
    ) -> None: ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]: ...
```

## MATLAB shim — `fit_swing_surrogate.m`

```matlab
function result = fit_swing_surrogate(target, options)
%FIT_SWING_SURROGATE  Fit polynomial coefficients to a measured swing using
%   the Option 2 differentiable forward surrogate.
%
%   result = FIT_SWING_SURROGATE(TARGET, OPTIONS) calls the Python entry-point
%   `option2_nn_surrogate.entry.fit_from_matlab` via `pyrunfile`. The Python
%   side (a) loads a trained surrogate checkpoint, (b) runs Adam on the input
%   coefficients per APPROACH.md, (c) optionally polishes via fit_swing_fmincon
%   from option1 if `options.polish == true`.
%
%   Inputs
%   ------
%   TARGET   1x1 struct as in shared/CLUB_IK_SPEC.md (.time, .butt, .clubhead,
%            .club_quat, .impact_idx, .source).
%   OPTIONS  1x1 struct, fields:
%       .checkpoint_path  string (required) — absolute path to best.pt
%       .n_restarts       uint32 (default 8)
%       .max_iters        uint32 (default 200)
%       .invert_lr        double (default 1e-2)
%       .polish           logical (default false) — chain into fit_swing_fmincon
%       .seed             uint64 (default 0)
%       .verbosity        string ("Silent"|"Normal"|"Verbose"|"Debug")
%
%   Output
%   ------
%   RESULT   1x1 struct conforming to shared/CODING_STANDARDS.md § Provenance.
%
%   Preconditions:
%     - TARGET satisfies CLUB_IK_SPEC.md output schema
%     - OPTIONS.checkpoint_path exists
%     - The Python environment is active in pyenv() and has torch installed
%
%   Postconditions:
%     - result.final_rmse_m is a non-negative scalar (Simscape round-trip RMSE)
%     - result.solver == "nn-surrogate" or "nn-surrogate+fmincon" if polish
%     - result.target_hash matches sha256(target)
%
%   See also: fit_swing_fmincon, simulate_with_coefficients
    arguments
        target  (1,1) struct {mustHaveFields(target, ["time","butt","clubhead","club_quat","impact_idx","source"])}
        options (1,1) struct {mustHaveFields(options, ["checkpoint_path"])}
    end
    error("NotImplemented:option2_matlab_shim", ...
        "Implementation lands in issue #031.");
end
```

The Python entry-point that the shim calls is `option2_nn_surrogate.entry.fit_from_matlab`; it is not part of the public Python API but is documented in [RUNBOOK.md](RUNBOOK.md). The function takes serializable Python types (numpy arrays, dicts) so MATLAB-Python marshalling is straightforward via `pyrunfile`.

## Cross-references

- All dataclass-typed inputs/outputs trace back to [shared/CLUB_IK_SPEC.md](../shared/CLUB_IK_SPEC.md), [shared/COST_FUNCTION_SPEC.md](../shared/COST_FUNCTION_SPEC.md), and [shared/DATASET_SCHEMA.md](../shared/DATASET_SCHEMA.md).
- Result struct schema follows [shared/CODING_STANDARDS.md § Provenance](../shared/CODING_STANDARDS.md#provenance-and-reproducibility).
- Tests for these contracts are in [TESTING.md](TESTING.md).
