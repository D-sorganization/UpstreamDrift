# Option 3 — Interfaces

> Python signatures with explicit `@precondition` / `@postcondition` decorators per [shared/CODING_STANDARDS.md §DbC](../shared/CODING_STANDARDS.md#dbc-design-by-contract). Implementations land under Issues [#032–#035](README.md#github-issues).

## Imports

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn

from src.shared.python.core.contracts import precondition, postcondition, invariant

# Reused from Option 2 (do NOT duplicate — see README.md "Dependency on Option 2").
from src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.motion_matching.option2_nn_surrogate.data import (
    SweepDataset,
    load_sweep_dataset,
    NormalizationStats,
)

# Canonical target schema, shared.
from src.shared.python.motion_matching.club_target import ClubTarget
```

> **Note.** The actual import path will resolve via `src/shared/python/motion_matching/__init__.py` once promoted; the long path above is the source of truth before promotion.

## Configuration

```python
@dataclass(frozen=True)
class CVAEConfig:
    """Hyperparameters for the inverse CVAE. See APPROACH.md §Architecture."""
    n_joints: int                     # = len(joint_names) from the dataset
    seq_len: int = 300                # 0.3s @ 1 kHz
    d_model: int = 256
    encoder_layers: int = 4
    encoder_heads: int = 8
    d_ctx: int = 256
    d_z: int = 32
    decoder_hidden: tuple[int, ...] = (512, 512)
    coef_bounds: tuple[float, ...] = (1000, 1000, 500, 500, 100, 100, 25)  # A..G
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    """Training-time hyperparameters."""
    batch_size: int = 64
    epochs: int = 200
    lr: float = 3e-4
    weight_decay: float = 1e-5
    kl_warmup_epochs: int = 20        # β: 0 → 1 over this many epochs
    lambda_theta: float = 1.0
    lambda_work: float = 1e-3
    work_estimator: str = "surrogate" # "surrogate" | "closed_form" | "simscape"
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 0
    device: str = "cuda"
    checkpoint_dir: Path = Path("models/")
```

## The CVAE module

```python
class SwingInverseCVAE(nn.Module):
    """Conditional VAE: club_kinematic_trajectory → torque_coefficients.

    See APPROACH.md for architecture rationale and loss definition.

    Invariants
    ----------
    - decoder output is bounded by `config.coef_bounds` via per-coefficient
      scaled tanh; clipping is never required.
    - encoder is causal-free (consumes the full sequence) and produces a
      single context vector h_x of size `config.d_ctx`.
    """

    def __init__(self, config: CVAEConfig) -> None:
        super().__init__()
        # Bodies live in inverse_cvae.py (skeleton only at scaffold time).
        ...

    @precondition(lambda self, x: x.dim() == 3,
                  "x must be (batch, seq_len, 12)")
    @precondition(lambda self, x: x.shape[-1] == 12,
                  "x must encode (butt[3], clubhead[3], quat[4], optional pad → 12)")
    @postcondition(lambda result: result.dim() == 2,
                   "encode returns (batch, d_ctx)")
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a club kinematic sequence into a context vector h_x."""
        ...

    @precondition(lambda self, z, h_x: z.dim() == 2 and h_x.dim() == 2,
                  "z and h_x are (batch, *)")
    @precondition(lambda self, z, h_x: z.shape[0] == h_x.shape[0],
                  "batch sizes must match")
    @postcondition(lambda result: torch.isfinite(result).all().item(),
                   "decoder output must be finite")
    def decode(self, z: torch.Tensor, h_x: torch.Tensor) -> torch.Tensor:
        """Decode a latent z and context h_x into a coefficient vector θ̂.

        Returns
        -------
        torch.Tensor of shape (batch, n_joints * 7), bounded by
        `config.coef_bounds` via scaled tanh.
        """
        ...

    @precondition(lambda self, x, theta: x.shape[0] == theta.shape[0],
                  "batch sizes must match")
    @postcondition(lambda result: set(result.keys()) >= {"theta_hat", "z", "mu", "log_sigma", "h_x"},
                   "forward must populate the full diagnostic dict")
    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> dict[str, torch.Tensor]:
        """Training-time forward pass.

        Encodes x, samples z from q(z | x, θ_truth) via reparameterization,
        decodes, and returns everything needed for the loss in
        APPROACH.md §Loss function.

        Returns dict with keys:
            theta_hat : (batch, n_joints*7)
            z         : (batch, d_z)
            mu        : (batch, d_z)         posterior mean
            log_sigma : (batch, d_z)         posterior log-stddev
            h_x       : (batch, d_ctx)
        """
        ...

    @precondition(lambda self, x, n_samples: n_samples >= 1,
                  "n_samples must be at least 1")
    @postcondition(lambda result: result.dim() == 3,
                   "returns (n_samples, batch, n_joints*7)")
    def sample_coefficients(
        self,
        x: torch.Tensor,
        n_samples: int = 32,
    ) -> torch.Tensor:
        """Inference-time sampling. Draws z ~ N(0, I), decodes, stacks samples.

        Each sample is one mode of the posterior over coefficients given x.
        See APPROACH.md §Inference for the rejection-sampling protocol.
        """
        ...
```

## Trained-model handle

```python
@dataclass(frozen=True)
class TrainedInverseCVAE:
    """Frozen handle to a trained CVAE plus the artifacts needed to use it."""
    model: SwingInverseCVAE          # in eval mode
    config: CVAEConfig
    train_config: TrainConfig
    norm_stats: NormalizationStats   # reused from Option 2
    joint_names: list[str]
    git_commit: str
    checkpoint_path: Path
    train_metrics: dict[str, float]  # final-epoch losses, KL, val rmse, ...
```

## Training entry point

```python
@precondition(lambda dataset, config: len(dataset.trials) >= 100,
              "training requires at least 100 trials; see ASSUMPTIONS.md §A6")
@postcondition(lambda result: result.train_metrics.get("val_round_trip_rmse_m") is not None,
               "training must record validation round-trip RMSE")
@postcondition(lambda result: result.checkpoint_path.exists(),
               "training must persist a checkpoint")
def train_inverse_cvae(
    dataset: SweepDataset,
    config: CVAEConfig,
    train_config: TrainConfig,
    *,
    sim_fn: Optional[Callable[[np.ndarray], "SimResult"]] = None,
    work_surrogate: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    log_dir: Optional[Path] = None,
) -> TrainedInverseCVAE:
    """Train the inverse CVAE on the random-sweep dataset.

    Parameters
    ----------
    dataset
        Loaded by Option 2's `load_sweep_dataset`. Option 3 does not own the
        loader; see DATASET_SCHEMA.md.
    config, train_config
        Hyperparameters. See APPROACH.md.
    sim_fn
        Optional Simscape callback. If provided, run round-trip validation on
        a small fixed subset every N epochs and log the RMSE distribution.
    work_surrogate
        Optional differentiable estimator of total mechanical work for the
        work-regularization term. If None, falls back to closed-form
        approximation per APPROACH.md §Work regularization.
    log_dir
        TensorBoard / training-curve sink. None disables.

    Returns
    -------
    TrainedInverseCVAE handle, persisted at `train_config.checkpoint_dir`.
    """
    ...
```

## Inference entry point

```python
@dataclass(frozen=True)
class InverseFitResult:
    """One Option-3 prediction. Mirrors the shape of Option 1/2 results
    so the shared visualization helpers consume it without glue code."""
    coefficients: np.ndarray            # (n_joints * 7,) float64; the chosen sample
    samples: np.ndarray                 # (n_samples, n_joints * 7) all draws
    final_rmse_m: float                 # round-trip clubhead RMSE; nan if not validated
    final_total_work_J: float
    validated: bool                     # was sim_fn invoked?
    threshold_met: bool                 # round-trip rmse < threshold?
    n_samples: int
    accepted_index: int                 # which row of `samples` was returned
    target_hash: str                    # sha256 of the ClubTarget
    solver: str = "inverse-cvae"
    solver_options: dict[str, object] = None
    git_commit: str = ""
    duration_s: float = 0.0
    timestamp_utc: str = ""


@precondition(lambda target, model, n_samples, validate, sim_fn:
              (not validate) or (sim_fn is not None),
              "validate=True requires a sim_fn callback")
@precondition(lambda target, model, n_samples, validate, sim_fn: n_samples >= 1,
              "n_samples must be at least 1")
@postcondition(lambda result: result.coefficients.shape[0] == result.samples.shape[1],
               "coefficient row matches the sample width")
@postcondition(lambda result: (not result.validated) or (result.final_rmse_m >= 0.0),
               "RMSE is non-negative when validated")
def predict_coefficients(
    target: ClubTarget,
    model: TrainedInverseCVAE,
    *,
    n_samples: int = 32,
    validate: bool = True,
    sim_fn: Optional[Callable[[np.ndarray], "SimResult"]] = None,
    rmse_threshold_m: float = 0.010,
    surrogate_prefilter: Optional[Callable[[np.ndarray], float]] = None,
) -> InverseFitResult:
    """Run inference and (optionally) round-trip validation.

    Parameters
    ----------
    target
        ClubTarget from CLUB_IK_SPEC.md.
    model
        TrainedInverseCVAE handle.
    n_samples
        Number of latent samples to draw. See APPROACH.md §Rejection-sampling
        budget for guidance (1 / 32 / 128).
    validate
        If True, every sample is round-tripped through `sim_fn`; the lowest
        RMSE wins. If False, the first sample is returned and `validated=False`.
    sim_fn
        Forward simulator: θ → SimResult (uses Option 4's adapter when
        available; else MATLAB Engine). Required when validate=True.
    rmse_threshold_m
        Acceptance threshold for `threshold_met`. Default 10 mm per
        ASSUMPTIONS.md §A1.
    surrogate_prefilter
        Optional Option-2 surrogate that scores candidates cheaply; only the
        top-K go to the real simulator. See APPROACH.md §Inference.
    """
    ...
```

## Round-trip validator

```python
@dataclass(frozen=True)
class ValidationReport:
    target_hash: str
    n_samples: int
    rmse_per_sample_m: np.ndarray        # (n_samples,) float64
    work_per_sample_J: np.ndarray        # (n_samples,) float64
    accepted_index: int
    threshold_met: bool
    threshold_m: float
    sim_wall_s: float


@precondition(lambda result, sim_fn: result.samples.size > 0,
              "result must contain at least one sample")
@postcondition(lambda report: report.rmse_per_sample_m.shape[0] == report.n_samples,
               "one RMSE per sample")
@postcondition(lambda report: 0 <= report.accepted_index < report.n_samples,
               "accepted_index in range")
def validate_round_trip(
    result: InverseFitResult,
    sim_fn: Callable[[np.ndarray], "SimResult"],
    *,
    threshold_m: float = 0.010,
) -> ValidationReport:
    """Re-run all samples through Simscape, score against the original target,
    and report which sample wins. Cheap to run after the fact (e.g. on
    persisted results) for offline analysis.
    """
    ...
```

## Skeleton location

The class signature lives in `inverse_cvae.py` in this folder. **No bodies** — methods raise `NotImplementedError`. Issue #032 implements them.
