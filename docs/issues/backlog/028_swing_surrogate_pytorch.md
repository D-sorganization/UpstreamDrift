# Issue: Implement SwingSurrogate nn.Module and Training Loop (Option 2)

## Summary

Implement a PyTorch forward surrogate `f_θ: coefficients → kinematic trajectory`
that learns the Simscape model's input/output mapping from the random-sweep
parquet dataset. Includes the `SwingSurrogate` `nn.Module`, the training loop,
checkpointing, and tensorboard logging.

## Motivation

See `motion_matching/README.md` "Why four options in parallel" — Option 2 sells
~seconds-per-fit at the cost of a trained model. The surrogate is the
prerequisite for the differentiable inversion in #029.

## Dependencies

- #019 (`load_sweep_dataset`) — provides training data.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option2_nn_surrogate\python\swing_surrogate.py` (`SwingSurrogate`, `SwingSurrogateConfig`)
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option2_nn_surrogate\python\train_surrogate.py` (training entry point)
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option2_nn_surrogate\python\dataset_adapter.py` (turns `SweepDataset` into `(coeffs, kinematics)` pairs)
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option2\test_swing_surrogate.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option2\test_train_surrogate.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option2\test_dataset_adapter.py`

## Public API

```python
from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn as nn

@dataclass(frozen=True)
class SwingSurrogateConfig:
    n_joints: int
    n_coefficients_per_joint: int = 7
    n_timesteps: int = 300        # at 1 kHz over 0.3 s
    output_kinematic_dim: int = 12  # 3 butt + 3 clubhead + 4 quat + 2 spare
    hidden_size: int = 256
    n_layers: int = 4
    architecture: Literal["mlp", "tcn", "transformer"] = "mlp"
    dropout: float = 0.1


class SwingSurrogate(nn.Module):
    """Forward surrogate f_theta: (coeffs) -> (N_timesteps, kinematic_dim)."""

    def __init__(self, config: SwingSurrogateConfig):
        super().__init__()
        ...

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """coeffs: (B, n_joints * 7) → kinematics: (B, N_timesteps, kinematic_dim)."""


@dataclass(frozen=True)
class TrainConfig:
    dataset_path: Path
    output_dir: Path
    batch_size: int = 64
    n_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_schedule: Literal["cosine", "step", "none"] = "cosine"
    val_split: float = 0.1
    seed: int = 42
    device: Literal["cpu", "cuda", "auto"] = "auto"
    log_to_tensorboard: bool = True


def train_surrogate(config: TrainConfig) -> SwingSurrogate:
    """Train the surrogate; checkpoint best-val-loss to config.output_dir/best.pt."""
```

## Required tests (TDD)

- `test_surrogate_forward_returns_correct_output_shape`
- `test_surrogate_forward_is_differentiable_wrt_coeffs`
- `test_surrogate_mlp_architecture_has_expected_parameter_count`
- `test_surrogate_tcn_architecture_handles_variable_n_timesteps`
- `test_surrogate_transformer_architecture_uses_positional_encoding`
- `test_dataset_adapter_returns_coeffs_kinematics_pairs_per_trial`
- `test_dataset_adapter_handles_solver_status_failed_by_excluding`
- `test_dataset_adapter_normalizes_coefficients_using_bounds_from_generateRandomCoefficients`
- `test_train_loop_decreases_validation_loss_over_5_epochs_on_tiny_synthetic_dataset`
- `test_train_loop_checkpoints_best_val_loss_to_best_pt`
- `test_train_loop_resumes_from_checkpoint`
- `test_train_loop_logs_train_and_val_loss_to_tensorboard`
- `test_train_loop_seeded_run_is_reproducible_to_within_1e_minus_5_loss`
- `test_train_loop_uses_cuda_when_device_auto_and_cuda_available`

## DbC contract

Use `@precondition` / `@postcondition` decorators from
`src.shared.python.core.contracts`.

Preconditions:

- `coeffs.shape[-1] == config.n_joints * config.n_coefficients_per_joint`.
- `coeffs` is finite.

Postconditions:

- `output.shape == (B, config.n_timesteps, config.output_kinematic_dim)`.
- `output` is finite when `coeffs` is within trained-distribution bounds.

## Acceptance Criteria

- [ ] `SwingSurrogate` supports MLP, TCN, and transformer architectures.
- [ ] All listed tests pass.
- [ ] `train_surrogate` produces a checkpoint at `output_dir/best.pt`.
- [ ] Tensorboard logs include train and val loss curves.
- [ ] DbC decorators applied; postconditions enforced.
- [ ] `ruff check` and `ruff format --check` clean.
- [ ] No file exceeds 1200 lines.
- [ ] No `print()`; use `get_logger`.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `option2`, `python`, `tdd`, `dbc`

## Effort estimate

L (3-7 days). The architecture sweep is a research task; expect to iterate on
hidden size and layer count.
