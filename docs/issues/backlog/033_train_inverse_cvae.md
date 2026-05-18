# Issue: Implement Training Pipeline for SwingInverseCVAE (Option 3)

## Summary

Implement the training pipeline for the inverse CVAE: data loading from
`SweepDataset`, KL annealing schedule, ELBO optimization, validation
diversity metrics, checkpointing, and tensorboard logging.

## Motivation

See `motion_matching/README.md` "Why four options in parallel" — Option 3
delivers ms-per-fit inference once trained. The training pipeline must guard
against KL collapse (a well-known CVAE failure mode) by annealing β and
monitoring per-batch sample diversity.

## Dependencies

- #019 (`load_sweep_dataset`) — provides training data.
- #032 (`SwingInverseCVAE`) — the model trained here.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option3_inverse_nn\python\train_cvae.py`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option3_inverse_nn\python\dataset_adapter.py` (turns `SweepDataset` into `(kinematics, coefficients)` pairs — sharable with #028's adapter)
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option3_inverse_nn\python\kl_annealing.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option3\test_train_cvae.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option3\test_kl_annealing.py`

## Public API

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass(frozen=True)
class CVAETrainConfig:
    dataset_path: Path
    output_dir: Path
    cvae_config: CVAEConfig
    batch_size: int = 64
    n_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_split: float = 0.1
    seed: int = 42
    device: str = "auto"
    kl_anneal_strategy: Literal["linear", "cyclic", "constant"] = "cyclic"
    kl_anneal_n_cycles: int = 4
    kl_anneal_max_beta: float = 1.0
    log_to_tensorboard: bool = True


def train_cvae(config: CVAETrainConfig) -> SwingInverseCVAE:
    """Train inverse CVAE; checkpoint best-val-elbo to config.output_dir/best.pt.
       Logs reconstruction loss, KL loss, validation diversity, and current beta."""


def kl_anneal_schedule(strategy: str, epoch: int, n_epochs: int,
                       n_cycles: int, max_beta: float) -> float:
    """Returns current beta value for the given epoch under the chosen strategy."""
```

## Required tests (TDD)

- `test_train_loop_decreases_validation_elbo_over_5_epochs_on_tiny_dataset`
- `test_train_loop_checkpoints_best_val_elbo_to_best_pt`
- `test_train_loop_logs_reconstruction_kl_diversity_to_tensorboard`
- `test_train_loop_seeded_run_is_reproducible_to_within_1e_minus_5_loss`
- `test_kl_anneal_linear_increases_beta_from_zero_to_max_over_training`
- `test_kl_anneal_cyclic_oscillates_between_zero_and_max_n_cycles_times`
- `test_kl_anneal_constant_returns_max_beta_at_every_epoch`
- `test_dataset_adapter_returns_kinematics_coefficients_pairs`
- `test_dataset_adapter_normalizes_coefficients_using_bounds`
- `test_diversity_metric_per_batch_is_pairwise_l2_distance_among_samples`
- `test_train_loop_aborts_with_clear_error_if_kl_collapses_below_threshold`

## DbC contract

Preconditions:

- `config.dataset_path` exists and contains a valid `SweepDataset`.
- `config.cvae_config` is a valid `CVAEConfig`.
- `config.batch_size >= 1`.

Postconditions:

- Returned model is a `SwingInverseCVAE` instance.
- Best checkpoint exists at `config.output_dir / "best.pt"`.
- Validation diversity metric is non-zero at end of training (no KL collapse).

## Acceptance Criteria

- [ ] Training loop runs end-to-end on the synthetic dataset fixture.
- [ ] All listed tests pass.
- [ ] KL annealing strategies all implemented and verified.
- [ ] DbC decorators applied; postconditions enforced.
- [ ] `ruff check` and `ruff format --check` clean.
- [ ] No file exceeds 1200 lines.
- [ ] No `print()`; use `get_logger`.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `option3`, `python`, `tdd`, `dbc`

## Effort estimate

L (3-7 days). CVAE training is research-flavoured; expect to iterate on
β-annealing schedule and architecture choices to avoid mode collapse.
