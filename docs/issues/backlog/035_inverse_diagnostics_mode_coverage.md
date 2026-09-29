# Issue: Implement Inverse Diagnostics — UMAP Latent + Diversity Plots (Option 3)

## Summary

Implement the diagnostic plots for the trained CVAE: a UMAP/t-SNE projection
of the latent space coloured by swing characteristics, and per-target sample
diversity plots that detect mode collapse.

## Motivation

See `motion_matching/shared/VISUALIZATION_SPEC.md` §"Latent space projection"
and "Round-trip residuals". Mode collapse is the failure mode for CVAEs;
without these diagnostics it is invisible until production.

## Dependencies

- #019 (`load_sweep_dataset`).
- #032 (`SwingInverseCVAE`).
- #034 (`predict_with_rejection_sampling`) — provides per-target candidate sets.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option3_inverse_nn\python\inverse_diagnostics.py`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option3_inverse_nn\python\latent_projection.py` (UMAP/t-SNE wrapper)
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option3_inverse_nn\python\diversity_metrics.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option3\test_inverse_diagnostics.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option3\test_latent_projection.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option3\test_diversity_metrics.py`

## Public API

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass(frozen=True)
class DiagnosticsConfig:
    cvae_checkpoint: Path
    dataset_path: Path
    n_validation_targets: int = 50
    n_samples_per_target: int = 32
    projection: Literal["umap", "tsne", "pca"] = "umap"
    output_dir: Path = Path("results/option3_diagnostics")
    seed: int = 42


@dataclass(frozen=True)
class DiagnosticsReport:
    latent_projection_path: Path        # 2D scatter colored by clubhead-speed-max
    diversity_per_target_path: Path     # per-target boxplot of pairwise candidate L2
    coverage_path: Path                  # candidate-cost histogram per target
    mean_diversity: float
    median_diversity: float
    fraction_collapsed_targets: float    # diversity below threshold


def run_inverse_diagnostics(config: DiagnosticsConfig) -> DiagnosticsReport:
    """Generate latent + diversity + coverage diagnostics for the trained CVAE."""


def project_latent(model: SwingInverseCVAE, dataset: SweepDataset,
                   strategy: str, seed: int) -> np.ndarray:
    """Returns (n_trials, 2) projection of the encoder's mu vectors."""


def diversity_metric(samples: np.ndarray) -> float:
    """Mean pairwise L2 distance among (n_samples, n_joints*7) candidate vectors."""
```

## Required tests (TDD)

- `test_run_diagnostics_writes_three_pngs_to_output_dir`
- `test_latent_projection_returns_n_trials_by_2_array_for_umap_strategy`
- `test_latent_projection_returns_n_trials_by_2_array_for_tsne_strategy`
- `test_latent_projection_returns_n_trials_by_2_array_for_pca_strategy`
- `test_latent_projection_seed_reproducibility_for_umap`
- `test_diversity_metric_zero_for_all_identical_samples`
- `test_diversity_metric_increases_with_more_diverse_samples`
- `test_diversity_metric_returns_finite_value_for_n_samples_one_pair`
- `test_diagnostics_flags_mode_collapse_when_mean_diversity_below_threshold`
- `test_coverage_plot_renders_one_subplot_per_target_with_J_distribution`
- `test_diagnostics_provenance_records_cvae_checkpoint_sha`

## DbC contract

Preconditions:

- `config.cvae_checkpoint` exists.
- `config.dataset_path` exists.
- `config.n_samples_per_target >= 2` (need pairs for diversity).

Postconditions:

- All three output PNGs exist on disk.
- `report.fraction_collapsed_targets in [0, 1]`.
- `report.mean_diversity >= 0`.

## Acceptance Criteria

- [ ] `run_inverse_diagnostics` produces all three plots end-to-end.
- [ ] All listed tests pass.
- [ ] UMAP, t-SNE, and PCA strategies all implemented.
- [ ] Mode-collapse detection threshold documented and configurable.
- [ ] DbC decorators applied; postconditions enforced.
- [ ] `ruff check` and `ruff format --check` clean.
- [ ] No file exceeds 1200 lines.
- [ ] No `print()`; use `get_logger`.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `option3`, `python`, `viz`, `tdd`

## Effort estimate

M (1-3 days). UMAP and t-SNE are off-the-shelf; the time goes into wiring the
CVAE encoder mu extraction and per-target sampling loops.
