# Issue: Implement SwingInverseCVAE Encoder/Decoder/CVAE (Option 3)

## Summary

Implement the inverse CVAE: a conditional variational autoencoder
`g_φ: kinematics → coefficients` that can sample multiple coefficient candidates
for a given target kinematic trajectory, addressing the multi-modality of the
inverse problem.

## Motivation

See `motion_matching/README.md` "Why four options in parallel" — Option 3
attacks the under-determined inverse directly with a model that **learns the
multi-modality** of valid coefficients for a given club trajectory. CVAE is the
right tool because a deterministic inverse model collapses to the mean of
multiple valid solutions.

## Dependencies

- #019 (`load_sweep_dataset`) — provides training data.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option3_inverse_nn\python\swing_inverse_cvae.py` (`KinematicEncoder`, `CoefficientDecoder`, `SwingInverseCVAE`)
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option3_inverse_nn\python\cvae_config.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option3\test_swing_inverse_cvae.py`

## Public API

```python
from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn as nn

@dataclass(frozen=True)
class CVAEConfig:
    n_joints: int
    n_coefficients_per_joint: int = 7
    n_timesteps: int = 300
    kinematic_dim: int = 12
    latent_dim: int = 16
    encoder_hidden: int = 256
    decoder_hidden: int = 256
    encoder_n_layers: int = 3
    decoder_n_layers: int = 3
    encoder_arch: Literal["mlp", "tcn", "transformer"] = "tcn"
    kl_beta: float = 1.0           # weight on KL term
    dropout: float = 0.1


class KinematicEncoder(nn.Module):
    """q_phi(z | kinematics, coefficients) — produces (mu, logvar)."""

    def __init__(self, config: CVAEConfig):
        super().__init__()

    def forward(self, kinematics: torch.Tensor, coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, logvar) of shape (B, latent_dim)."""


class CoefficientDecoder(nn.Module):
    """p_theta(coefficients | z, kinematics)."""

    def __init__(self, config: CVAEConfig):
        super().__init__()

    def forward(self, z: torch.Tensor, kinematics: torch.Tensor) -> torch.Tensor:
        """Returns coefficients of shape (B, n_joints * 7)."""


class SwingInverseCVAE(nn.Module):
    """Full CVAE module — encoder + decoder + reparametrize + sampling."""

    def __init__(self, config: CVAEConfig):
        super().__init__()

    def forward(self, kinematics: torch.Tensor, coefficients: torch.Tensor):
        """Returns (recon_coeffs, mu, logvar) for ELBO computation."""

    def sample(self, kinematics: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Sample n_samples coefficient candidates for each kinematic input.

        Returns: (B, n_samples, n_joints * 7).
        """

    def elbo_loss(self, recon_coeffs: torch.Tensor, target_coeffs: torch.Tensor,
                  mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Returns (scalar_loss, terms_dict with reconstruction, kl, total)."""
```

## Required tests (TDD)

- `test_encoder_forward_returns_mu_logvar_with_correct_shape`
- `test_decoder_forward_returns_coefficients_with_correct_shape`
- `test_cvae_forward_returns_recon_mu_logvar`
- `test_cvae_sample_returns_n_samples_per_input`
- `test_cvae_sample_diversity_increases_with_n_samples`
- `test_cvae_reparametrize_is_differentiable_wrt_mu_and_logvar`
- `test_elbo_loss_reconstruction_term_is_mean_squared_error_on_coefficients`
- `test_elbo_loss_kl_term_is_kl_divergence_to_unit_gaussian`
- `test_elbo_loss_kl_beta_scales_kl_term_proportionally`
- `test_encoder_tcn_arch_handles_variable_n_timesteps`
- `test_decoder_output_within_normalized_coefficient_bounds_after_tanh_or_clamp`
- `test_cvae_forward_is_differentiable_end_to_end`
- `test_cvae_seed_reproducibility_for_sample_method`

## DbC contract

Preconditions:

- `kinematics.shape == (B, n_timesteps, kinematic_dim)`.
- `coefficients.shape == (B, n_joints * 7)`.
- All inputs finite.

Postconditions:

- Output shapes match those documented in each method's signature.
- Sampled coefficients normalized to bounded distribution (tanh or clamp).
- ELBO loss is non-negative.

## Acceptance Criteria

- [ ] `SwingInverseCVAE` supports MLP, TCN, and transformer encoder architectures.
- [ ] All listed tests pass.
- [ ] DbC decorators applied; postconditions enforced.
- [ ] `ruff check` and `ruff format --check` clean.
- [ ] No file exceeds 1200 lines.
- [ ] No `print()`; use `get_logger`.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `option3`, `python`, `tdd`, `dbc`

## Effort estimate

L (3-7 days). CVAE training has well-known stability footguns (KL collapse, mode
collapse) that take time to diagnose.
