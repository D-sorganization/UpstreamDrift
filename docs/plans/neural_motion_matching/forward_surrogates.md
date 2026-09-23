# NM-07 Forward Surrogates and Physics-Structured Alternatives

Governing issue: #10622 (epic #10603).

Schema:
`neural-surrogate-comparison/1.0.0`

## What Landed

Primary package under `src/shared/python/neural_motion/surrogates/`:

- `SurrogateCandidateKind`, `SurrogateComparisonConfig`, `SurrogateAblationResult`, `SurrogateComparisonReport` (schema `neural-surrogate-comparison/1.0.0`).
- `PhysicsStructuredSurrogate`: combines an analytical rigid polynomial prior with bounded residual dynamics (`max_residual_norm`), tracking trust regions and flagging contact boundary transitions.
- `compare_surrogates_and_alternatives`: comparative benchmark across:
  - Unconstrained forward surrogate inversion: flagged for adversarial exploitation risk on native mechanics.
  - Hybrid polish: initial neural proposal refined by native/analytical physics steps.
  - Physics-structured residual: rigid analytical prior + bounded neural correction.
  - Masked proposal (NM-06 baseline).
  - Diffusion fallback: ablation demonstrating unacceptable latency (>500 ms) and sample inefficiency without multimodal ambiguity.

Validation and metrics under `src/shared/python/motion_matching/surrogate/validate.py`:

- `resample_to_timegrid`: real-clock timegrid resampling preserving physical velocity/acceleration profiles over non-uniform native timestamps.
- `quaternion_geodesic_error_rad`: antipodal double-cover sign-invariant SO(3) angular distance `min(||q - q'||, ||q + q'||)`.
- `check_trust_region`: state-space bounds validation preventing extrapolation into non-physical regimes.
- `compute_directional_derivative` & `compare_gradient_fidelity`: detects adversarial gradients where surrogate loss decreases while diverging from native physics direction.
- `check_contact_boundary_failure`: rejects smooth rigid models when ground impact or collision discontinuities occur.

Discovery link:

- `src/shared/python/motion_matching/surrogate/nm07_comparison.py`: exports comparison routines for legacy and cross-package discovery.

## Evidence

`evidence/nm07_forward_surrogates_receipt.json`

## Limitations

Synthetic fixtures and software contracts only. No fabricated native training success,
acceleration claim, or physical certification. Contact boundaries require hybrid or
native collision resolution; smooth rigid approximations fail closed on impacts.

## Next Action

NM-08 (#10623): Closed-loop tracking vs. native tracking baseline under the neural
motion matching program.
