# NM-08 Native Verification, Distribution Checks and Safe Fallback

Governing issue: #10623 (epic #10603).

Schema:
`neural-verified-inference/1.0.0`

## What Landed

Primary package under `src/shared/python/neural_motion/inference/`:

- `VerifiedInferenceOrchestrator`: single entry service coordinating:
  - Input validation: checks target timestamps, non-finite / NaN values, and observation dimensions.
  - Distribution diagnostics: compares target duration, peak velocities, and contact regimes against empirical training coverage bounds (`DistributionBounds`).
  - Empirical confidence scoring: computes domain distance metrics ($[0.0, 1.0]$) representing domain density/support, explicitly distinct from golfer truth probability.
  - Checkpoint contract verification: validates `ProposalCheckpointContract` model ID, control dimension, and weight digest before execution.
  - Fail-closed neural execution: detects NaN outputs or wrong control dimensions and falls back immediately.
  - Native refinement & independent replay: mandates independent forward dynamic replay before dynamic acceptance.
  - Shared-budget classical fallback: on neural failure, passes remaining wall-clock budget to classical/retrieval solver and retains all attempts in `VerifiedInferenceReport`.
- `InferenceStatus` (`NEURAL_ACCEPTED`, `NEURAL_REFINED`, `CLASSICAL_FALLBACK`, `REJECTED`).
- `VerifiedInferenceReport` and `AttemptRecord` tracking every attempt's phase, cost, duration, and rejection reason.

Integration into `src/shared/python/motion_matching/hybrid.py`:

- Re-exports verified inference types and orchestrator.
- Adds `fit_swing_verified_inference` facade supporting both `ClubTarget` and complete `MultiSourceTarget`.

## Evidence

`evidence/nm08_native_verification_receipt.json`

## Limitations

Synthetic fixtures and software contracts only. No fabricated native training
acceleration or physical qualification claims. Confidence metrics report empirical
domain coverage, not human golfer truth probabilities.

## Next Action

NM-09 (#10624): Multi-engine native parity and dynamic feasibility transfer across
physics lanes under epic #10603.
