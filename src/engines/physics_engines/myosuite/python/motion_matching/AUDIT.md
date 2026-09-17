# MyoSuite Motion-Matching Plumbing Audit

> [!WARNING] > **SUPERSEDED (2026-09-17)**: The claims in this audit document are superseded by the Matched Swing Program single source of truth. See [`docs/development/matched_swing_program/README.md`](../../../../../../docs/development/matched_swing_program/README.md) for authoritative physical gates, engine qualification status, and run evidence.

## Current State

- MyoSuite provides high-fidelity, 290-muscle simulation via MuJoCo.
- Control inputs are muscle activations $\in [0, 1]$, not joint torques.
- Motion matching cannot be directly mapped to the polynomial torque coefficients paradigm used by Drake/Pinocchio/Simscape.
- No existing `fit_swing` architecture exists specifically for MyoSuite in this codebase.

## Architectural Gap

Motion matching (currently) searches for 7 polynomial coefficients per joint. For MyoSuite, the optimization space is 290 muscle activations over time. Using `scipy.optimize.minimize` with finite-differences over 290 time-varying signals is computationally intractable.

## Recommended Path (Phase 2)

1. **Surrogate Model**: Develop a neural surrogate (similar to Simscape Option 2) that maps muscle activations to joint torques/kinematics, allowing fast differentiable optimization via JAX.
2. **Inverse Muscle Model**: Use a trained inverse dynamics network to map desired joint trajectories (from standard MuJoCo/Pinocchio motion matching) down to muscle activations.

## Implementation Details (MS-50 Audit Update)

- **Provider Registration**: `MyoSuiteFitSwingProvider` is registered under engine key `"myosuite"` in `PROVIDER_REGISTRY`.
- **Fail-Closed Contract**: `fit_swing` fails closed, returning a `CanonicalFitResult` with `solver_status="unsupported"` and an explicit diagnostic message. It never executes unvalidated surrogate passes or synthesizes fabricated muscle activations.
- **Target Support**: `supports_body_target()` returns `False` pending body marker IK and muscle inversion work (MS-53); `supports_ball_target()` returns `False`.
- **Tile Status & UI**: The launcher tile status is marked `experimental` in `src/config/models.yaml` and `src/config/launcher_manifest.json`. The dashboard GUI probes engine availability on open and provides explicit installation instructions for the missing wheel (`myosuite`) and submodule (`shared/models/myosuite/myo_sim`).
