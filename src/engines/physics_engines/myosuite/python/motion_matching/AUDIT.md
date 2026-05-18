# MyoSuite Motion-Matching Plumbing Audit

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

## Implementation Details

The canonical API structure has been established (`provider.py`) and is hooked into the global `provider_registry` to fulfill the `MultiSourceTarget` compatibility interface, but `fit_swing` currently raises `NotImplementedError` pending the Phase 2 surrogate or muscle inversion implementation.
