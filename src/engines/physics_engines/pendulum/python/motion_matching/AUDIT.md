# Pendulum Motion-Matching Plumbing Audit

## Current State

- The `pendulum` engine is a standalone, analytic Double Pendulum (2-DOF) model.
- The `PendulumFitSwingProvider` (`provider.py`) currently satisfies the structural API of the `provider_registry` but returns an analytic, zero-cost dummy `FitResult` instead of actually optimizing the simulation to track a target.

## Architectural Gap

1. **Dimensionality & Projection**: The `MultiSourceTarget` captures full spatial 3D kinematics (6-DOF club, spatial body markers). The Pendulum model operates in a pure 2D plane (theta1, theta2). The tracking target must be projected into the 2D swing plane before tracking error can be meaningfully evaluated.
2. **Missing Cost Function & Scipy Linkage**: To perform true motion matching, the provider must run a `scipy.optimize.minimize` loop (similar to Drake or MuJoCo) that wraps `PendulumPhysicsEngine.step()` and measures the error between the projected 2D C3D target and the simulated 2D clubhead.

## Recommended Path (Phase 2)

1. **Target Projection**: Utilize a shared 3D-to-2D projection utility (which will also serve Simscape 2D) to map `ClubTarget` into the double-pendulum plane.
2. **Scipy Optimization**: Implement the `fit_swing_scipy` algorithm inside the pendulum provider, searching for 14 polynomial coefficients (7 per joint for shoulder/wrist) that minimize the tracking error over the 2D projected trajectory.
3. **Analytic Surrogate Validation**: Use the pendulum model as the primary testbed for JAX-based surrogate optimizers (Issue #4075) since its forward dynamics are analytically fast and perfectly known.
