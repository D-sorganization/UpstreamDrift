# Canonical-Core Residuals

Issue #6791 adds the first shared residual surface for the canonical-core
estimator under `src/shared/python/estimation/residuals.py`.

The module is intentionally backend-light:

- `reprojection_residual(q, observed_uv, joint_center_fn, K, confidence, ...)`
  evaluates a pure forward-kinematics callback, applies optional CC-15
  keypoint offsets, projects through a pinhole camera, and weights each 2-D
  keypoint by `sqrt(confidence)`.
- `dynamics_residual(q, v, a, rnea_fn, ...)` evaluates an explicit RNEA
  callback. If no torque target is passed, torques are eliminated from the
  decision vector and the inverse-dynamics torque itself is penalized. Use
  `dof_indices=slice(6, None)` to drop unactuated floating-base coordinates.
- `anthropometric_prior_residual(parameters, nominal, sigma, weights=...)`
  implements a diagonal Gaussian prior over anthropometric parameters.
- `smoothness_residual(trajectory, dt=..., order=...)` applies first- or
  second-order finite differences over tangent-coordinate trajectories.

All residual functions are stateless and accept arrays plus explicit callbacks.
`residual_jacobian()` uses JAX autodiff when requested or available, with a
central finite-difference fallback for core installs and regression tests.

The first implementation is Python-only. The RNEA signature is deliberately the
same shape a future Rust/PyO3 kernel should expose:

```python
def rnea_fn(q, v, a):
    ...
```

That keeps CC-19 estimator assembly agnostic to whether the callable is backed
by Pinocchio Python, `upstream-pinocchio-id`, or a future Rust estimation crate.
