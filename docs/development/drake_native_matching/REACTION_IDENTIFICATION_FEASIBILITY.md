# Reaction-Eliminated Polynomial Identification Feasibility

## Conclusion and Scope

A closed loop does not require assuming its reaction forces are zero. A linear identification step can eliminate those unknown reactions and recover a global sixth-order effort profile **when a smooth, dynamically compatible native trajectory is already available**. This is a diagnostic prototype, not a built-in tree inverse-dynamics call, Simscape prescribed-motion bypass, C3D fitting program, or accepted swing.

The bounded study successfully reconstructs generalized efforts and held-out accelerations on a known native baseline. The one authorized continuous replay reproduces marker motion to8.84e-9m, but fails the unchanged1e-4 scalar-rate reconstruction gate. Therefore this study supports further initializer development; it does not certify strict full-state reconstruction or replace existing forward acceptance.

## Mathematics and Exact Input Convention

Let q,v,a be a closure-feasible trajectory and let h contain the inertial bias and gravity terms with the sign convention

```math
M(q)a+h(q,v)=B u(t)+J(q)^T\lambda,\qquad
J(q)v=0,\qquad J(q)a+\gamma(q,v)=0.
```

The six weld reaction components are lambda. At each state choose an orthonormal basis N for the nullspace of J. Multiplying the equation by N.T eliminates the reactions:

```math
N^T B u(t)=N^T(Ma+h),\qquad N^T J^T=0.
```

No reaction is set to zero. In this adapter `h = CalcBiasTerm - CalcGravityGeneralizedForces`, and `J`/gamma use Drake's relative spatial velocity Jacobian and bias spatial acceleration for the weld frames. Both are expressed in the same closure-frame convention. Coordinate indices are explicitly permuted from Drake order to the native order.

Each input uses one global degree-six Bernstein polynomial on the fixed interval[0,T]. The basis is evaluated at absolute t/T with a single origin, not reset per phase. Stack all27 times7 controls into theta in coordinate-major order. The existing `NativeEffortProfile.bernstein_control_jacobian(t, basis_duration_s=T, first_control=0)` supplies E(t), and each linear block is `N.T @ E(t)`. Shared `bernstein_to_simscape` converts the result to the required highest-power-first absolute-seconds polynomial. No polynomial evaluator or force mapping was duplicated in the prototype.

The input mapping is `B = diag(R_world_to_hip_base, I_24)`, with the rotation equal to the transpose of the root joint's exported `parent_to_base` rotation. The first three inputs are world forces; the other24 are primitive-conjugate torques. This is the exact existing shared profile convention. It must not be replaced by a state-dependent guessed rotation or a generic engine actuator-order mapping.

## What Was Tested

Six analytic tests first failed because the prototype module was absent, then passed. A two-coordinate closed-weld toy has14 sextic coefficients but identifies only their seven summed coefficients; a minimum-norm solution differs from the generating coefficients while satisfying the projected equations. Further tests verify arbitrary weld-reaction additions vanish under projection, incompatible accelerations are rejected, nonfinite inputs are rejected, and a rank-deficient constraint Jacobian is rejected. The mathematical helper checks acceleration closure; callers remain responsible for position/rate closure and derivative consistency.

The native study reused the existing qualified0.8-second baseline,81 saved states, original model hash and original baseline candidate `2afaf8b21a05a44b071e7328e2d624bba5f6a999aa85920b53b61b43961e6673`. Pinocchio supplied acceleration values at those existing states from its actual constrained forward dynamics. Independently, Drake supplied M,h,J,gamma. The known coefficients generated the reference acceleration data, as expected in a synthetic feasibility experiment; the linear least-squares solve did not receive them as a prior or initial guess. This is not an estimate of human torque from observed C3D data.

Position/rate closure maximum was3.54e-11 and acceleration compatibility `J a+gamma` was1.35e-9. Forty-one alternating samples trained the global profile;40 interleaved samples were held out. All inputs remain native27-coordinate,31-solid,6D-weld geometry. No model, runtime or fitter source was changed.

| Quantity                                            |                Result |
| --------------------------------------------------- | --------------------: |
| Global Controls                                     |                   189 |
| Training Scalar Equations                           |                   861 |
| Column-Normalized Rank at Relative Cutoff1e-10      |                   189 |
| Maximum / Minimum Normalized Singular Values        | 2.56227 / 0.000424185 |
| Normalized Condition Number                         |             About6040 |
| Training Projected Residual Maximum                 |              6.85e-10 |
| Held-Out Projected Residual Maximum                 |              4.63e-10 |
| Held-Out Same-State Acceleration Difference         |               1.96e-7 |
| Held-Out Scaled Acceleration Difference             |               4.19e-9 |
| Recovered Versus Known Bernstein Control Difference |               6.54e-8 |
| Sampled Primitive Effort Difference                 |               3.97e-8 |

Native power coefficients differ by up to3.01e-5 because conversion magnifies coefficient-scale differences. The largest absolute control is818.488, also present in the known original profile; it is not a newly invented torque limit. Forces and torques have different physical units, so coefficient magnitudes and raw projected residuals require unit-aware interpretation.

At one instant,27 efforts and21 independent dynamic equations leave six reaction-related directions undetermined. Across this particular moving trajectory, the changing constraint geometry plus the global polynomial restriction produced full rank189. That result does not imply full rank for another swing, static pose, shorter window, different degree, noisy trajectory, or fewer samples. The toy explicitly demonstrates rank deficiency. Conditioning here is based on column normalization and this sampling, not a universal identifiability guarantee.

## Continuous Replay Result

The identified candidate is `2bca73b6c0533b685ed191c4b8ee1bc4df7c69b4053be202bc85e4c7d1db2c96`. Only its polynomial coefficients differ from the original candidate; q0/qd0 are unchanged. A single continuous Pinocchio replay used the original81-sample clock through0.8s, DOP853, relative/absolute tolerances1e-11/1e-13 and maximum step0.00025s.

| Comparison Against Original Baseline |              Result |
| ------------------------------------ | ------------------: |
| Maximum q Difference                 |             1.85e-7 |
| Maximum Scalar-Rate Difference       |             3.37e-4 |
| Maximum Marker Vector Difference     |            8.84e-9m |
| Closure Pose / Rate Maximum          | 2.32e-11 / 1.67e-10 |
| Replay Time                          |               3.21s |
| Full-State Reconstruction Gate       |              Failed |

The scalar-rate difference exceeds the unchanged1e-4 gate; the JSON explicitly records `baseline_reconstruction_passed=false`. No tolerance sweep or gate relaxation followed. The earlier documented Euler-coordinate sensitivity is relevant context but is not a substitute for proving convergence or changing criteria scientifically.

## Prerequisites for a Useful C3D Initializer

1. First produce a smooth native closure-feasible kinematic trajectory that reasonably matches the actual marker data. Generic reference-model coordinates, independent static poses, interpolated Euler angles and numerically differentiated noisy C3D samples do not automatically satisfy this requirement. Enforce position, velocity and acceleration closure, and preserve a continuous branch through coordinate-chart difficulties.
2. Estimate derivatives from a smooth fitted representation with stated uncertainty. Second differentiation amplifies measurement noise. A low marker error with noisy qdd can generate implausible efforts; projected dynamic equations alone do not repair that path.
3. Calibrate any permitted geometry/length changes separately. Their changes affect M,h,J and require new model identities and qualification. A rigid-marker feasibility floor cannot be removed by torque identification.
4. Build the reaction-eliminated global system using the shared native effort design. Report rank, singular values, residuals and uncertainty weighting. Use physically justified row scaling and coefficient regularization; the prototype only uses column normalization and unconstrained linear least squares.
5. Add the authorized effort/control bounds and early-motion priors to a bounded linear solve. Preserve actual numerical search bounds separately from physical model limits. Earlier torque samples can be weighted prior equations; freezing a polynomial exactly over a continuous early interval fixes it everywhere, so later behavior cannot change independently under one global polynomial.
6. Forward replay the resulting single polynomial from original q0/qd0 through the full capture. Apply the current marker, continuity, closure and state criteria, observed masks and independent-engine checks. The identification residual is only an initializer diagnostic. If necessary, use the profile to seed the existing forward nonlinear optimizer rather than copying that optimizer.
7. Preserve the fitted path, derivative representation, exact model/capture identities, coefficient basis, ranks, bounds, priors and every replay receipt. Extend the production shared interface only after the root agent reviews this bounded study.

This route may provide substantially better full-swing starting controls once a suitable native kinematic path exists. The expensive unresolved work is obtaining that path reliably and validating its forward motion, rather than solving the small linear coefficient system.

## Files and Reproduction

- Prototype: `reproduction/reaction_identification.py`; no production imports reference it.
- Tests: `reproduction/test_reaction_identification.py`.
- Native study: `reproduction/study_reaction_identification.py`, with separate `reference` and `identify` phases.
- One replay: `reproduction/replay_identified_baseline.py`.
- Reports and exact raw bundle: `evidence/reaction-identification/`.

The ZIP preserves generating and identified candidates, state/acceleration references, training and held-out matrices, singular values, coefficients, actual replay, exact executed scripts, immutable runtime sources and environment distributions. Adjacent JSON is formatted for review; use ZIP bytes for hashes. The only identified-profile integration is the failed-gate replay described above. No C3D optimization or current root fitting process was modified.

```powershell
python3 -m pytest docs/development/drake_native_matching/reproduction/test_reaction_identification.py -q --no-cov
python3 -m mypy --follow-imports=silent docs/development/drake_native_matching/reproduction/reaction_identification.py
```

Native executions used the existing ControlTower Pinocchio and Drake environments with `PYTHONPATH=/home/dieterolson/drake-native-runtime-10022-02`, `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. All jobs are terminal. TDD tests, direct Ruff checks and direct mypy passed. Next action is root review of this feasibility report, not a new broad implementation in this lane.
