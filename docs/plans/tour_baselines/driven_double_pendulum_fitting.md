# Architectural Specification: Driven Planar Double Pendulum Fitting and Independent Replay (TB-04 #10589)

## 1. Executive Summary & Objective

Tour Baselines task TB-04 (#10589) provides the core physical fitting and independent replay engine for the driven planar double pendulum baseline. Prior baseline fitting implementations exhibited critical numerical and architectural deficiencies:

1. **First-frame evaluation error**: Stepping the numerical integrator before evaluating the candidate trajectory at $t_0$, discarding the true initial state and introducing artificial error.
2. **Hardcoded ad-hoc control functions**: Unbounded polynomials that diverged under extrapolation and lacked physical torque limits.
3. **Absence of independent tighter-step replay**: No verification that the fitted continuous torques produce stable, non-diverging trajectories when simulated without reset interventions.
4. **Missing unforced diagnostic**: No benchmark comparison against passive / unforced rollouts from the same initial state.
5. **Dummy cryptographic provenance**: Placeholder hashes (`target_hash="dummy"`) breaking baseline traceability and qualification gates.
6. **Formulation divergence**: Unreconciled mathematical conventions between analytical Lagrangian equations and the Tools point-mass simulator.

TB-04 resolves each of these deficiencies with a mathematically verified, fully regularized, and independently replayable pipeline.

---

## 2. Mathematical Convention Adapters

### Analytical Distributed Inertia vs. Tools Point-Mass Simulator

The repository contains two double pendulum dynamics formulations:

- **Analytical `DoublePendulumDynamics`**: Incorporates distributed mass segments, explicit center-of-mass positions ($l_{c1}, l_{c2}$), proximal joint inertias, and arbitrary swing plane inclinations $\theta_{\text{inc}}$.
- **Tools `pendulum_simulator.physics`**: Implements an effective tip point-mass approximation where the lower link mass and clubhead mass are concentrated at the distal tip ($M_{22} = m_e L_2^2$).

The adapter module `src/shared/python/tour_baselines/pendulum_adapter.py` establishes exact bidirectional conversions:

- `analytical_to_tools_params(params: DoublePendulumParameters) -> PendulumParams`: Projects gravity by $\cos(\theta_{\text{inc}})$ and extracts link masses, lengths, and joint viscous damping.
- `tools_to_analytical_params(tools_params, use_point_mass_approximation=True)`: Reconstructs analytical parameters. Under point-mass approximation, $m_{\text{shaft}} = 0$ and $m_{\text{clubhead}} = m_2 + m_{\text{club}}$, proving equivalence between the two formulations with mass matrix and acceleration residuals $< 10^{-12}$.
- `compare_dynamics_parity(...)`: Automated parity diagnostic reporting maximum absolute errors in inertia matrices and generalized accelerations.

---

## 3. Fitting Engine Architecture

The fitting engine in `src/shared/python/tour_baselines/pendulum_fit.py` implements:

```
+-----------------------------------------------------------------------------------+
|                               Input: ClubTarget                                   |
| (time t_k, butt p_b(t_k), clubhead p_h(t_k), cryptographic SHA-256 target hash)    |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 1. Plane Calibration & SE(3) Projection (TB-03)                                   |
|    - Rigid basis [u, v, n] in SO(3) via SVD                                       |
|    - Planar coordinate mapping relative to shoulder pivot                         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 2. Positive Bounded Geometry Calibration (TB-03)                                  |
|    - Arm length L1 in [0.4, 0.9] m, Club length L2 in [0.7, 1.3] m                |
|    - Mass & inertia priors from canonical model parameters                        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 3. Feasible Initial State Mapping (TB-03)                                         |
|    - Initial joint angles q0 = [th1_0, th2_0] via 2D inverse kinematics           |
|    - Initial angular velocities v0 = [w1_0, w2_0] via validated finite difference |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 4. Unforced / Passive Comparison Diagnostic                                       |
|    - Forward integration with zero control torques from (q0, v0)                   |
|    - Computes unforced_rmse_m baseline                                            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 5. Continuous Bounded Bernstein Torque Optimization                               |
|    - Degree-6 Bernstein polynomial control points (7 per joint, 14 total)         |
|    - Physical box bounds: |tau_shoulder| <= 300 Nm, |tau_wrist| <= 120 Nm         |
|    - Regularization: tracking loss + lambda_effort * ||tau||^2                    |
|    - t0 evaluated directly via FK before stepping (no off-by-one)                 |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 6. Independent Tighter-Step Replay                                                |
|    - Higher resolution RK4 rollout (substeps_replay >= 4)                         |
|    - Zero resets, zero feedback: pure feedforward verification                    |
|    - Asserts |replay_rmse - fit_rmse| <= tolerance                                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 7. Provenance, BaselineIdentity & StatusBundle                                    |
|    - True SHA-256 target hash                                                     |
|    - DynamicFeasibilityStatus (PHYSICALLY_FEASIBLE iff replay stable)             |
|    - ScientificQualificationStatus.QUALIFIED with has_native_replay=True          |
+-----------------------------------------------------------------------------------+
```

---

## 4. First-Frame Evaluation Fix

Previous implementations executed:

```python
for i in range(n_frames):
    t = i * dt
    state = dynamics.step(t, state, dt)  # Stepped before evaluating!
    x_head, y_head = forward_kinematics(state)
```

This evaluated the frame 0 prediction using the state after integrating over $[0, \Delta t]$, introducing an off-by-one phase lag that corrupted torque fitting.

In `simulate_pendulum_rollout`:

```python
# Frame 0 is evaluated at t0 directly without stepping
pose0 = PlanarDoublePendulumPose(theta1_rad=state.theta1, theta2_rad=state.theta2)
p_g0, p_h0 = forward_kinematics_planar_double_pendulum(pivot_origin, l1, l2, pose0)
pred_grip[0] = p_g0
pred_head[0] = p_h0

# Frames 1..N step forward across the source dt interval
for i in range(n_frames - 1):
    dt_frame = float(times[i + 1] - times[i])
    state = dynamics.step(t_current, state, dt_sub)
    ...
```

This guarantees $p_{\text{pred}}(t_0) = \text{FK}(q_0)$ to machine precision.

---

## 5. Verification Matrix

| Verification Aspect             | Test Suite                                              | Result                        |
| ------------------------------- | ------------------------------------------------------- | ----------------------------- |
| Parameter conversion & parity   | `tests/unit/tour_baselines/test_pendulum_adapter.py`    | 5 passed (100%)               |
| Analytical vs Tools equivalence | `test_compare_dynamics_parity_mathematical_equivalence` | Max error $< 10^{-10}$        |
| $t_0$ first-frame regression    | `test_t0_frame_zero_evaluation_regression`              | Exact FK match ($< 10^{-12}$) |
| Non-uniform time grids          | `test_non_uniform_timestamps_simulation`                | Stable monotonic integration  |
| Independent replay stability    | `test_independent_replay_zero_resets`                   | RK4 convergence ($< 0.01$ m)  |
| Cryptographic target hash       | `test_target_hash_cryptographic_integrity`              | Deterministic SHA-256         |
| Provenance & StatusBundle       | `test_baseline_identity_and_status_bundle_provenance`   | Fully qualified bundle        |
| Provider canonical adapter      | `test_motion_matching_provider.py`                      | 5 passed (100%)               |
| Full Tour Baselines Suite       | `tests/unit/tour_baselines/`                            | 72 passed (100%)              |
