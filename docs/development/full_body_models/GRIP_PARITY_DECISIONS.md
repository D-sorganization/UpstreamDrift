# Grip Parity Decisions for the Full-Body Showpiece

Continuation of [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md), split out to respect the documentation size budget. Section numbering continues from that file.

---

## 18. Same-Input Bushing Grip Parity: MuJoCo, Drake and Pinocchio Against OpenSim (OSV-7 Phase 2, #11739)

### What

The section 17 OpenSim bushing grip is the reference. MuJoCo, Drake and Pinocchio each simulate the same free club on the same two hand bushings, driven by the same prescribed swing, and the shared metrics compare every engine's per-hand force, net force, internal force, squeeze, couple, deflection and rotation with the reference.

Shared code (`src/shared/python/grip_contact/`):

- `bushing_law.py`: the OpenSim `BushingForce` law, engine-agnostic. It is checked against OpenSim records to 1e-9 (`test_bushing_law.py`, three random seeds).
- `prescribed_motion.py`: `CoordinateSpline`, a vectorised Forsythe-Malcolm-Moler cubic spline that reproduces OpenSim's `SimmSpline` to round-off; `RigidBodyState.frame`; and `hand_frame_states`.
- `parity.py`:
  - `GripKineticsSeries`, one series per engine. It computes deflection and rotation from frame poses identically for every engine, and its analyses go through the GCV-10 `analyze_grip(split_method="bushing")` and `GripSeries` path.
  - `parity_errors` and the tolerance constants.

Engine modules, `src/engines/physics_engines/<engine>/python/grip_bushing.py`. Each exposes `simulate_grip_bushing` and `probe_bushing_forces`.

### Same Input

1. All engines prescribe the 44 coordinates of the OSV-10 fixture `tests/fixtures/club_face/swing_q_<club>.npz` with the same `SimmSpline`-equivalent spline. The coordinate speeds are the spline derivative.
2. Each engine computes the club pose and spatial velocity of its own rigid-weld full-body model with its own forward kinematics:
   - MuJoCo: `mj_kinematics` and `mj_comVel`. `mj_objectVelocity` reports the velocity at the centre of mass, so it is shifted to the body origin.
   - Drake: `EvalBodyPoseInWorld` and `EvalBodySpatialVelocityInWorld`.
   - Pinocchio: `forwardKinematics`.
3. The weld club frame carries both hand bushing frames at the spec grip frames, as in the OpenSim model, where both are on the left hand body.
4. The FK agrees across engines to round-off: Pinocchio against MuJoCo is 1.3e-15 in rotation, 1.6e-15 m in position, 4.6e-14 m/s in velocity and 2.1e-14 rad/s in angular velocity, over 40 times per club.
5. The club is a separate free body with the spec mass, centre of mass and inertia (0.313 kg, hands excluded) and spec gravity (0, 0, -9.80665). It starts at the t0 weld pose with zero velocity, as in the reference.
6. Stiffness and damping are the section 16 values: $K_t = 10^6$ N/m and $K_r = 1600$ N m/rad per axis, $c_t = (554, 209, 212)$ N s/m and $c_r = (0.81, 28.6, 27.3)$ N m s/rad.
7. Samples are taken at the 2 ms fixture times.

Simulating the club as a separate body is exact: the prescribed hand is kinematically driven, so the club affects it only through the bushing reaction, which a prescribed motion absorbs.

### Force Law per Engine

Frames: $F_1$ is the hand frame (prescribed) and $F_2$ the club frame. $R_i$ is world from frame.

- Rotational deflection: $\boldsymbol\theta$ is the body-fixed X-Y-Z angles of $R_1^T R_2$.
- Translational deflection: $\boldsymbol\delta = R_1^T(p_2 - p_1)$.
- Translation rate: $\dot{\boldsymbol\delta} = R_1^T(v_2 - v_1 - \omega_1 \times (p_2 - p_1))$.
- Rotation rate: $\dot{\boldsymbol\theta} = N(\boldsymbol\theta) R_2^T(\omega_2 - \omega_1)$.
- Force on the club, applied at the origin of $F_2$: $R_1(-(K_t\boldsymbol\delta + C_t\dot{\boldsymbol\delta}))$.
- Moment on the club: $R_2 N^T(-(K_r\boldsymbol\theta + C_r\dot{\boldsymbol\theta}))$.

How each engine applies it:

- **MuJoCo.** The shared law is applied through `mjcb_passive` with `mj_applyFT` into `qfrc_passive`. MuJoCo calls the callback in every RK4 stage at the stage time.
  - A soft weld equality is not used. Its one `solref` and `solimp` (or one stiffness and damping) applies to all six rows, it acts on the quaternion error, and it is scaled by the constraint-space inverse inertia and the impedance. It cannot express three translational and three rotational per-axis stiffnesses and dampings in X-Y-Z angle coordinates.
- **Drake.** Native `LinearBushingRollPitchYaw`, with the same coefficients, between a free hand body (pose and velocity overwritten from the FK at every derivative evaluation) and the free club.
  - This is a different law. It uses roll-pitch-yaw angles (space-fixed X-Y-Z, which is body-fixed Z-Y-X), expresses the translation in the halfway frame and applies the force at the midpoint of the two frame origins.
  - It agrees with OpenSim to first order in the deflection. The difference is second order, about $\theta/2$ relative, which is 0.7 % at the 0.84 degree peak. This difference is documented, not corrected.
  - `CalcBushingSpatialForceOnFrameC` gives the wrench on the club frame for the series.
- **Pinocchio.** The shared law gives the total wrench at the club body origin in body axes. That wrench is the free-flyer joint torque, and `aba` gives the club acceleration.

### Numerical Method

| Engine              | Integrator                                     | Settings                                 | Work, driver                        | Wall time, loaded host |
| ------------------- | ---------------------------------------------- | ---------------------------------------- | ----------------------------------- | ---------------------- |
| OpenSim (reference) | `Manager` Runge-Kutta-Merson, error controlled | accuracy 1e-5                            | n/a                                 | 411 s                  |
| MuJoCo              | RK4, fixed step                                | dt = 1e-4 s                              | about 18 100 steps                  | 504 s                  |
| Drake               | `Simulator` runge_kutta3, error controlled     | accuracy 1e-7, max step 2.5e-4 s         | 228 008 steps                       | 1791 s                 |
| Pinocchio           | SciPy `solve_ivp` DOP853, error controlled     | rtol 1e-9, atol 1e-12, max step 2.5e-4 s | 208 298 right-hand-side evaluations | 1770 s                 |

The bushing modes reach 946 Hz. The fixed step resolves them with about 10 RK4 steps per period, and the max-step caps keep the adaptive integrators from stepping over impact-phase transients.

Convergence: four independent integrators, three of them error controlled, agree to under 0.05 % RMS on every quantity except the Drake squeeze. That residual is the documented law difference, not integration error, so no single-engine step-halving study is reported.

### Acceptance (Set Before Any Comparison)

`tests/unit/grip_contact/test_grip_parity_metrics.py` guards the constants.

- **Quantities.** Every quantity in `PARITY_QUANTITIES`: force L and R, net force, internal force, squeeze, couple, deflection L and R, and rotation L and R.
- **Peak error.** $|\max\|x\| - \max\|x_\mathrm{ref}\|| / \max\|x_\mathrm{ref}\| \le 5\%$.
- **RMS error.** $\sqrt{\mathrm{mean}\,\|x - x_\mathrm{ref}\|^2} / \max\|x_\mathrm{ref}\| \le 2\%$, taken over the whole 0 to 1.8 s window.
- **Static hold.** With the hand held still for 0.2 s, the total bushing force equals the club weight within 1 %. This passes for every engine.
- **F = K delta.** A 0.1 mm offset along each hand axis gives a per-hand force of $-R_\mathrm{hand} K_t \boldsymbol\delta$. The engine's own total (MuJoCo `qfrc_passive`; Drake and Pinocchio $m(a_\mathrm{com} - g)$ from their dynamics) equals twice that within 1e-6. This passes for every engine.

The full-swing parity test is `test_full_swing_matches_the_opensim_reference[engine-club]`. It is slow-marked: it re-simulates each engine.

### Results

Driver (capture A), errors relative to the OpenSim peak, given as peak / RMS:

| Quantity           | OpenSim peak     | MuJoCo          | Drake           | Pinocchio       |
| ------------------ | ---------------- | --------------- | --------------- | --------------- |
| Left hand force    | 515.2 N          | 0.003 / 0.001 % | 0.018 / 0.044 % | 0.000 / 0.001 % |
| Right hand force   | 564.0 N          | 0.001 / 0.001 % | 0.013 / 0.041 % | 0.000 / 0.001 % |
| Net force          | 439.3 N          | 0.005 / 0.002 % | 0.006 / 0.004 % | 0.000 / 0.001 % |
| Internal force     | 509.8 N          | 0.000 / 0.000 % | 0.012 / 0.045 % | 0.000 / 0.000 % |
| Squeeze            | 3.28 N           | 0.011 / 0.006 % | 0.962 / 0.415 % | 0.002 / 0.003 % |
| Couple at midpoint | 99.5 N m         | 0.005 / 0.004 % | 0.051 / 0.016 % | 0.000 / 0.001 % |
| Deflection L / R   | 0.515 / 0.564 mm | 0.005 / 0.003 % | 0.028 / 0.044 % | 0.001 / 0.001 % |
| Rotation L / R     | 0.84 deg         | 0.000 / 0.000 % | 0.005 / 0.050 % | 0.000 / 0.000 % |

7-iron (capture B), errors relative to the OpenSim peak, given as peak / RMS:

| Quantity           | OpenSim peak     | MuJoCo          | Drake           | Pinocchio       |
| ------------------ | ---------------- | --------------- | --------------- | --------------- |
| Left hand force    | 533.1 N          | 0.001 / 0.001 % | 0.011 / 0.021 % | 0.000 / 0.000 % |
| Right hand force   | 584.4 N          | 0.002 / 0.001 % | 0.038 / 0.020 % | 0.000 / 0.000 % |
| Net force          | 456.7 N          | 0.003 / 0.002 % | 0.001 / 0.002 % | 0.000 / 0.000 % |
| Internal force     | 524.1 N          | 0.000 / 0.000 % | 0.021 / 0.022 % | 0.000 / 0.000 % |
| Squeeze            | 3.86 N           | 0.013 / 0.005 % | 1.697 / 0.365 % | 0.000 / 0.001 % |
| Couple at midpoint | 98.6 N m         | 0.004 / 0.002 % | 0.021 / 0.006 % | 0.000 / 0.001 % |
| Deflection L / R   | 0.534 / 0.585 mm | 0.001 / 0.001 % | 0.037 / 0.021 % | 0.000 / 0.000 % |
| Rotation L / R     | 0.84 deg         | 0.001 / 0.000 % | 0.282 / 0.037 % | 0.000 / 0.000 % |

Iron wall times on the loaded host: OpenSim 768 s, MuJoCo 261 s, Drake 1904 s (297 403 steps), Pinocchio 1158 s (318 896 right-hand-side evaluations).

Every engine passes every quantity for both clubs, with no tolerance changed.

The largest deviation is the Drake squeeze. The squeeze is a 3 N difference of two forces of about 500 N, so it is the quantity most sensitive to the second-order roll-pitch-yaw and midpoint convention. It is still about three times inside the peak tolerance (1.70 % on the iron, 0.96 % on the driver). Pointwise, its Drake error reaches 2.7 % of the squeeze peak at single samples between 1.30 and 1.40 s (`error_driver.png`); the acceptance metrics are peak magnitude and RMS, and both pass.

MuJoCo and Pinocchio evaluate the same law as OpenSim, so their residual is integration error only.

### Evidence Receipt

- Series (one per engine and club) and run logs: `evidence/grip_kinetics/parity/<engine>_<club>_series.npz` and `runs_<club>.json`.
- Metrics: `evidence/grip_kinetics/parity/metrics_<club>.json`.
- Reproduce, one heavy run at a time:
  - Reference: `PYTHONPATH=.:src python3 docs/development/full_body_models/evidence/grip_kinetics/run_grip_parity.py --club driver --reference`
  - Engines: the same command with `--engines mujoco drake pinocchio`.
  - Report: the same command with `--report`.
  - Repeat for `--club iron7`.
- Tests:
  - `python3 -m pytest -n 0 tests/unit/grip_contact` covers the law, spline, metrics, static hold and F = K delta.
  - Add `-m slow -k full_swing` for the full-swing parity.
- Plots (outside the repository) in `~/Videos/Parity Audit/golfer_realism/grip_kinetics/parity/`:
  - `overlay_<club>.png` and `overlay_<club>_impact.png`: all engines over the reference, with per-panel errors.
  - `error_<club>.png`: error traces as a percentage of the reference peak.
  - `gcv10_<engine>_<club>.png`: each engine through the GCV-10 `build_grip_plot_series` and `plot_grip_wrench` path.

### What Was Tried and Rejected

- MuJoCo soft weld equality as the bushing: rejected on the grounds given above, before any run. It is not a same-law comparison.
- Using `mj_objectVelocity` directly as the origin velocity: rejected. It reports the centre-of-mass velocity, and the finite-difference check failed until it was shifted to the body origin.
- `np.cross` and `np.allclose` in the per-stage law: correct but about 50 times slower per call. The MuJoCo static test hit the 60 s timeout, so they were replaced by `cross3` and a cheaper orthonormality check.
- Pinocchio fixed-step RK4: not used, because the error-controlled DOP853 meets the plan's requirement for an error-controlled integrator. No engine is reported as unavailable.

### Limitations

- This is parity of one bushing model across engines. It is not validation of the grip: the section 16 and 17 limitations (stiffness defaults, the free-torque split being 50/50 by construction, no ball) all still apply.
- Drake is compared with its native law, so its small residual is a model-definition difference, not an engine defect. A Drake run that applies the shared law through an `ExternallyAppliedSpatialForce` would isolate Drake's integrator and is not done.
- On the loaded shared host, the full-swing parity test takes about 10 to 30 minutes per engine and club.
- This covers software correctness only. Scientific qualification stays in the design-manual governance pathway.
