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

| Engine | Integrator | Settings | Work, driver | Wall time, loaded host |
| --- | --- | --- | --- | --- |
| OpenSim (reference) | `Manager` Runge-Kutta-Merson, error controlled | accuracy 1e-5 | n/a | 411 s |
| MuJoCo | RK4, fixed step | dt = 1e-4 s | about 18 100 steps | 504 s |
| Drake | `Simulator` runge_kutta3, error controlled | accuracy 1e-7, max step 2.5e-4 s | 228 008 steps | 1791 s |
| Pinocchio | SciPy `solve_ivp` DOP853, error controlled | rtol 1e-9, atol 1e-12, max step 2.5e-4 s | 208 298 right-hand-side evaluations | 1770 s |

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

| Quantity | OpenSim peak | MuJoCo | Drake | Pinocchio |
| --- | --- | --- | --- | --- |
| Left hand force | 515.2 N | 0.003 / 0.001 % | 0.018 / 0.044 % | 0.000 / 0.001 % |
| Right hand force | 564.0 N | 0.001 / 0.001 % | 0.013 / 0.041 % | 0.000 / 0.001 % |
| Net force | 439.3 N | 0.005 / 0.002 % | 0.006 / 0.004 % | 0.000 / 0.001 % |
| Internal force | 509.8 N | 0.000 / 0.000 % | 0.012 / 0.045 % | 0.000 / 0.000 % |
| Squeeze | 3.28 N | 0.011 / 0.006 % | 0.962 / 0.415 % | 0.002 / 0.003 % |
| Couple at midpoint | 99.5 N m | 0.005 / 0.004 % | 0.051 / 0.016 % | 0.000 / 0.001 % |
| Deflection L / R | 0.515 / 0.564 mm | 0.005 / 0.003 % | 0.028 / 0.044 % | 0.001 / 0.001 % |
| Rotation L / R | 0.84 deg | 0.000 / 0.000 % | 0.005 / 0.050 % | 0.000 / 0.000 % |

7-iron (capture B), errors relative to the OpenSim peak, given as peak / RMS:

| Quantity | OpenSim peak | MuJoCo | Drake | Pinocchio |
| --- | --- | --- | --- | --- |
| Left hand force | 533.1 N | 0.001 / 0.001 % | 0.011 / 0.021 % | 0.000 / 0.000 % |
| Right hand force | 584.4 N | 0.002 / 0.001 % | 0.038 / 0.020 % | 0.000 / 0.000 % |
| Net force | 456.7 N | 0.003 / 0.002 % | 0.001 / 0.002 % | 0.000 / 0.000 % |
| Internal force | 524.1 N | 0.000 / 0.000 % | 0.021 / 0.022 % | 0.000 / 0.000 % |
| Squeeze | 3.86 N | 0.013 / 0.005 % | 1.697 / 0.365 % | 0.000 / 0.001 % |
| Couple at midpoint | 98.6 N m | 0.004 / 0.002 % | 0.021 / 0.006 % | 0.000 / 0.001 % |
| Deflection L / R | 0.534 / 0.585 mm | 0.001 / 0.001 % | 0.037 / 0.021 % | 0.000 / 0.000 % |
| Rotation L / R | 0.84 deg | 0.001 / 0.000 % | 0.282 / 0.037 % | 0.000 / 0.000 % |

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

## 19. Contact Grip, MyoSuite Bushing and Kinetics Validation (OSV-7 Phase 3, #11739)

### What

Phase 3 adds a second grip interface model, a distributed pad `contact` grip, next to the section 16 bushing, runs the bushing in MyoSuite, and reports kinetics validation for the two. Nothing in the bushing law, the parity metrics or any tolerance changed.

- MyoSuite bushing: `src/engines/physics_engines/myosuite/python/grip_bushing.py`. MyoSuite is MuJoCo behind `MujocoEnv`; the model is loaded through `load_myosuite_runtime` and handed to the MuJoCo bushing module through an injected loader, so the physics is the MuJoCo section 18 code and the runtime is MyoSuite's.
- Contact grip: `src/shared/python/grip_contact/pad_layout.py`, `pad_contact.py`, `contact_run.py`, and per engine `src/engines/physics_engines/{mujoco,drake,pinocchio}/python/grip_contact_sim.py`.
- Static balance: `src/shared/python/grip_contact/static_balance.py`.
- Evidence runners: `evidence/grip_kinetics/run_grip_contact.py`, `run_grip_validation.py`, `render_grip_clip.py`.

### Interface Model

Per hand, two rings (axial offsets $\pm a$) of $n = 6$ spherical pads of radius 8 mm press on a rigid cylinder of radius $r = 12.7$ mm (the spec hand standoff) that is rigidly part of the club. The grip frame origin of each hand sits on the hand surface, $r$ from the shaft axis, so the two hand frames are $2r$ apart across the grip. A pad centre rests $r + r_\mathrm{pad} - \delta_0$ from the axis, where $\delta_0$ is the preload interference. The right-hand rings are turned by half a pad spacing. The hands are the prescribed OSV-7 hand frames, as in the bushing runs; only the interface differs.

### Parameters (Derived, Not Fitted)

- Stiffness. A translation across the axis of one hand compresses the pads in proportion to $\cos^2$, so $K_t = k_\mathrm{pad}\, n/2 - f_\mathrm{pad}/\rho$, where $f_\mathrm{pad}$ is the pad preload and $\rho$ the pad centre distance from the axis. The second term is the negative geometric stiffness of a preloaded pad. So $k_\mathrm{pad} = 2K_t/n + f_\mathrm{pad}/\rho$, which is about $1.71 \times 10^5$ N/m for both clubs. Bending uses $a = \sqrt{K_r/K_t} = 0.04$ m, the bushing's own effective radius.
- Damping. Hunt-Crossley $c = c_t / (N_\mathrm{pads}/2 \cdot \text{preload})$ so the damping across the grip equals the bushing translational damping.
- Squeeze (total pad normal force per hand), derived from the OpenSim bushing demand of the same club and not tuned: $N = \max\left(F_\mathrm{ax}/\mu,\ \tau_\mathrm{ax}/(\mu r),\ 2F_\perp\right)$ with the peak axial force, axial torque and cross-axis force of the bushing reference and $\mu = 0.9$. This gives 1104 N for the driver and 1155 N for the 7-iron.
- Friction: static 0.9 and dynamic 0.7 from `ContactMaterial` (still engineering placeholders, not measurements). Stribeck transition speed 1 mm/s.

### Equations (Shared Pad Law)

For pad $j$ with centre $p_j$ and velocity $v_j$ relative to the cylinder surface, with penetration $\delta_j$ and normal $\hat n_j$ (from the cylinder axis to the pad centre):

- Normal force (Hunt-Crossley): $F_{n,j} = k_\mathrm{pad}\,\delta_j\,(1 + c\,\dot\delta_j)$ for $\delta_j > 0$ and $F_{n,j} \ge 0$.
- Tangential force: regularised Coulomb with a Stribeck blend, magnitude $\le \mu(v_t) F_{n,j}$, acting against the tangential slip speed. This is `physics.contact_law.sphere_ground_contact` applied against the cylinder tangent plane at the pad.
- Hand wrench on the club: $F = \sum_j (F_{n,j} \hat n_j + F_{t,j})$ and $M = \sum_j (p_j - p_\mathrm{ref}) \times (\cdot)$ about the club grip origin, plus torsional friction about each pad normal (`torsional_friction_moment`).
- There is no spring against translation along the axis or rotation about it; those are carried by friction only. This differs from the bushing, which has springs on all six freedoms.

Engine implementations:

- MuJoCo: native sphere-cylinder contacts, `condim` 6, elliptic cone, soft constraint with per-pad `solref` calibrated so each pad force equals $k_\mathrm{pad}\,\delta$ (calibration error under $10^{-3}$). The friction rows need an explicit `solreffriction` of two timesteps. Creep scales with the timestep, so the runs use $\Delta t = 10^{-5}$ s (Euler); 300 to 360 s per full swing. The hand bodies are heavy (100 kg) and overwritten every step, and the club starts with the weld velocity, to avoid a stiff-friction impulse at release.
- Pinocchio: the shared law evaluated analytically and integrated with `aba` and SciPy Radau (rtol $10^{-6}$, atol $10^{-9}$, max step $2 \times 10^{-4}$ s).
- Drake: continuous `MultibodyPlant` with SceneGraph point contact (`kPoint`), Hunt-Crossley, stiction tolerance $10^{-3}$, implicit Euler (an explicit RK3 shrinks to steps near $5 \times 10^{-7}$ s and is unusable). Drake has no torsional friction in point contact, so its hold is checked without that term.

### Acceptance (Fixed Before the Runs)

- Static hold: pad forces sum to the club weight within 1 %; tests `test_engine_grip_contact.py` for MuJoCo, Drake and Pinocchio (Drake and Pinocchio slow-marked, 0.05 s hold; MuJoCo 0.2 s).
- Friction prevents sliding: axial slip under 0.1 mm over the hold at the real friction, and a nearly frictionless grip ($\mu = 10^{-4}$, below weight over squeeze) does slide, so the test can fail.
- Quasi-static balance: with the hands still at four swing attitudes, $\sum F + mg = 0$ and the moment of the per-hand wrenches about the club centre of mass vanishes (`static_balance`, `test_static_balance.py`).
- MyoSuite bushing parity: peak error at most 5 % and RMS at most 2 % of the OpenSim peak on every quantity, as in section 18.
- Bushing deflection at peak load: at most 3 mm and 2 degrees. This is a flag, never a tuning target.

### Results

MyoSuite bushing against OpenSim (same input as section 18), worst quantity, peak / RMS:

| Club | Worst peak error | Worst RMS error | Wall time |
| --- | --- | --- | --- |
| Driver | 0.011 % (squeeze) | 0.006 % | 107 s |
| 7-iron | 0.013 % (squeeze) | 0.005 % | 113 s |

All ten quantities pass for both clubs, with no tolerance changed. The MyoSuite physics is the MuJoCo section 18 code, so agreement with MuJoCo is expected; the result confirms the MyoSuite model and runtime path.

Quasi-static balance (MuJoCo contact, hands held 0.2 s at swing times covering address to the downswing): the settled force residual is at most $3 \times 10^{-7}$ of the weight and the moment residual at most $1.6 \times 10^{-7}$ of the gravity moment about the hands. Axial slip during the hold is at most 0.016 mm.

Full swing, peak values (OpenSim bushing / MuJoCo contact):

| Quantity | Driver | 7-iron |
| --- | --- | --- |
| Left hand force | 515 / 2086 N | 533 / 1667 N |
| Right hand force | 564 / 2118 N | 584 / 1748 N |
| Net force at midpoint | 439.3 / 438.0 N | 456.7 / 452.5 N |
| Internal force | 510 / 2100 N | 524 / 1697 N |
| Axial squeeze (internal) | 3.3 / 763 N | 3.9 / 438 N |
| Couple at midpoint | 99.5 / 76.3 N m | 98.6 / 92.3 N m |
| Pad normal force sum (per hand) | n/a / 29.4 kN | n/a / 28.8 kN |
| Hand-to-club deflection | 0.56 mm, 0.84 deg / 1.61 mm, 0.30 deg | 0.58 mm, 0.84 deg / 1.10 mm, 0.12 deg |
| Peak axial slip, roll slip | n/a / 1.51 mm, 5.2 mrad | n/a / 0.97 mm, 2.1 mrad |

The bushing deflections (0.56 mm and 0.84 degrees; 0.58 mm and 0.84 degrees) and the contact hand-to-club displacements are all inside the 3 mm and 2 degree flags. Nothing is flagged and nothing was tuned.

### Indeterminacy Report

Two hands holding one rigid club are statically indeterminate: the net wrench is fixed by the club motion, the split between the hands and the internal pair is not. Each grip model resolves it differently:

| Model | Driver, L / R peak force | Lead share at peak net force | Basis |
| --- | --- | --- | --- |
| Weld (min-norm proxy) | 1176 / 1234 N | 0.47 | Minimum-norm split of the OpenSim net wrench with zero free torques; a labelled proxy, not a model |
| OpenSim bushing | 515 / 564 N | 0.43 | Spring split; free torques are 50/50 by construction |
| MuJoCo contact | 2086 / 2118 N | 0.48 | Friction and pad stiffness |

For the 7-iron the shares are 0.49 (weld proxy), 0.50 (bushing) and 0.50 (contact). The net force agrees across models to within 0.3 % (driver) and 0.9 % (7-iron). The per-hand forces and the internal pair do not: the contact grip carries 4 times the per-hand peak of the bushing, with an internal force of about 2100 N against 510 N, and its couple at the midpoint is 23 % below the bushing's for the driver and 6 % below for the 7-iron.

Reading of the difference, as a flag and not a conclusion: the bushing carries axial force and twist about the grip through springs, and the contact grip through friction only, which needs normal force. The pad normal force sum reaches 29 kN per hand against the 1.1 kN squeeze the demand estimate gives. Window diagnostics on the driver, 1.2 to 1.42 s, showed the peak is not a pure solver artefact: lowering `impratio` from 10 to 1 reduces the per-hand peak from 2967 N to 2434 N, and a pyramidal cone loses contact (slip 317 mm). The remaining inflation is not explained, so neither the contact internal force nor the squeeze should be read as measured grip squeeze. Which model matches a real hand is not decided by this work; only measured grip pressure would.

### Why the Contact Grip Carries 4 Times the Bushing's Per-Hand Force (#11986)

The issue's hypothesis was that the prescribed hand-to-hand relative pose drifts over the swing and the stiff friction resolves the drift as internal force. Measured, the first half is false and the second half does not matter.

**Drift of the prescribed input** (`run_grip_hand_drift.py`, trail frame in the lead frame, every 2 ms sample, change against sample 0): both hand frames are grip frames carried by the same weld club pose (`hand_frame_states`), so the pair is rigid by construction.

| Club | Along the grip | Across the grip | Rotation | Hand spacing (80.32 mm) change |
| --- | --- | --- | --- | --- |
| Driver | 4.2e-13 mm | 3.6e-13 mm | 0 deg (round-off) | 3.5e-13 mm |
| 7-iron | 3.9e-13 mm | 2.8e-13 mm | 0 deg (round-off) | 3.9e-13 mm |

So projecting the trail hand onto the lead hand's pose plus the nominal offset is the identity, and a "rigid-consistent input" is what the run already used. The helper is `grip_contact.hand_drift.hand_relative_drift`.

**Diagnostics on the same full swing** (MuJoCo contact, Euler, dt = 1e-5 s, peak values; driver / 7-iron; ControlTower, about 1 min per run). Bushing is the OpenSim reference of section 16.

| Run | Peak L / R hand force (N) | Peak net (N) | Peak internal (N) | Peak axial squeeze (N) | Pad normal sum per hand (kN) | Peak axial slip (mm) |
| --- | --- | --- | --- | --- | --- | --- |
| OpenSim bushing | 515 / 564 and 533 / 584 | 439 and 457 | 510 and 524 | 3 and 4 | n/a | n/a |
| Contact, baseline `solreffriction` 2·dt | 2086 / 2118 and 1667 / 1748 | 438 and 453 | 2100 and 1697 | 763 and 438 | 29.4 and 28.8 | 1.51 and 0.97 |
| Lead hand only (trail pads removed) | 448 / 0 and 465 / 0 | 448 and 465 | n/a (one hand) | n/a | 35.5 and 30.4 | 1.59 and 1.04 |
| `solreffriction` 1e-3 s | 8415 / 8507 and 11124 / 11572 | 507 and 2254 | 8442 and 11348 | 3688 and 10665 | 118.6 and 70.2 | grip lost (metres) |
| `solreffriction` 5e-3 s | 6510 / 6645 and 7252 / 7448 | 239 and 884 | 6578 and 7350 | 6537 and 6081 | 16.1 and 22.5 | grip lost (metres) |
| Trail hand follows the club | 66241 / 65833 and 62190 / 61746 | 568 and 741 | 66037 and 61967 | 65273 and 61107 | 97.7 and 98.7 | 5.4 and 3.6 |
| Trail shift 0.001 mm along the grip | 2095 / 2126 and 1687 / 1786 | 438 and 453 | 2108 and 1724 | 772 and 444 | 29.4 and 28.8 | 1.51 and 0.97 |
| Trail shift 0.01 mm along the grip | 2544 / 2531 and 1974 / 2075 | 438 and 452 | 2535 and 2016 | 803 and 497 | 29.8 and 28.7 | 1.51 and 0.96 |
| Trail shift 0.1 mm along the grip | 4726 / 4784 and 4914 / 4904 | 438 and 444 | 4754 and 4905 | 1488 and 1105 | 31.4 and 31.2 | 1.50 and 0.96 |

The across-grip shifts (0.01 mm) give internal 2576 / 2178 N (driver, y / z) and 2025 / 1860 N (7-iron), with squeeze up to 1571 N. Full numbers are in `contact/internal_force_<club>.json`.

**Mechanism verdict.**

1. The 4x is not caused by input drift. The prescribed relative pose is rigid to 4e-13 mm, nine orders of magnitude below the micron-level mismatch that matters (next point).
2. The two-hand contact grip is a hyperstatic loop of two very stiff rings. A deliberate trail-hand shift shows its sensitivity: 0.01 mm along the grip adds 435 N (driver) and 319 N (7-iron) of internal force, and 0.1 mm adds 2650 N and 3210 N, about 26 to 44 N per micron against the bushing's roughly 1 N per micron (510 N for 0.56 mm). The baseline internal force of about 2100 N therefore corresponds to an effective mismatch of tens of microns, which comes from inside the contact model (pad ring preload and per-pad calibration, and the 1.5 mm axial and 5 mrad roll friction creep that the two rings take up differently), not from the input. It is a redundant force fixed by relative compliance, so by the indeterminacy above it is not a measured squeeze.
3. The net wrench is not affected: the net force stays within 0.3 % (driver) of the bushing in the baseline and shift runs, and with the lead hand alone the club is held with a single-hand peak of 448 N and 465 N, equal to the net force. The 4x is therefore the internal pair only.
4. The friction stiffness is not a usable lever. The default 2·dt is the stiffest time constant MuJoCo accepts. Softening it to 1e-3 s or 5e-3 s makes the club slip out of the hands (slip of metres) at this squeeze and timestep, so those rows are not a valid grip and their forces are not comparable. Nothing was changed in the default.
5. The 29 kN pad normal sum is not a two-hand effect: a single hand already reaches 35 kN. It is MuJoCo's cone force, set by the friction demand and the impedance, not by pad penetration, so it should not be read as grip pressure.

**Failed attempts, kept.**

- Trail hand follows the club (its pose and velocity set from the club state at each step): unstable, 66 kN internal force. A one-step-delayed stiff friction coupling between the two bodies makes it a numerical feedback, not a physical grip. It cannot stand for "constraining the other hand".
- Softer `solreffriction`: loses the grip (above).
- Projecting the trail hand onto the lead frame: identity (above), so no separate run.

Acceptance read-out for #11986: the mechanism is identified with numbers (drift and sensitivity); the drift does not explain the result, so there is no rigid-consistent re-run to report beyond the controls above; no tolerance and no default changed. The per-hand force difference remains an indeterminacy property of the two-ring contact model.

### OpenSim Contact Grip (OSV-7 Phase 4, #11739)

**Model.** The contact variant is `grip_model="contact"` in `export_full_body_osim(..., grip_contact=ContactGripConfig(...))`; the weld stays the default. It reuses the bushing topology (free club with `ClubFree*` coordinates, hands prescribed) and replaces the bushings with the shared pad layout: 2 rings of 6 pads per hand, pad radius 8 mm, grip radius 12.7 mm, axial offsets $\pm 0.04$ m. Each pad is a `ContactSphere` (`grip_pad_<Side><nn>`) on the hand body, each hand has a closed `ContactMesh` cylinder (`grip_mesh_<Side>`, 96 segments, 1 mm ring pitch) on the club, and each pad has its own `ElasticFoundationForce` (`grip_contact_<Side><nn>`), so per-pad normal force is observable. The right hand is tied to the left hand body, as in the bushing; right pads are re-parented to `LGrip` with the tie transform. Meshes pass `validate_closed_mesh`, which raises `ValueError` for open, non-manifold or inward-wound meshes.

**Parameters (matched to the shared pad law, not fitted).** OpenSim's elastic foundation is a Winkler law, $F = k_\mathrm{ef}\,\pi\,\delta^2/\sqrt{AB}$ with $A = 1/R_p + 1/R_g$, $B = 1/R_p$, so it is quadratic in penetration where the shared law is linear. Documented difference, handled by matching at the operating point: the preload interference is $\delta_0 = 2 f_\mathrm{pad}/k_\mathrm{pad}$ (twice the shared preload) so that both the squeeze and the tangent stiffness $dF/d\delta = 2F/\delta_0 = k_\mathrm{pad}$ equal the shared values, and $k_\mathrm{ef} = f_\mathrm{pad}/(\mathrm{factor}\,\delta_0^2)$. The real mesh gives about 0.89 of the analytic value, so one static single-pad evaluation against the actual mesh rescales $k_\mathrm{ef}$ (calibration, not tuning against the swing). Dissipation, static and dynamic friction (0.9 / 0.7) and the 1 mm/s transition speed are the shared values. Driver: squeeze 1104 N per hand, $k_\mathrm{pad} = 1.71 \times 10^5$ N/m, $k_\mathrm{ef} = 4.53 \times 10^9$ N/m$^3$, $\delta_0 = 1.08$ mm; the 7-iron squeeze is 1155 N. Both squeezes are derived from the same swing's bushing demand.

**Numerical method.** Explicit Runge-Kutta-Merson is infeasible: the regularised-Coulomb friction acts as a damper of about $\mu N/v_t \approx 8 \times 10^4$ N s/m per pad, which forces steps near $10^{-7}$ s (a 20 ms hold exceeded a 120 s budget at $t = 4$ ms). CPodes (implicit BDF) completes the same 20 ms hold in 12.7 s. Full swings use CPodes. The club is released with the weld velocity (hand velocity carried to the club origin; the free joint's rotational speeds are the ground-frame angular velocity, checked by a postcondition and a test), because releasing at rest against moving hands produced a 1328 N impulse at $t = 0$.

**Verification (tests first).** `tests/unit/grip_contact/test_grip_mesh.py`, `test_elastic_foundation.py`, `test_opensim_contact_grip.py` and the export test cover: closed-mesh validation, a single pad carrying the matched squeeze, tangent stiffness within 10 % of the shared one, no force without contact, a pad on a fixed grip supporting the expected load, a static hold summing to the club weight within 1 % with moment residual and squeeze checks, friction preventing sliding (and a near-frictionless grip slipping), and the weld-velocity release.

**Results (CPodes, accuracy $10^{-8}$; peak values, OpenSim contact / OpenSim bushing).**

| Quantity | Driver | 7-iron |
| --- | --- | --- |
| Net force at midpoint | 428.0 / 439.3 N (-2.6 %) | 442.5 / 456.7 N (-3.1 %) |
| Net force RMS error over the swing, relative to bushing RMS | 7.1 N (7.4 %) | 4.3 N (4.4 %) |
| Left hand force | 697 / 515 N | 603 / 533 N |
| Right hand force | 804 / 564 N | 733 / 584 N |
| Internal force | 728 / 510 N | 638 / 524 N |
| Axial squeeze (internal) | 151 / 3.3 N | 93 / 3.9 N |
| Couple at midpoint | 129 / 99.5 N m | 119 / 98.6 N m |
| Lead share at peak net | 0.51 / 0.43 | 0.43 / 0.50 |
| Pad normal force sum (per hand, peak) | 1365 N | 1353 N |
| Hand-to-club deflection | 0.59 mm, 0.75 deg | 0.44 mm, 0.62 deg |
| Peak axial slip, roll slip | 0.38 mm, 8.7 mrad | 0.18 mm, 2.3 mrad |
| Wall time (ControlTower) | 355 s | 404 s |

Convergence: driver accuracy $10^{-4}$ gives a net-force RMS error of 131 N (spikes, not converged), $10^{-6}$ gives 10.7 N and $10^{-8}$ gives 7.1 N; peak net force is 428.04 N at $10^{-6}$ and 428.00 N at $10^{-8}$.

**Internal pair, read honestly.** The net force, which the club motion fixes, matches the bushing to within about 3 % at the peak. The per-hand split and the internal force do not match and are not expected to: the two-hand contact is hyperstatic (see the indeterminacy report and the #11986 analysis above). The OpenSim contact grip carries 1.3 to 1.4 times the bushing's per-hand peak and an internal force 1.2 to 1.4 times larger, far below the MuJoCo contact grip (about 4 times the bushing), and its pad normal force sum stays near the squeeze (1.4 kN per hand against 29 kN in MuJoCo). The lead share at peak net force differs between models, which is the indeterminacy and not a measurement. None of these internal numbers is a measured grip squeeze.

**Differences from the shared law.** Quadratic force-penetration relation (matched at the operating point), mesh-based normal direction, a different contact solver (implicit CPodes with an elastic foundation rather than MuJoCo's soft constraints), and a calibrated stiffness factor.

**Failed experiments.** Hunt-Crossley sphere-sphere, sphere-cylinder and sphere-ellipsoid pairs gave no force; open meshes throw at construction; explicit RK-Merson (above); a rest release (1328 N impulse); accuracy $10^{-4}$ (net-force spikes).

**Reproduce.** On a heavy host, one run at a time: `PYTHONPATH=.:src python3 docs/development/full_body_models/evidence/grip_kinetics/run_grip_contact_opensim.py --club driver --accuracy 1e-8 --tag _acc1e-8` (likewise `--club iron7`). Timing: `bench_opensim_contact_integrators.py --method CPodes --accuracy 1e-8`. Evidence: `contact/opensim_<club>_acc1e-8_{series.npz,series.contact.npz,summary.json}`, `contact/opensim_driver_acc1e-6_*` and `contact/integrator_bench_*.json`.

### Limitations

- Software correctness only. Stiffness, damping, friction, pad count and radius are engineering defaults with derived matching, not measured values.
- Contact exists in MuJoCo and OpenSim (full swings), and Drake and Pinocchio (holds, 0.05 s). A full-swing Drake or Pinocchio contact run was not attempted: Drake's implicit Euler and Pinocchio's Radau at these stiffnesses cost hours per swing on the shared host.
- The cross-engine contact comparison is the hold acceptance (weight within 1 %, slip), not a full-swing comparison.
- The contact grip has no distributed ball (palm) of the hand and no finger geometry beyond the two pad rings; all pads are rigid-hand-anchored and the hand is prescribed.
- The squeeze for the contact grip is derived from the OpenSim bushing demand of the same swing, so it inherits that model's indeterminacy.

### What Was Tried and Rejected

- Hunt-Crossley sphere-sphere and sphere-cylinder contact pairs, as tried in the earlier phase 3 probes, gave no force for the pad geometry, so the Drake contact uses point contact between pad spheres and a primitive cylinder with the shared Hunt-Crossley parameters set on the geometry.
- Open (non-closed) meshes failed as contact geometry; an elastic-foundation contact needs a closed mesh. The OpenSim contact variant therefore builds closed capped-cylinder meshes and validates them.
- MuJoCo defaults: a single `solref` for all pads gave pad forces that did not follow $k\,\delta$ (the negative geometric stiffness was not included in $k_\mathrm{pad}$), friction rows with default `solreffriction` crept with the timestep, and a 1e-4 s step gave creep; the club also needed the weld velocity at release.
- Drake explicit RK3: step sizes near $5 \times 10^{-7}$ s, abandoned for implicit Euler.
- Pyramidal cone in MuJoCo: loses grip (slip 317 mm in the window run), kept the elliptic cone.

### Evidence Receipt

- Contact series and runs: `evidence/grip_kinetics/contact/{mujoco_<club>_series.npz, .contact.npz, _run.json}` and `validation_<club>.json`.
- MyoSuite series: `evidence/grip_kinetics/parity/myosuite_<club>_series.npz`, with `metrics_<club>.json` and `runs_<club>.json`.
- Reproduce, one heavy run at a time:
  - Contact swing: `PYTHONPATH=.:src python3 docs/development/full_body_models/evidence/grip_kinetics/run_grip_contact.py --club driver --engine mujoco`.
  - Validation: the same prefix with `run_grip_validation.py --club driver`.
  - MyoSuite: `run_grip_parity.py --club driver --engines myosuite --report`, with the MyoSuite site-packages directory appended to `PYTHONPATH`.
  - Clips: `render_grip_clip.py --club driver --variant contact`.
- Internal-force diagnostics (#11986): `contact/hand_drift_<club>.json`, `contact/internal_force_<club>.json`, and the `_ft*`, `_trail`, `_lead`, `_s[xyz]*` series in `contact/`. Reproduce: `run_grip_hand_drift.py` locally; on a heavy host `run_grip_contact.py --club driver --tag _ft5e-3 --friction-time 5e-3` (likewise `--hand-mode lead_only|trail_follows_club`, `--trail-shift-mm 0.01 0 0`), then `run_grip_internal_force_compare.py`.
- Clips (outside the repository): `~/Videos/Parity Audit/golfer_realism/grip_kinetics/<club>_{contact,bushing}_grip_hands_closeup_0p5x_60fps.mp4`, 1920 by 1080 at 60 fps, 0.5x, with per-hand force arrows from the GCV-10 overlay glyphs.
- Tests: `python3 -m pytest -n 0 tests/unit/grip_contact` (add `-m slow` for the Drake and Pinocchio holds).
