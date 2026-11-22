# Flight and Impact Shortcomings Register

Current-state list of what the swing, impact, ball-flight and rollout pipeline
does **not** model (epic #15443, child 10). Keep it current: a PR that closes
or adds a simplification below must edit its entry in the same change.

Pipeline: `SwingBallFlightPipeline` (`src/shared/python/physics/swing_ball_flight_pipeline.py`)
solves impact with `ImpactSolverAPI`, then integrates the flight with the Rust
kernel (`rust_core/upstream-physics/src/ball_flight.rs::simulate_ball_trajectory`).
`flight_for_impact` (`src/shared/python/impact_parameters/flight_report.py`) is
the entry the exporters use.

Each entry gives what is missing, what the code does instead, the size of the
effect on carry, side or spin, and the tracking issue. "Unknown" means no
published figure is recorded in this repository; no number here is invented.
Sizes quoted from tests are the band a test asserts, not a published value.

## Impact Model

### Ball Compression and Gear Effect Detail

- **Missing:** a compliant ball and face (compression, contact duration, spin
  from face-ball slip history) and a coupled head-ball angular momentum solve.
- **Instead:** `RigidBodyImpactModel.solve` in
  `src/shared/python/physics/impact_model/models.py` applies one instantaneous
  impulse with a fixed COR and a capped Coulomb friction impulse
  (`ImpactParameters.friction_coefficient = 0.4`). Gear effect is an additive
  heuristic, `compute_gear_effect_spin` in
  `src/shared/python/physics/impact_model/utils.py`, called from
  `ImpactSolverAPI.solve_pre_impact_state`; its own docstring states it is not
  equipment calibration. It runs only when an impact offset is supplied.
- **Effect:** unknown. Epic #15443 child 1 measured spin about 20 to 40 % high
  per degree of spin loft (PR #15450).
- **Tracking:** #15443 (F2, F5).
- **Spring-damper model (F5, fixed):** `SpringDamperImpactModel` gave smash
  0.82 and zero spin because its fixed damping of 1e3 N s/m was over-damped
  (damping ratio about 2.6, COR near 0) and it applied no tangential impulse.
  Damping is now derived from `params.cor` (`kelvin_voigt_damping_for_cor`) and
  one capped Coulomb friction impulse is shared with the rigid-body model.
  Driver strike: smash 1.469 against 1.464 for rigid body, identical spin. It
  stays in the choices. Remaining limits: the tangent plane is not
  time-resolved, stiffness 1e6 N/m is uncalibrated, and
  `params.contact_damping` is ignored.

### Face Bulge and Roll

- **Missing:** face curvature, so a toe or heel strike does not change the
  effective loft and face angle at the contact point.
- **Instead:** one flat face normal per sample (`clubhead_orientation` in
  `PreImpactState`, built by `swing_state_from_series` in
  `src/shared/python/impact_parameters/flight_report.py`). No bulge or roll
  term exists in `src/shared/python/physics` or `impact_parameters`.
- **Effect:** unknown.
- **Tracking:** none.

### Head Angular Velocity at Impact

- **Missing:** the head's rotation rate (for example shaft-lead release) in the
  impact solve.
- **Instead:** `swing_state_from_series` sets
  `clubhead_angular_velocity=np.zeros(3)`.
- **Effect:** unknown.
- **Tracking:** none; related to the shaft entry below.

### Iron COR Source

- **Missing:** a sourced iron-face COR.
- **Instead:** `cor_by_club` in
  `src/shared/python/impact_parameters/club_impact_settings.json` holds 0.83 for
  `iron7`, the driver value (`DRIVER_COR` in
  `src/shared/python/core/physics_constants.py`, a USGA/R&A limit). The file
  records that no source for a lower iron value was found.
- **Effect:** unknown; it scales iron ball speed directly.
- **Tracking:** #15443 ("Sourced iron COR and calibrated gear effect").

### Shaft Bend in the Delivery

- **Missing:** shaft lead, droop and toe-down reaching the face.
- **Instead:** the matched bundle fits a rigid club; the impact solve sees one
  head velocity and face normal. The flexible chain shaft state is read back for
  reports only (`src/shared/python/impact_parameters/shaft_state.py`).
- **Effect:** in one capture the rigid fit lost about 2.6 degrees of spin loft
  (epic #15443 delivery audit); a single capture, not a population figure.
- **Tracking:** #15443 (F8, F9, child 8); #15150 Track C.

### Turf Interaction at Impact

- **Missing:** club-turf contact (divot, speed loss, spin loss) in the
  swing-to-flight path.
- **Instead:** the ball starts at the origin on the ground plane with no turf
  (`BallFlightSimulator._simulate_native`, `pos0 = [0, 0, 0]`). A lie penalty
  exists as `lie_perturbation` in
  `src/shared/python/physics/lie_interaction.py`, but only the shot-model layer
  (`src/shared/python/sg_optimizer/shot_model/dispersion_lie.py`) uses it. Its
  docstring says every value is an assumption.
- **Effect:** unknown.
- **Tracking:** none.

## Ball Flight

### Wind

- **Missing:** gusts, shear with height and turbulence.
- **Instead:** one steady uniform vector in `EnvironmentalConditions.wind_velocity`
  (`src/shared/python/physics/ball_launch_conditions.py`), subtracted from ball
  velocity in the kernel derivative. The pipeline default is calm, and
  `SwingBallFlightPipeline` also accepts an explicit environment. Two
  consumers set wind: the shot-model landing simulator
  (`PipelineLandingSimulator._pipeline` in
  `src/shared/python/sg_optimizer/shot_model/physics_simulator.py`) and the
  ball flight GUI (`_build_environment` in `src/tools/ball_flight_gui/gui.py`,
  passed to `BallFlightSimulator` by `_compute_trajectory`). The native viewer
  export flight runs calm; it has no wind setting.
- **Effect:** a 5 m/s steady head wind removes 9 to 18 yd of driver carry, the
  band asserted by `test_five_meter_per_second_wind_changes_driver_carry_sensibly`
  in `tests/unit/physics/test_trackman_benchmarks.py`; not a published figure.
  Gust effect: unknown.
- **Tracking:** none.

### Spin Decay Model

- **Missing:** a decay law fitted to measured spin, and spin-axis migration.
- **Instead:** `d(omega)/dt = -rate * omega` with a constant rate of 0.05 per
  second (`SPIN_DECAY_RATE_S` in `src/shared/python/core/physics_constants.py`,
  used through `BallProperties.spin_decay_rate`); the axis is held constant
  (kernel docstring in `rust_core/upstream-physics/src/ball_flight.rs`). The
  constant is labelled Trackman-derived but no fitting data is bundled
  (`docs/physics/GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md`). The registry models in
  `src/shared/python/physics/flight_models.py` use their own constants of 0.02
  to 0.05 per second.
- **Effect:** unknown.
- **Tracking:** none.

### Flight Model Registry Names

- **Missing:** the published models. `FlightModelRegistry` in
  `src/shared/python/physics/flight_models.py` lists Nathan, Ballantyne,
  J. Cole, Rospie DL and Charry L3, but each is a constant Cd/Cl preset
  (`ConstantCoefficientModel`) with illustrative coefficients, not the
  author's model. `MacDonaldHanzelyModel` is likewise a constant Cd/Cl ODE with
  exponential spin decay whose coefficients are not taken from the 1991 paper.
  Waterloo/Penner is a Penner lift fit with a constant-spin TrackMan
  calibration of `cd1` and the lift cap.
- **Instead:** names are kept for API and UI compatibility (#15443 F6); the
  `description` of each model now says it is not the published model and what
  it implements.
- **Effect:** unknown; carry differences between presets come from the chosen
  constants, not from the named authors' data.
- **Tracking:** #15443 (F6).

### Altitude and Air Density

- **Missing:** density change along the trajectory, non-standard lapse, and
  temperature-dependent viscosity in the native kernel.
- **Instead:** density is one constant, 1.225 kg per cubic metre by default
  (`EnvironmentalConditions`); `from_altitude` gives an ISA value but the Rust
  path never re-evaluates it in flight. Along-trajectory tracking exists only in
  the Python `EnhancedBallFlightSimulator` (`track_altitude_density`, default off).
  Viscosity is fixed at 1.81e-5 in
  `BallFlightSimulator._simulate_native`
  (`src/shared/python/physics/ball_simulator.py`).
- **Effect:** unknown for carry. A test asserts humid air is 1 to 2 % less dense
  than dry air at 30 C (`test_humid_air_is_less_dense_than_dry_air`).
- **Tracking:** none.

### Drag Law Range

- **Missing:** validity of the spin-drag term at high spin ratio.
- **Instead:** `cd_spinning_ball` in
  `rust_core/upstream-physics/src/aerodynamics.rs` adds `0.18 * S` (Smits and
  Smith 1994), extrapolated beyond the S <= 0.3 wind-tunnel range for wedges.
- **Effect:** both wedge reference rows fall short; each is a **strict
  expected failure** (`xfail(strict=True)`) in
  `tests/unit/physics/test_reference_shots_15443.py`. The 6-iron and 7-iron
  rows that earlier disagreed by about 6 % came from memory; the sourced
  rows below replace them and all irons from the 5-iron to the 9-iron pass.

  | Row | Metric | Reference | Model | Band | Result |
  | --- | --- | --- | --- | --- | --- |
  | PGA driver | carry yd | 275 | 265.5 | 261.2 to 288.8 | pass |
  | LPGA driver | carry yd | 218 | 216.5 | 207.1 to 228.9 | pass |
  | PGA 5-iron | carry yd | 194 | 197.4 | 184.3 to 203.7 | pass |
  | PGA 6-iron | carry yd | 183 | 184.5 | 173.8 to 192.2 | pass |
  | PGA 7-iron | carry yd | 172 | 168.0 | 163.4 to 180.6 | pass |
  | PGA 8-iron | carry yd | 160 | 155.6 | 152.0 to 168.0 | pass |
  | PGA 9-iron | carry yd | 148 | 142.4 | 140.6 to 155.4 | pass |
  | PGA wedge | carry yd | 136 | 127.3 | 129.2 to 142.8 | strict xfail |
  | LPGA 7-iron | carry yd | 141 | 137.9 | 133.9 to 148.1 | pass |
  | LPGA wedge | carry yd | 107 | 99.5 | 101.7 to 112.3 | strict xfail |
  | LPGA wedge | apex yd | 23 | 19.3 | 19.6 to 26.4 | strict xfail |

  Apex and descent pass on every other row (the 5-iron descent, 45.3 deg
  against a 45.0 lower bound, and the LPGA 7-iron apex, 22.3 yd against 22.1,
  are within 0.3 of the edge). The carry error grows with spin and falls with
  ball speed, consistent with the spin-drag extrapolation.
- **Tracking:** #15443 (F1).

## Rollout and Reference Coverage

### Rollout Terrain

- **Missing:** terrain. The flight ends at the first crossing of the plane
  `z = 0` (kernel `terminate` closure in
  `rust_core/upstream-physics/src/ball_flight.rs`), and the exporter reports
  carry only: `FlightSummary.total_m` is never set (`TOTAL_UNAVAILABLE` in
  `src/shared/python/impact_parameters/flight_report.py`).
- **Instead:** `SlopeAwareGround` in
  `src/shared/python/sg_optimizer/shot_model/ground_runout.py` gives one hop
  plus roll from fixed gains, a skill multiplier standing in for spin, a lie
  multiplier and a stance-bin slope factor. It is used only by the dispersion
  tables (`dispersion_table.py`). Its parameters are marked "assumption,
  unmeasured turf, no primary citation". No undulation, firmness, green,
  hazard, spin-dependent check or bounce from the actual landing spin.
- **Effect:** unknown.
- **Tracking:** #15443 (child 6).

### Launch-Monitor Reference Coverage

- **Missing:** reference data beyond ten tour-average rows.
- **Instead:** `tests/unit/physics/reference_shots_15443.json` holds PGA driver,
  5-iron to pitching wedge and LPGA driver, 7-iron and pitching wedge, all read
  from the published Trackman tour-average tables (citation, access date and
  URLs are in its `tour_table_source` block); it bands carry, apex and descent
  only. There are no woods, hybrids, long irons or lob
  wedges, no side or spin-axis data, no roll or total reference, and no
  impact-model reference table (club delivery to ball data). The older bands in
  `tests/unit/physics/test_trackman_benchmarks.py` are wide (driver 238 to 290
  yd, 7-iron 148 to 182 yd).
- **Effect:** not applicable; this limits what can be detected.
- **Tracking:** #15443 (children 2, 3, F7).
