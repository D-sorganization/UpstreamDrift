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
- **Tracking:** #15443 (F2, F5). `SpringDamperImpactModel` is listed as broken
  there (smash 0.833, zero spin); fix or removal is open.

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
- **Effect:** the pitching-wedge reference row carries 127.3 yd against 136 yd,
  minus 6.4 %, a **strict expected failure** (`xfail(strict=True)`) in
  `tests/unit/physics/test_reference_shots_15443.py`.
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

- **Missing:** reference data beyond five tour-average rows.
- **Instead:** `tests/unit/physics/reference_shots_15443.json` holds PGA driver,
  LPGA driver, 6-iron, 7-iron and pitching wedge (Trackman 2019 averages); it
  bands carry, apex and descent only, and several values are recorded as
  approximate or from memory. There are no woods, hybrids, long irons or lob
  wedges, no side or spin-axis data, no roll or total reference, and no
  impact-model reference table (club delivery to ball data). The older bands in
  `tests/unit/physics/test_trackman_benchmarks.py` are wide (driver 238 to 290
  yd, 7-iron 148 to 182 yd).
- **Effect:** not applicable; this limits what can be detected.
- **Tracking:** #15443 (children 2, 3, F7).
