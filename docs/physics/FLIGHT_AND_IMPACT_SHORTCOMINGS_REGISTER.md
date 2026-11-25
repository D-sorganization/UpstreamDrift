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
  ball flight GUI (`src/tools/ball_flight_gui/gui.py::_build_environment`,
  passing the environment to `BallFlightSimulator` via
  `src/tools/ball_flight_gui/gui.py::_compute_trajectory`). The native viewer
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
- **Effect:** both wedge reference rows fall short in carry; each is a
  **strict expected failure** (`xfail(strict=True)`) in
  `tests/unit/physics/test_reference_shots_15443.py`. Numbers below are on
  the kernel with the Choi crisis refit (physics revision 3); "Before" is
  revision 2.

  | Row | Metric | Reference | Before | Model | Band | Result |
  | --- | --- | --- | --- | --- | --- | --- |
  | PGA driver | carry yd | 275 | 265.5 | 266.3 | 261.2 to 288.8 | pass |
  | LPGA driver | carry yd | 218 | 216.5 | 220.2 | 207.1 to 228.9 | pass |
  | PGA 5-iron | carry yd | 194 | 197.4 | 198.5 | 184.3 to 203.7 | pass |
  | PGA 6-iron | carry yd | 183 | 184.5 | 185.4 | 173.8 to 192.2 | pass |
  | PGA 7-iron | carry yd | 172 | 168.0 | 169.0 | 163.4 to 180.6 | pass |
  | PGA 8-iron | carry yd | 160 | 155.6 | 156.4 | 152.0 to 168.0 | pass |
  | PGA 9-iron | carry yd | 148 | 142.4 | 143.0 | 140.6 to 155.4 | pass |
  | PGA wedge | carry yd | 136 | 127.3 | 127.5 | 129.2 to 142.8 | strict xfail |
  | LPGA 7-iron | carry yd | 141 | 137.9 | 139.3 | 133.9 to 148.1 | pass |
  | LPGA wedge | carry yd | 107 | 99.5 | 99.3 | 101.7 to 112.3 | strict xfail |
  | LPGA wedge | apex yd | 23 | 19.3 | 19.8 | 19.6 to 26.4 | pass (was xfail) |

  Apex and descent pass on every row. The refit steepens every iron and
  wedge descent by 1.7 to 2.4 degrees toward the reference (5-iron 45.3 to
  47.7 against 49, 6-iron 46.6 to 49.1 against 50, PGA wedge 51.7 to 53.6
  against 52) and moves the 5-iron descent and LPGA 7-iron apex off their
  band edges. The LPGA wedge apex now passes by 0.2 yd. The carry error
  still grows with spin and falls with ball speed, consistent with the
  spin-drag extrapolation.
- **Tracking:** #15443 (F1).
- **F1 investigation:** wind-tunnel points, digitised with
  their sources, are in `tests/unit/physics/wind_tunnel_drag_15443.json`. The
  drag laws below were fitted to those points only; the five launch-monitor
  rows were the held-out test. Launch spin ratios run from 0.08 (driver) to
  0.46 (wedge), so the wedge flies above the measured range from launch on.

| Re | Choi 2006 Cd | Current Cd at S = 0 | Fitted to Choi |
| --- | --- | --- | --- |
| 5e4 | 0.483 | 0.459 | 0.503 |
| 7e4 | 0.428 | 0.351 | 0.428 |
| 9e4 | 0.214 | 0.263 | 0.227 |
| 1e5 | 0.204 | 0.240 | 0.208 |
| 2.2e5 | 0.230 | 0.210 | 0.230 |

  The current crisis is too broad (rms 0.034 against Choi). Refitting its
  centre, width, floor and recovery centre to Choi gives centre 7.59e4,
  width 0.065 decade, floor 0.187 and recovery centre 2.81e5 (rms 0.008),
  and it matches the Lyu 2018 check (0.50 at 5e4, 0.21 at 1e5). The Bearman
  and Harvey spin data fit a linear spin term of 0.159 through their
  zero-spin value, 0.237 with a free intercept, or a hinge of 0.299 above
  S 0.088 (rms 0.0008). Held-out carries (yd) with the Choi-fitted
  Reynolds curve:

| Spin term | PGA driver | LPGA driver | 6-iron | 7-iron | Wedge | Rows out |
| --- | --- | --- | --- | --- | --- | --- |
| Current law, 0.18 S | 265.5 | 216.5 | 190.1 | 168.0 | 127.3 | wedge carry |
| 0.18 S | 266.3 | 220.2 | 190.7 | 169.0 | 127.5 | wedge carry |
| 0.159 S | 268.3 | 221.8 | 193.9 | 172.1 | 130.2 | 6-iron carry |
| 0.237 S, intercept | 272.7 | 224.6 | 188.6 | 165.9 | 123.4 | wedge carry |
| Hinge 0.299 | 276.7 | 227.4 | 184.9 | 161.5 | 118.6 | 7-iron, wedge carry |
| Reference | 275 | 218 | 183 | 172 | 136 | bands 5 % |

  **Result:** no drag law fitted to the wind-tunnel data clears the wedge
  carry band without pushing another row out. The 6-iron (4 % long) needs
  more high-spin drag and the wedge (6 % short) needs less, and the measured
  spin data rise more steeply above S 0.1, not less. Holding the spin term
  flat above S 0.3, the end of the measured range, puts the 6-iron 6 % long.
  Swapping in the Bearman and Harvey lift curve makes the drivers and the
  wedge worse. The wedge row stays a strict expected failure. The Choi crisis
  fit keeps the other 14 cases inside their bands and moves the iron and
  wedge descents closer to the reference (6-iron 47.8 to 50.3 degrees); it
  is now applied (below).
  Remaining suspects: the 6-iron and wedge rows are recorded from memory,
  and lift and drag above S 0.3 are unmeasured in these sources.
- **F1 refit applied:** the Rust kernel and the Python fallback use the
  Choi-fitted crisis (centre 7.59e4, width 0.065 decade, floor 0.187,
  recovery centre 2.81e5; rms 0.008 against 0.034 before) with the spin term
  unchanged at 0.18 S. `tests/unit/physics/test_drag_crisis_fit_15443.py`
  re-derives the fit from the wind-tunnel fixture and holds the named
  constants to it, with the Lyu 2018 values as a held-out check. The wedge
  carries remain strict expected failures.

## Rollout and Reference Coverage

### Rollout Terrain

- **Missing:** terrain. The flight ends at the first crossing of the plane
  `z = 0` (kernel `terminate` closure in
  `rust_core/upstream-physics/src/ball_flight.rs`), and the exporter reports
  carry only: `FlightSummary.total_m` is never set (`TOTAL_UNAVAILABLE` in
  `src/shared/python/impact_parameters/flight_report.py`).
- **Instead:** `SlopeAwareGround` in
  `src/shared/python/sg_optimizer/shot_model/ground_runout.py` (model
  `sg_runout_v2`) gives one hop plus roll. It is used by the dispersion tables
  (`dispersion_table.py`) and the native viewer roll
  (`src/tools/native_viewer_export/rollout.py`).
  - **Hop:** spin-aware since #15443. It is the first bounce of Penner, "The
    run of a golf ball", Can. J. Phys. 80, 931-940 (2002), eqs. (2)-(8): the
    turf acts as a rigid surface tilted by `theta_c`, and the ball either rolls
    out (`vx' = 5/7 vx' - 2/7 r w`) or slides, by the critical friction of
    eq. (4). The hop is the drag-free range of the rebound, so more landing
    backspin gives a shorter hop, or a backward one. The landing backspin is
    the launch spin decayed at the kernel rate.
  - **Coefficients:** in `src/config/runout_bounce.json` with their equation
    numbers. `theta_c` (eq. 8) was fitted to a single green impact (Haake),
    `e` (eq. 5) was measured on soft turf, and `mu = 0.40` is Daish's
    unmeasured green value. The same bounce applies to every lie.
  - **Paper discrepancy:** at Penner's Haake check point, eqs. (3), (6) and
    (7) as printed reproduce his `mu_c = 0.29` and `e = 0.14`. They do not
    reproduce his quoted rebound of 4.34 m/s at 34.1 deg: they give 4.16 m/s
    at 41.7 deg. His quote needs a normal rebound of 2.82 m/s, which eq. (5)
    cannot give at any normal speed up to 20 m/s (maximum 2.42 m/s), so the
    figure is inconsistent with his own equations; an independent re-derivation
    found no reading that matches within 2 %. The equations do reproduce his
    quoted drive and 9-iron `mu_c` and bounce heights, which
    `test_ground_runout.py` uses as the primary validation; the Haake case is
    pinned there as computed.
  - **Roll:** still fixed gains, with a per-skill multiplier standing in for
    spin, a lie multiplier and a stance-bin slope factor. These parameters are
    marked "assumption, unmeasured turf, no primary citation".
  - **Not modelled:** undulation, firmness, green, hazards, later bounces and
    spin-dependent check in the roll.
- **Effect:** unknown. Tour wedge run/carry still exceeds the short iron's (a
  strict expected failure in `test_ground_runout.py`). `PipelineLandingSimulator`
  runs the pipeline without a club, so every skill's launch spin is clamped to
  the 3500 rpm driver envelope. That leaves the wedge too little landing
  backspin for the spin-aware hop to shorten it.
- **Tracking:** #15443 (child 6).

### Launch-Monitor Reference Coverage

- **Missing:** reference data beyond ten tour-average rows.
- **Instead:** `tests/unit/physics/reference_shots_15443.json` holds PGA driver,
  5-iron to pitching wedge and LPGA driver, 7-iron and pitching wedge, all read
  from the published Trackman tour-average tables (citation, access date and
  URLs are in its `tour_table_source` block); it bands carry, apex and descent
  only. There are no woods, hybrids, long irons or lob
  wedges, no side or spin-axis data, no roll or total reference, and no
  impact-model reference table (club delivery to ball data). The older driver
  benchmarks in `tests/unit/physics/test_trackman_benchmarks.py` and
  `tests/unit/test_flight_models.py` now read the PGA driver row and its
  carry and apex bands through `tests/unit/physics/reference_bands.py` (F7);
  the 7-iron benchmark keeps its wide 148 to 182 yd band. Against the driver
  band (261.3 to 288.8 yd) three alternative flight models overshoot and are
  strict xfails with their measured numbers: Nathan 294.3 yd (apex 34.4 m),
  Ballantyne 297.2 yd (apex 33.7 m) and Rospie DL 290.8 yd.
- **Effect:** not applicable; this limits what can be detected.
- **Tracking:** #15443 (children 2, 3, F7).
