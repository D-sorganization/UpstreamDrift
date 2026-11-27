# Showcase Clip Assumptions and Known Shortcomings

Assumptions behind the impact, flight, runout and dispersion pipeline that
the native viewer showcase clips use (epic #15443, child 9). Every statement
names the file, and where one exists the settings key, it comes from on
`main`. Nothing here is a measurement of a real golfer or ball; read the clips
as a model prediction. For the full per-effect register see
[`FLIGHT_AND_IMPACT_SHORTCOMINGS_REGISTER.md`](FLIGHT_AND_IMPACT_SHORTCOMINGS_REGISTER.md).
"Unknown" means the repository records no figure, and none is invented here.

## Pipeline at a Glance

1. Swing replay: the fitted body and club motion is replayed kinematically
   (`src/tools/native_viewer_export/kinematic_replay.py`).
2. Impact: `SwingBallFlightPipeline`
   (`src/shared/python/physics/swing_ball_flight_pipeline.py`) solves the
   club-ball impact with `ImpactSolverAPI`, rigid-body model by default.
3. Flight: the Rust kernel
   (`rust_core/upstream-physics/src/ball_flight.rs`) integrates drag, Magnus
   lift and spin decay. `flight_for_impact`
   (`src/shared/python/impact_parameters/flight_report.py`) is the exporter's
   entry.
4. Runout: `src/tools/native_viewer_export/rollout.py` drives the repository
   runout model `sg_runout_v2`
   (`src/shared/python/sg_optimizer/shot_model/ground_runout.py`).
5. Hole view and dispersion: `hole_view.py`, `shot_dispersion.py` and
   `dispersion_traces.py` in the same folder.

The ball is placed at the origin on a flat ground plane with no turf
(`pos0 = [0, 0, 0]` in `BallFlightSimulator._simulate_native`,
`src/shared/python/physics/ball_simulator.py`), and the flight ends at the
first crossing of `z = 0` (kernel `terminate` closure in `ball_flight.rs`).
Carry is that landing distance.

## Ball and Air Constants

| Quantity | Value | Where |
| --- | --- | --- |
| Ball mass | 0.04593 kg (USGA maximum, 1.620 oz) | `GOLF_BALL_MASS_KG`, `src/shared/python/core/physics_constants.py`; kernel default `AeroBallProperties` in `aerodynamics.rs` |
| Ball diameter | 0.04267 m (USGA minimum) in the constants; the kernel radius is 0.02135 m | `GOLF_BALL_DIAMETER_M`; `AeroBallProperties::default` |
| Ball density | not an input; only mass and radius enter | `AeroBallProperties` fields |
| Air density | 1.225 kg/m^3 (ISA sea level), one constant for the whole flight | `EnvironmentalConditions.air_density`, `src/shared/python/physics/ball_launch_conditions.py`; `AIR_DENSITY_SEA_LEVEL_KG_M3` |
| Air temperature | 15.0 C passed to the kernel | `EnvironmentalConditions.temperature`; converted by `_air_temperature_kelvin` in `ball_simulator.py` |
| Air viscosity | 1.81e-5 kg/(m s), fixed | `BallFlightSimulator._simulate_native`, `ball_simulator.py` |
| Gravity | 9.80665 m/s^2 | `GRAVITY_M_S2`, `physics_constants.py` |
| Spin decay | `d(omega)/dt = -0.05 omega` per second, axis held fixed | `SPIN_DECAY_RATE_S`; `ball_flight.rs` docstring |

The constants file also holds a "standard" temperature of 20 C
(`TEMPERATURE_C`) that the flight does not use; the flight default is the 15 C
above. The 0.05 per second decay is labelled "Trackman Data" in
`physics_constants.py`, but no fitting data is bundled
(`GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md`); treat it as an assumption.

## Drag, Lift and Spin Decay

- **Drag:** `Cd(Re, S) = Cd_Re(Re; floor) + 0.18 S`, with spin ratio
  `S = r omega / v` (`rust_core/upstream-physics/src/aerodynamics.rs`,
  `CD_SPIN_SLOPE`, source Smits and Smith 1994, Science and Golf II; the
  primary text was not reached, see `GOLF_BALL_SPIN_DRAG_SLOPE` in
  `physics_constants.py`).
- **Drag crisis:** a fit to Choi, Jeon and Choi 2006, Phys. Fluids 18, 041702,
  Fig. 1 (centre Re 7.59e4, width 0.065 decade, floor 0.187, recovery centre
  Re 2.81e5; rms 0.008), checked against Lyu et al. 2018. The subcritical Cd
  (0.50) and high-Re Cd (0.30) are held fixed from Bearman and Harvey 1976
  (`aerodynamics.rs`). The fit is re-derived in
  `tests/unit/physics/test_drag_crisis_fit_15443.py` from
  `tests/unit/physics/wind_tunnel_drag_15443.json`.
- **Lift:** a Penner power law, scale 0.70, exponent 0.645, capped at 0.26
  (`PENNER_LIFT_SCALE`, `PENNER_LIFT_EXPONENT`, `MAX_LIFT_COEFFICIENT` in
  `aerodynamics.rs`). The `cd1`, `cd2` and `cl1` fields of the Python
  `BallProperties` are not used by the kernel
  (`GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md`).
- **Validity limit:** the wind-tunnel data stop at spin ratio `S` of about 0.28
  to 0.3. Launch spin ratios run from 0.08 (driver) to 0.46 (wedge), so wedge
  flights use the spin-drag term extrapolated beyond the data
  (`docs/physics/FLIGHT_AND_IMPACT_SHORTCOMINGS_REGISTER.md`, Drag Law Range;
  reasons in the `xfail` entries of
  `tests/unit/physics/reference_shots_15443.json`).
- **Spin decay:** one constant rate for every club; no fitted law and no
  spin-axis migration (`SPIN_DECAY_RATE_S`).
- **Kernel version guard:** Python refuses a Rust kernel older than the
  expected physics revision (`PHYSICS_REVISION`; issue #15443 F11), so a stale
  local wheel cannot silently drop a drag term.

## Wind

Wind is not modelled in the showcase clips. The kernel accepts one steady
uniform vector, `EnvironmentalConditions.wind_velocity`
(`ball_launch_conditions.py`), default `[0, 0, 0]`. The native viewer export
has no wind setting and flies calm (`FLIGHT_AND_IMPACT_SHORTCOMINGS_REGISTER.md`,
Wind). Gusts, shear and turbulence do not exist anywhere in the pipeline.

## Impact Model

- **Model:** a rigid body with one instantaneous impulse, a fixed coefficient of
  restitution and a capped Coulomb friction impulse of 0.4
  (`ImpactParameters.friction_coefficient`,
  `src/shared/python/physics/impact_model/types.py`; `RigidBodyImpactModel` in
  `models.py`). No compression, contact duration, face bulge or roll.
- **Club head mass:** 0.200 kg default (`clubhead_mass` in `types.py` and
  `solver.py`). Per-club values come from the catalogue
  (`club_models.CLUBS`, which holds only the driver and `iron7`); a club
  without an entry launches with the driver defaults
  (`src/shared/python/impact_parameters/club_impact_settings.json`,
  `sources`, `pipeline_default_clubs`).
- **Club head inertia:** the reference scalar is 5e-3 kg m^2
  (`reference_head_moi_kg_m2`). Other catalogue heads scale it by the
  catalogue head `m k^2` ratio; the settings call this an approximation from
  catalogue geometry, not a measured inertia. The rigid-gear heads
  (`rigid_gear_heads`: driver inertia diagonal 0.00025, 0.00040, 0.00045 kg
  m^2 with CG depth 0.035 m; `iron7` 0.00012, 0.00020, 0.00025 kg m^2 with 0.010
  m) are, in the file's words, research assumptions, not measured properties of
  a specific head (`rigid_gear_sources`).
- **COR:** `cor_by_club` in `club_impact_settings.json`. Driver 0.830, the
  USGA/R&A limit for driver faces (intercept 0.83, slope 0). The iron line is
  `e = 0.86 - 0.0029 v_n` with `v_n` the normal approach speed in m/s
  (`iron7`), from Penner, Am. J. Phys. 69, 563 (2001), Eq. (5), a linear fit
  to Lieberman and Johnson (Science and Golf II, 1994). It is fitted over about
  37 to 50 m/s and extrapolated for iron strikes (about 25 to 35 m/s), assumes a
  rigid iron face and is for the 1994 test balls (`iron_cor_source.validity`,
  `iron_cor_source.applicability`). Other clubs get the driver law.
- **Spin:** from the impulse and friction above. The register records that
  spin ran 20 to 40 percent high per degree of spin loft in the epic's
  child 1 review (PR #15450) and that tour-delivery over-spin is a modelling
  decision still open (issue #15443, F2).
- **Gear effect:** an off-centre strike uses the rigid-contact gear law with
  the head above (`rigid_gear.py`, `rigid_gear_head_for` in
  `club_impact.py`). The flight report passes zero head angular velocity
  (`flight_report.py` docstring; `swing_state_from_series`), so rotation-driven
  spin is absent. The calibration of gear effect is open (register, Iron COR
  Source).
- **Face:** one flat face normal per sample; no bulge or roll
  (register, Face Bulge and Roll).

## Per-Club Spin Caps

Launch spin above a club-class cap is refused rather than flown
(`src/shared/python/physics/ball_flight_validity.py`,
`launch_spin_envelope`). The caps, with the sources the file records, are in
`src/shared/python/physics/ball_flight_validity_settings.json`
(`classes.<class>.max_launch_spin_rpm`):

| Class | Cap (rpm) | Recorded basis |
| --- | --- | --- |
| driver | 3500 | tour driver launch spin, tour average near 2,500 rpm |
| wood | 3500 | shares the driver envelope by decision of #15443; 3-wood tour average about 3,700 rpm, so conservative |
| hybrid | 5500 | tour average about 4,400 rpm plus about 25 percent |
| iron | 10000 | 9-iron average 8,647 rpm plus about 15 percent; long irons share the cap |
| wedge | 12500 | pitching wedge average about 9,300 rpm plus about 35 percent |

The class comes from the club name through `club_name_patterns`. A name that
matches no pattern, an empty name or no club falls back to the driver envelope
(`fallback_club_class`). Pitch, chip and bunker are not patterns, so a club
named "pitch", "chip", "bunker" or "sand" gets the 3500 rpm driver cap; only
names containing "wedge" or the abbreviations pw, gw, aw, sw, lw reach the
wedge cap (checked against the patterns in the settings file). Landing azimuth
beyond 45 degrees is also refused (`MAX_LANDING_AZIMUTH_DEG`,
`ball_flight_validity.py`), an assumption rather than a measured bound.
`PipelineLandingSimulator` runs without a club, so every skill's launch spin is
clamped to the driver envelope there (register, Rollout Terrain). PR #15757
passes each skill's representative club to the pipeline; once it lands, only
pitch, chip and bunker keep the driver envelope.

## Reference Rows and Known Strict Failures

`tests/unit/physics/reference_shots_15443.json` holds ten rows from published
Trackman PGA and LPGA tour averages (its `tour_table_source` block records the
citation and access date) with carry, apex and descent bands of about 5
percent. Strict expected failures (`xfail(strict=True)`), so a fix that makes
them pass fails the suite until promoted:

- PGA pitching wedge carry: measured 127.5 yd against 136 yd, minus 6.3
  percent (`pga_tour_pitching_wedge_2019`).
- LPGA pitching wedge carry: measured 99.3 yd against 107 yd, minus 7.2
  percent (`lpga_tour_pitching_wedge_2019`).
- Alternative flight models above the PGA driver carry band of 261.3 to 288.8
  yd: Nathan 294.3 yd, Ballantyne 297.2 yd, Rospie DL 290.8 yd
  (`_DRIVER_OVERSHOOT` in `tests/unit/test_flight_models.py`; bands read through
  `tests/unit/physics/reference_bands.py`).
- Tour wedge run to carry ratio exceeding the short iron's
  (`tests/unit/sg_optimizer/shot_model/test_ground_runout.py`, 0.0629 against
  0.0523).

The flight kernel itself passes the other rows. The reference rows do not
cover woods, hybrids, long irons, lob wedges, spin axis, roll or total, or any
impact-model table (register, Launch-Monitor Reference Coverage).

## Penner First Bounce and Runout

Model id `sg_runout_v2` (`ground_runout.py`); the exporter wraps it in
`rollout.py` with `rollout_settings.json` (`enabled`, `lie` fairway,
`slope` flat, `default_skill` driver, label "rollout: flat-fairway model").
The lie and slope are assumptions, never read from the scene.

- **Hop:** the first bounce of Penner, "The run of a golf ball", Can. J. Phys.
  80, 931-940 (2002), eqs. (2) to (8), coefficients in
  `src/config/runout_bounce.json`: `theta_c_ref_deg` 15.4 (eq. 8), `friction_mu`
  0.4, restitution `c0` 0.51, `c1` 0.0375 s/m, `c2` 0.000903 s^2/m^2 below 20 m/s
  and 0.12 above (eq. 5), ball radius 0.02135 m. The hop is the drag-free range
  of the rebound over flat ground; only the first hop is modelled.
- **Source limits:** `theta_c` is a first-order law fitted to one green impact
  (Haake); `e` was measured on relatively soft turf; `mu = 0.40` is Daish's
  unmeasured green value (`runout_bounce.json`, `limitations`). One bounce law
  serves every lie (register, Rollout Terrain).
- **Haake inconsistency:** at Penner's Haake check point eqs. (3), (6) and (7)
  reproduce his critical friction 0.29 and `e` 0.14 but give a rebound of 4.16
  m/s at 41.7 degrees, not the 4.34 m/s at 34.1 degrees he quotes. His quote
  needs a normal rebound of 2.82 m/s, which eq. (5) cannot give (maximum 2.42
  m/s up to 20 m/s). The quoted drive and 9-iron figures are reproduced and
  are the primary validation (`runout_bounce.json`, `limitations`).
- **Roll after the hop:** fixed gains marked "assumption" in
  `GROUND_PARAMETERS` (`ground_runout.py`): `roll_gain_yd_per_ms` 1.35 yd per
  m/s, `slope_roll_per_tan` 1.8, skill multipliers driver 1.0, fairway wood
  0.75, long iron 0.58, mid iron 0.45, short iron 0.30, wedge 0.18, and lie
  multipliers tee 1.0, fairway 1.0, rough 0.35, sand 0.15. No primary citation
  is recorded for them.
- **Not modelled:** undulation, firmness, green, hazards, later bounces and
  spin-dependent check in the roll (register, Rollout Terrain). Roll and
  total are labelled a model assumption on the HUD and manifest.

## Dispersion

`src/tools/native_viewer_export/shot_dispersion.py` replays the one fitted
swing `n_shots` times with a random club roll in the hands and ball position
(`shot_dispersion_settings.json`):

- `n_shots` 10, `seed` 0, drawn with `numpy.random.default_rng(seed)`
  Gaussians; the same seed gives the same shots.
- `club_roll_sd_deg` 0.5 and `ball_position_sd_m` 0.005. The settings call both
  "first illustrative value, not a measurement" to be replaced by measured grip
  and setup repeatability. These are placeholders.
- Only those two inputs vary. Swing speed, path, strike point, spin and the
  other launch inputs of a real golfer are not sampled, so the spread is a
  replay-level sensitivity, not a golfer's dispersion
  (`dispersion_hole_settings.json` `sources.title_format`).
- Statistics are population standard deviation (`ddof=0`) via
  `compute_metric_statistics`; the ellipse comes from the sample covariance at
  `ellipse_sigma` 1 and needs `min_points_for_ellipse` 3 points.
- The sensitivities quoted in the settings (about 6.6 m of side per degree of
  roll and about 0.5 degree of face per centimetre of ball position) are from
  one matched right-handed driver swing, not a population.

## Rigid Versus Flexible Shaft

The current golfer model carries the flexible chain shaft (Stiff) by default;
a rigid club is a logged ballpark fallback that the canonical gate refuses
(`docs/agents/golfer_model_versions.md`, issue #15684). A bundle fitted with
the rigid shaft sends no shaft lead, droop or toe-down to the face: the impact
solve sees one head velocity and face normal, and the flexible state is read
back for reports only (`src/shared/python/impact_parameters/shaft_state.py`;
register, Shaft Bend in the Delivery). In one capture the rigid fit lost about
2.6 degrees of spin loft (epic #15443 delivery audit), a single capture and not
a population figure. The flex sweep summary in
`changes/15684-flex-sweep-reports-the-peak-shaft-bend-u.md` gives a peak bend
of about 105 mm; this repository records no published reference bend to
compare it with. The effect of the flexible shaft on the showcase flights
(epic child 8) is not yet measured.

## Target Frame and Hole Transform

- The pipeline frame is x down-range, y left, z up
  (`flight_report.py` docstring). The UD world frame is Z-up with the golfer
  facing -X and the target line along -Y (ADR-0041,
  `src/shared/python/impact_parameters/target_frame.py`, `UD_DEFAULT_TARGET_DIR`).
  Lateral side is positive to the golfer's right, so an open face goes right.
- The hole view projects the flight path into the target frame
  (`flight_coordinates`) and, when the aim line is the finish, rotates the plan
  by the aim yaw so tee-to-finish lies on the hole axis (`_aimed_local`,
  `hole_view.py`). The path is then placed on the tee of the hole
  (`shots_on_hole`, `src/tools/golf_environment/shots.py`). Carry and side are
  the flight model's; roll is added from the runout model.
- Hole presets are illustrative, not surveyed
  (`src/tools/golf_environment/shots_settings.json`, `hole.presets`): par 3 is
  160 m with a 30 m fairway and 12 m green radius, par 4 is 380 m with 35 m and
  15 m, par 5 is 480 m with 35 m and 15 m. The hole is flat and has no hazards or terrain.
- Draw and fade are solved by changing the club roll in the hands, not the
  swing: straight is the roll that lands on the target line, and a draw or
  fade is that roll minus or plus `shape_roll_step_deg` 2.0
  (`shot_setup_settings.json`). The bracket is about 6 degrees either side and
  a roll beyond `max_abs_roll_deg` 20 is refused. The default aim line for a
  shape is the finish line (`aim_line_settings.json`, `aim_line` auto).

## Known Shortcomings

1. Wind is absent; gusts, shear and air-density change along the flight are not
   modelled.
2. Drag has no data beyond spin ratio 0.3; both wedge carries are strict
   failures, 6 to 7 percent short of the tour averages.
3. Spin decay is one constant; the Trackman label has no bundled fit.
4. The impact model is rigid, flat-faced, with no compression, bulge, roll or
   head rotation; spin ran high per degree of spin loft in the child 1 review.
5. Head mass, inertia and gear heads are assumptions; only the driver and
   `iron7` have catalogue entries, and any other club uses driver defaults.
6. Only the 7-iron has a sourced COR law, extrapolated below its fitted range.
7. Pitch, chip and bunker clubs get the driver spin cap by name fallback.
8. Runout is one Penner first hop plus unmeasured roll gains on a flat fairway
   with no turf at impact; the Haake figure in Penner's paper is inconsistent
   with his equations.
9. Dispersion varies two inputs with placeholder standard deviations over 10
   shots.
10. The shaft bend reaches the face only if the bundle is fitted with the chain
    shaft; the effect on these flights is unmeasured.
11. Ten tour-average rows are the only reference for the flight; there is no
    reference for spin axis, roll, total or impact outputs.
12. The hole is an illustrative flat plan, not a surveyed course.

## How to Reproduce the Showcase Clips

Run headless; never open a viewer window:

```bash
export QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl
python3 -m src.tools.native_viewer_export --help
```

Every flag below is listed by that help on `main`. `driver.npz` stands for the
matched swing bundle (`--bundle`, a same-input `.npz`); `--swing` only names the
output files.

```bash
# Impact factors with smash factor, flight with carry and total, hole view
python3 -m src.tools.native_viewer_export \
  --bundle driver.npz --swing driver --club Driver --out out/ \
  --engines mujoco --impact-factors --ball-flight --hole-view \
  --hole-preset par4 --distance-unit yd --speed-unit mph

# Shot shapes aimed at the target (repeat for draw and fade)
python3 -m src.tools.native_viewer_export \
  --bundle driver.npz --swing driver_draw --club Driver --out out/ \
  --engines mujoco --impact-factors --ball-flight --hole-view \
  --shot-shape draw --aim-line finish

# Ten-shot dispersion (settings: shot_dispersion_settings.json)
python3 -m src.tools.native_viewer_export \
  --bundle driver.npz --swing driver_dispersion --club Driver --out out/ \
  --engines mujoco --dispersion-traces --hole-view
```

- `--shot-shape {straight,draw,fade}` excludes a non-zero `--club-roll-in-hands`;
  `--ball-position-shift` combines with it.
- `--hole-preset` takes `par3`, `par4` or `par5`; use `par3` for irons with
  `--club "7-iron"`.
- `--dispersion-traces` and `--hole-view` turn the ball flight on.
- `--lighting showpiece` selects the showpiece scene lighting.
- Units: `--distance-unit {yd,m}` and `--speed-unit {mph,m/s}`; the manifest
  stays SI.

Run the physics reference suite with
`python3 -m pytest tests/unit/physics/test_reference_shots_15443.py
tests/unit/test_flight_models.py
tests/unit/sg_optimizer/shot_model/test_ground_runout.py --timeout=60`.
