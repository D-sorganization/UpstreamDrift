# Golf Ball Flight and Impact Source Map

This source map covers the lightweight validation slice used by
`tests/unit/shared_python/test_golf_physics_source_backed_validation.py`. It is
not a full launch-monitor certification protocol.

Numeric constants below are wrapped in `calc:name` HTML-comment
markers and checked against the physics code by
`tests/docs/test_ball_flight_calc_sheet_parity.py`.

## Selected Constants and Assumptions

| Model value or range                                                                                                                                                     | Runtime location                                                                                                                                                  | Provenance                                                                                                                                                                              | Validation status                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Ball mass: 0.04593 kg maximum                                                                                                                                            | `src/shared/python/core/physics_constants.py::GOLF_BALL_MASS_KG`; `BallProperties.mass` uses 0.0459 kg                                                            | USGA Appendix III / Equipment Rules: ball weight limit is 1.620 oz / 45.93 g                                                                                                            | Source-backed regulatory limit                                                                                         |
| Ball diameter: 0.04267 m minimum                                                                                                                                         | `src/shared/python/core/physics_constants.py::GOLF_BALL_DIAMETER_M`; `BallProperties.diameter`                                                                    | USGA Appendix III / Equipment Rules: ball diameter must be at least 1.680 in / 42.67 mm                                                                                                 | Source-backed regulatory limit                                                                                         |
| Club spring effect / COR sanity upper bound near 0.83                                                                                                                    | `ImpactParameters.cor`; `RigidBodyImpactModel.solve`                                                                                                              | Current USGA Equipment Rules define conformance by the Pendulum Test Protocol for spring effect. USGA historical material describes the older COR method as 0.822 plus 0.008 tolerance. | Sanity guard only; not a conformance test                                                                              |
| Driver smash factor near 1.48, bounded below 1.52 in tests                                                                                                               | `RigidBodyImpactModel.solve` post-impact ball speed divided by pre-impact club speed                                                                              | TrackMan PGA tour averages report driver smash factor around 1.48.                                                                                                                      | Launch-monitor sanity range, not player fitting                                                                        |
| Driver launch around 11 deg and spin around 2500-2700 rpm                                                                                                                | `LaunchConditions` examples and docs                                                                                                                              | TrackMan tour averages and `docs/physics/BALL_FLIGHT_MODEL_DOCUMENTATION.md` validation targets                                                                                         | Typical measured launch-monitor quantities                                                                             |
| Aerodynamic drag coefficient base `cd0 = `<!-- calc:cd0 -->0.21<!-- /calc -->                                                                                            | `BallProperties.cd0`                                                                                                                                              | Existing model docs cite Waterloo / wind-tunnel-style coefficient fits.                                                                                                                 | Illustrative model coefficient                                                                                         |
| Legacy (unused) lift slope `cl1 = `<!-- calc:legacy_cl1 -->0.38<!-- /calc --> and lift cap `MAX_LIFT_COEFFICIENT = `<!-- calc:max_lift_coefficient -->0.26<!-- /calc --> | Penner power law in `ball_properties.py::calculate_spin_lift_coefficient` (`cl1` is a dead dataclass field; see `BALL_FLIGHT_MODEL_DOCUMENTATION.md` Section 2.2) | Existing model docs cite spin-parameter coefficient fits.                                                                                                                               | Illustrative model coefficient and numerical guard                                                                     |
| Spin decay rate `SPIN_DECAY_RATE_S = `<!-- calc:spin_decay_rate_s -->0.05<!-- /calc --> 1/s                                                                              | `BallProperties.spin_decay_rate`; `compute_spin_decay`                                                                                                            | Labeled as TrackMan-derived in constants, but no raw fitting dataset is bundled.                                                                                                        | Illustrative assumption; tests verify monotonic exponential behavior only until a bundled fitting dataset is available |

## Validation Boundary

The focused tests verify that:

- aerodynamic coefficient and force calculations remain finite for ordinary
  driver launch states;
- drag opposes relative motion and Magnus force remains perpendicular to ball
  velocity for the selected coordinate convention;
- spin decay is finite and monotone for a positive exponential decay constant;
- rigid-body impact results produce finite velocities and a driver-like smash
  factor without exceeding the selected sanity cap;
- impossible launch speed and impossible COR inputs are rejected by existing
  contracts.

The tests deliberately do not claim that the selected constants calibrate a
specific ball, clubhead, player, or environment. Values without bundled primary
data are treated as model assumptions.

## Sources

- USGA, Appendix III / Equipment Rules for ball weight and diameter:
  https://www.usga.org/etc/designs/usga/content/rule-book/rule-14324.html
- USGA, Equipment Rules, spring effect and dynamic properties:
  https://www.usga.org/equipment-standards/equipment-rules-2019/equipment-rules/part-2-rule-5.html
- USGA, History of Equipment Rules, legacy COR method:
  https://www.usga.org/content/dam/usga/pdf/Equipment/History%20of%20Equipment%20Rules.pdf
- TrackMan Help Center, PGA/LPGA tour averages:
  https://support.trackmangolf.com/hc/en-us/articles/5089752464667-Shot-Analysis-Tour-Averages-On-PGA-LPGA-Tour
- Existing project model documentation:
  `docs/physics/BALL_FLIGHT_MODEL_DOCUMENTATION.md`
