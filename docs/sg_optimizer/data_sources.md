# sg_optimizer — data sources

All numeric values used by the optimizer trace back to a citation here. If a
value is interpolated or estimated it carries an inline `# INTERPOLATED:` or
`# ESTIMATE:` tag in the YAML.

## Baseline shot dispersion

- **Broadie, Mark.** _Every Shot Counts: Using the New Science of Golf to
  Win at the Game_ (2014). Tables A.1–A.4 — PGA Tour driving accuracy, GIR
  by distance, proximity to hole by distance.
- **Fawcett, Scott.** _Decade Golf_ publicly-shared dispersion distributions
  for wedges and short irons (used to triangulate the lower bag).

Per-club `rho` (correlation between along-target and lateral error) is not
directly tabulated in the above sources. Values of 0.15–0.30 are estimates
informed by clubface-correlation literature; see spec §1.1.

## Course conditions

Numeric coefficients implemented in `src/shared/python/sg_optimizer/course/conditions.py` trace to:

### RoughModel

- Rough distance penalty curve `1 - 0.08r - 0.12r²` matches USGA Tour-prep
  data (heavy rough costs ~25%; US-Open-like ~30%).
- `dispersion_multiplier` `1.0 + 0.4 * severity`: # ESTIMATE: calibrated to
  PGA Tour ShotLink driving accuracy decay across light-to-heavy rough (approx. 40% wider dispersion under US Open / maximum rough).
- `flyer_probability` `4.0 * r * (1.0 - r) * 0.25` (peaking at 0.25 for `severity = 0.5`): # ESTIMATE: models the "between lies" regime where reduced groove interaction prevents spin without sufficient grass friction to grab the ball.
- `spin_reduction` `0.5 * severity`: # ESTIMATE: based on TrackMan launch monitor rough studies showing up to 50% spin decay on dry, lush rough lies.

### TreeModel

- `penalization`: Phase 1 heuristic model treating tree density like scaled rough; full recovery-distribution lands in Phase 2 (#6271).
- `is_forced_punch_out` threshold `> 0.85`: # ESTIMATE: dense/jail tree lies force chip-out advances.
- `distance_multiplier` `max(0.05, 1.0 - 0.9 * penalization)`: # ESTIMATE: models severely restricted swing paths and advance distance.
- `dispersion_multiplier` `1.0 + 0.6 * penalization`: # ESTIMATE: deflection risk and punch-out angle variance.

### GreenModel

- Stimpmeter-vs-make-% slope `α = 0.015` calibrated from PGA Tour ShotLink
  putting tables published 2018–2022.
- `leave_distribution_modifier` `1.0 + 0.08 * max(0.0, stimp - 10.0)`: # ESTIMATE: modeled from ShotLink lag-putting variance showing increased second-putt distance variance on stimp > 10 greens (~8% increase per stimp foot).
- `effective_green_depth_multiplier` `1.0 - 0.06 * max(0.0, stimp - 10.0)`: # ESTIMATE: approach shot bounce and roll-out decay on faster greens (~6% reduction in effective landing zone depth per stimp foot above 10).

## Pin-position difficulty coupling

Heuristic; not directly sourced. Tagged `pin_position_difficulty` in
`CourseConditions` and treated as a multiplier on long-putt make-%. # ESTIMATE:
tucked pins increase three-putt probability and conservative leave requirements.

## Classic-hole geometries (Phase 2)

Polygons are hand-traced from public satellite imagery. The traced GeoJSON
representations are committed to this repo; the source imagery is copyrighted
and not redistributed. Each `hole.geojson` records the trace date and source
in its `properties.provenance` field per spec §4.2.
