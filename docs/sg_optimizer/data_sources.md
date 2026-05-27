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

- Rough distance penalty curve `1 - 0.08r - 0.12r²` matches USGA Tour-prep
  data (heavy rough costs ~25%; US-Open-like ~30%).
- Stimpmeter-vs-make-% slope `α = 0.015` calibrated from PGA Tour ShotLink
  putting tables published 2018–2022.

## Pin-position difficulty coupling

Heuristic; not directly sourced. Tagged `pin_position_difficulty` in
`CourseConditions` and treated as a multiplier on long-putt make-%.

## Classic-hole geometries (Phase 2)

Polygons are hand-traced from public satellite imagery. The traced GeoJSON
representations are committed to this repo; the source imagery is copyrighted
and not redistributed. Each `hole.geojson` records the trace date and source
in its `properties.provenance` field per spec §4.2.
