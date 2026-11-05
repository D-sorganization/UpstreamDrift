# ADR-0059: Lumbar and Thoracic Trunk DOFs Are Coupled by Default

- Status: Accepted
- Date: 2026-11-04
- Decision Makers: SHOW sprint (lane show-trunk), acceptance gate show-review
- Related Issues/PRs: #14554 (SHOW-A5), #14552 (SHOW-A3), #12737 (cited priors), epic #14549

## Context

The `full-body-v2` trunk has two 3-DOF joints, `lumbar_L5_S1` and
`thoracic_T12_L1` (six coordinates). SHOW-A3 added an optional per-axis
coupling `thoracic = k * lumbar` from the Bruno 2015 regional split
(`src/config/spine_coupling_laws.json`; lumbar/thoracic share of flexion
51/49, lateral bending 27/73, axial rotation 17/83), off by default. The
question for SHOW-A5 is whether the marker set can identify the six angles
as free DOFs, or whether they must be coupled.

## Evidence

`src/shared/python/motion_matching/trunk_observability.py` computes the
central-difference Jacobian of all attached marker positions (26 markers,
78 rows) with respect to the six trunk coordinates on the real v2 swing
(`evidence/trunk_visuals/driver_v2_bundle.npz`) at address (frame 0),
top (1103) and impact (1325):

| Event   | Rank (of 6) | Condition number | Smallest singular value | Lumbar/thoracic column cosine (flexion / lateral / axial) |
| ------- | ----------- | ---------------- | ----------------------- | --------------------------------------------------------- |
| address | 5           | 7.9e9            | 5.6e-10                 | 0.965 / 0.932 / 0.997                                     |
| top     | 5           | 7.3e9            | 5.1e-10                 | 0.960 / 0.957 / 0.998                                     |
| impact  | 5           | 7.5e9            | 6.0e-10                 | 0.956 / 0.956 / 0.987                                     |

Only three markers (BackLeft/Right/Top) sit on the trunk, all on the thorax
frame; none sits on the lumbar body, and the pelvis markers cannot see
lumbar motion apart from thoracic motion. One combination of the six
coordinates is therefore unobservable (rank 5), and every lumbar/thoracic
pair is near parallel. The cited priors (Fujii 2007 and Bruno 2015, as recorded in #12737; not
re-verified against the papers here) report the regional split above.

## Decision

Free-versus-coupled is decided by `decide_trunk_coupling` (free only if every
event has condition number below 1e3 and every axis cosine below 0.95). It
returns **coupled**, so the derived showpiece spec
`full_body_spec_anthro_driver_v2.json` now carries
`trunk_coupling.enabled = true` (`TRUNK_COUPLED` in
`full_body_v2_showpiece.py`). A test asserts the committed default equals the
decision, so the default cannot drift from the evidence.

## Limits of the Evidence

- **The rank result is geometric.** Rank 5 of 6 comes from where the markers
  sit relative to the two joints, so it does not depend on the particular
  lumbar/thoracic angles. It is reproduced at three poses.
- **The swing's lumbar/thoracic split is not a measurement.**
  `driver_v2_bundle.npz` is a v1 export migrated to v2 with the same
  Bruno 2015 prior (`evidence/trunk_visuals/README.md`), so its axial ratio
  (0.17 to 0.29 lumbar per thoracic, prior 0.205) only restates the prior.
  No second swing exists in v2 coordinates, so the 17/83 split could not be
  checked on independent data; it rests on the cited literature alone.
- **ROM versus the swing.** The migrated reference reaches thoracic axial
  -53.5 degrees (-52.5 at the top) against the +-50 degree soft limit, from a
  63 degree total trunk turn. With 17/83 the limit binds at about 60 degrees
  total. The limit is unchanged (n = 1, prior-derived); it is a candidate
  binding joint for high clubhead speeds and needs more swings fitted with
  free trunk angles (extra lumbar marker) to judge.
- Coupling is a soft MuJoCo equality, so the realised ratio has residual
  error (the A3 test accepts 20 percent); tighten `solref`/`solimp` before
  using it as a hard constraint.

## Consequences

- Fitting solves for three effective trunk angles (one per axis) instead of six;
  the thoracic angle follows the lumbar one by the cited ratio, as a soft
  MuJoCo joint equality.
- The ratio is a population prior, not a per-golfer fact. Adding a marker on
  the lumbar body (for example over L3) is the way to make the DOFs free;
  rerun the report, the decision follows.
- Numbers are for the current spec. They are regenerated after SHOW-A2/A4
  merge, and the decision is re-checked then.
