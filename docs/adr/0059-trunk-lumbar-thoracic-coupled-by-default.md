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
pair is near parallel. Independent literature (Fujii 2007, Bruno 2015 via
#12737) reports the regional split above as a stable ratio in the golf swing.

## Decision

Free-versus-coupled is decided by `decide_trunk_coupling` (free only if every
event has condition number below 1e3 and every axis cosine below 0.95). It
returns **coupled**, so the derived showpiece spec
`full_body_spec_anthro_driver_v2.json` now carries
`trunk_coupling.enabled = true` (`TRUNK_COUPLED` in
`full_body_v2_showpiece.py`). A test asserts the committed default equals the
decision, so the default cannot drift from the evidence.

## Consequences

- Fitting solves for three effective trunk angles (one per axis) instead of six;
  the thoracic angle follows the lumbar one by the cited ratio, as a soft
  MuJoCo joint equality.
- The ratio is a population prior, not a per-golfer fact. Adding a marker on
  the lumbar body (for example over L3) is the way to make the DOFs free;
  rerun the report, the decision follows.
- Numbers are for the current spec. They are regenerated after SHOW-A2/A4
  merge, and the decision is re-checked then.
