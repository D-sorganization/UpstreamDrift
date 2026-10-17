# Open-FP Integration Delivery Backlog

Proposed child issues for epic [#13061](https://github.com/D-sorganization/UpstreamDrift/issues/13061). None is filed or implemented by
this planning change. Each child is one narrow pull request with tests first.

## Phases

| Phase | Children            | User Outcome                                                              |
| ----- | ------------------- | ------------------------------------------------------------------------- |
| P0    | OF-00               | Upstream format carries version, time anchor, geometry and calibration id |
| P1    | OF-01, OF-02, OF-03 | Import open-fp swing files and see them as ground reaction data           |
| P2    | OF-04, OF-05, OF-07 | Overlay measured vertical GRF and CoP on a matched model swing            |
| P3    | OF-06, OF-08        | Pull swings from the board; compare model and measurement numerically     |
| P4    | OF-09, OF-10        | Live weight shift; research use in contact-wrench allocation              |
| NSX   | OF-11, OF-12        | Plates as measured constraints that eliminate candidate swing solutions   |
| Gate  | OF-HW               | Hardware-qualified device record (deferred validation DV-13061)           |

P1 does not depend on P0: the reader pins the reviewed commit and works with
today's format. P0 improves alignment and provenance when it lands.

## Children

| Id    | Title                                                            | Depends On          | Acceptance                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----- | ---------------------------------------------------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OF-00 | Upstream: versioned swing format                                 | none                | Owner-approved issue or PR in open-fp proposing `schema_version`, `t0_unix_ms` for `t[0]`, a geometry snapshot (spacing and cell positions) and a calibration id in `<id>.json`. External; UpstreamDrift does not block on it                                                                                                                                                                                                                       |
| OF-01 | `VerticalPlateTrial` contract and conversion (#13065, PR #13074) | none                | Frozen record with postconditions; `to_ground_reaction_series` fills shear and free moment with NaN and an observability record; hand-computed corner and centre point-load tests                                                                                                                                                                                                                                                                   |
| OF-02 | open-fp file reader (json + npz, csv) (#13066, PR #13075)        | OF-01               | Reads the pinned format; `allow_pickle=False`; id pattern and size checks; rejects unknown keys, length mismatches and non-finite times; CSV path requires explicit handedness and body weight; fixtures produced by an UpstreamDrift generator                                                                                                                                                                                                     |
| OF-03 | Stance-to-model frame and handedness mapping                     | OF-01               | Documented target-axis decision; golden tests for RH and LH; no silent mirroring; optional plate placement in a capture volume                                                                                                                                                                                                                                                                                                                      |
| OF-04 | Force-event namespace in ground reaction plots                   | OF-01               | open-fp events shown as `force.*`; never mapped to swing positions; impact offered as an anchor only when sourced                                                                                                                                                                                                                                                                                                                                   |
| OF-05 | Plate reading at model time (#13069, PR #13078)                  | OF-01               | Linear interpolation on the actual time stamps; measurement sigma, skew term and interpolation bias **estimate** (pinned as blind to under-sampled impact transients; curvature floors required when the estimate is unavailable); common-mode clock offset kept out of sigma and profiled; alignment derived from a model cannot test that model                                                                                                   |
| OF-06 | Read-only LAN client                                             | OF-02               | Only GET endpoints; timeouts and size caps; responses validated by the OF-02 code path; offline tests against an in-process fake server                                                                                                                                                                                                                                                                                                             |
| OF-07 | Ground reaction tile: import and overlay                         | OF-03, OF-05        | Import dialog, measured-versus-model per-foot Fz and CoP, unqualified-data banner; `feature_parity.json` entry for `analysis.ground_reaction` updated                                                                                                                                                                                                                                                                                               |
| OF-08 | Model-versus-measured vertical comparison metrics                | OF-05               | Per-foot Fz and CoP residuals on observed components only, lead % timing deltas with uncertainty; results labelled unqualified until OF-HW                                                                                                                                                                                                                                                                                                          |
| OF-09 | Live bridge to realtime IPC                                      | OF-06               | Optional extra with a ledger row for the client library; channel registered in both realtime registries; reconnect and back-pressure tests                                                                                                                                                                                                                                                                                                          |
| OF-10 | Research: measured vertical wrench in contact QP                 | OF-08, OF-HW        | Proposal only: measured Fz and CoP-derived Mx, My as nominal targets in `contact_wrench_qp`, shear left free; error budget from OF-05                                                                                                                                                                                                                                                                                                               |
| OF-11 | Vertical-cell observation rows (#13067, PR #13077)               | NSX-60              | Per-foot load and CoP-moment rows about the force-sensing plane with the exact shear lever; sympy goldens for both feet variants and both depths; spec section updated                                                                                                                                                                                                                                                                              |
| OF-12 | Family-wise plate screen (#13070, PR #13079)                     | OF-01, OF-05, OF-11 | Bonferroni coverage over all tested rows, offset profiled with half-step skew, joint feasible-set verdict deciding, no vacuous rescue; calibrated by an exact single-row Monte Carlo                                                                                                                                                                                                                                                                |
| OF-HW | Qualified device record                                          | hardware            | Catalog record [DV-13061](../../development/planning/DV-13061.md): per-device calibration file; open-fp M4 static checks (total force within 1 %, CoP within 10 mm, body weight within 1 %); M6 real-swing capture on production firmware (20 swings, none missed, no false triggers in 2 minutes of quiet standing); M3 frames with no sequence gaps over 10 minutes; temperature and drift notes. Bench-only or simulator data never satisfies it |

## Do Not

- Do not add open-fp, or code copied from it, to this repository (AGPL-3.0).
- Do not fill unobserved shear or free moment with zeros.
- Do not call open-fp's mutating endpoints.
- Do not label `max_trail` as "top" or present force events as club positions.
- Do not treat imported data as model acceptance evidence before OF-HW.
- Do not claim sub-10 ms timing agreement without a matching alignment
  uncertainty.

## Risks

| Risk                                          | Mitigation                                                    |
| --------------------------------------------- | ------------------------------------------------------------- |
| Upstream format changes without a version     | Pin the commit; structural detection; fail closed; OF-00      |
| Hardware never reaches qualification          | P1 and P2 remain useful for visualization; OF-HW gates claims |
| Wi-Fi time error dominates timing metrics     | Uncertainty-aware alignment; prefer in-swing anchors          |
| Licence boundary eroded by convenience copies | Review checklist item; no dependency row allowed for open-fp  |
| Unauthenticated LAN API returns hostile data  | Read-only client; strict validation; size limits              |
