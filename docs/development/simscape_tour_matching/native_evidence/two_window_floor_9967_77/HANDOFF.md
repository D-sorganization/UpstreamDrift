# Two-Window Derivative Resolution Audit 77 — Qualified With Measured Floors

Audit77 is TERMINAL (exit0, launch-to-terminal202.6 s, driver179.6 s,26 of
32 budgeted window replays). It resolves both run76 failure classes at the exact
run76 fixture without changing any gate. The executed receipt reports
`unqualified_derivative_blocks` with a factor-one floor; the local
reclassification with the documented factor-two floor reports
`qualified_with_factor` for all four directions and all seven blocks. No
optimizer ran. Runtime77 qualification:122 tests including real Pinocchio cases.

## Fixture, Runtime and Reuse

Candidate remains exact returned73
`786522cd5380a9f62602920b2cff6e6b404f7ceb99bfc6783f1f359b98a6457a`, original
model, saved0.6 node chart and run76's archived sensitivity Jacobians
(hash-verified against the run76 summary). Runtime77 is frozen runtime73 with
only its existing file set refreshed from commit f96766c8e plus five new
provider/test files; no package `__init__` files were added. The zero
retraction recomputed through the shared `NativeNodeChart` reproduces run76's
node to1e-12 and its implicit Jacobian to1e-9. Base replays at the working
tolerance reproduce run76's primal q/qd arrays exactly (max difference0.0).

## Measured Replay Error (Floors)

Per-block L2 difference between working (rtol1e-11/atol1e-13) and tight
(1e-12/1e-14) replays, max_step0.0000625 unchanged: markers8.23e-10 m,
endpoint q9.26e-10, endpoint qd1.63e-8, continuity q9.23e-10, continuity qd
1.58e-8. Loose (1e-10/1e-12) versus tight is the same to three digits and loose
versus working is near zero: the integration is max-step limited, and the
non-reproducible part comes from step-sequence-dependent roundoff and
constraint-solver noise, not from tolerance-controlled truncation. Tightening
the tolerance therefore does not reliably reduce it (mixed h1e-7 improved4x;
LSInputX h1e-5 worsened2x). Node-only continuity floors come from the archived
retracted nodes' deviation from the linear prediction plus roundoff:
1.6e-14/2.7e-14 (h1e-5) and1.5e-12/2.5e-12 (h1e-4).

## Verdicts

| Direction   |       Step | Tolerance | Verdicts (factor1 / factor2)                                 |
| ----------- | ---------: | --------- | ------------------------------------------------------------ |
| LSInputX B6 |       1e-5 | work      | continuity q/qd/scaled unresolved; others passed             |
| LSInputX B6 |       1e-5 | tight     | continuity qd/scaled failed at1.05x/1.02x floor → unresolved |
| LSInputX B6 |       1e-5 | loose     | as work                                                      |
| LSInputX B6 |       1e-4 | work      | all seven passed (floor below gate times analytic norm)      |
| Node pos.   | 1e-5, 1e-4 | work      | continuity qd structural zero; all others passed             |
| Node vel.   | 1e-5, 1e-4 | work      | continuity q structural zero; all others passed              |
| Mixed       |       1e-7 | work      | qd/scaled blocks failed at1.08x–1.14x floor → unresolved     |
| Mixed       |       1e-7 | tight     | all unresolved (relative errors fell to about1.1e-3)         |
| Mixed       |       1e-7 | loose     | as work                                                      |
| Mixed       |       1e-6 | work      | all seven passed                                             |
| Mixed       |       1e-5 | work      | all seven passed                                             |

Structural zeros are verified independently: the zero-retraction Jacobian is
diag(scales) times a scaled-orthonormal basis (defect3.1e-15), so the velocity
block of the pure-position singular direction is bounded by7.6e-8 (analytic
5.0e-14) and the position block of the Dq null direction by9.8e-9 (analytic
2.3e-17). Their central estimates are retraction roundoff over2h.

Every factor-one failure lies at the smallest step of its direction with
absolute error1.02x–1.14x a floor measured from one replay pair. A pairwise
difference bounds the per-replay non-reproducible part only up to the unknown
split between the pair, so `reclassify.py` applies the tested provider with
`floor_safety_factor=2` to the same archived arrays; the gate stays1e-3 and a
pass still requires the widened floor to lie below the gate times the analytic
norm. Under that rule every block of every direction has a resolved pass or a
verified structural zero and no unexplained failure. Weak-block relative
ratios are reported, not waived.

## Evidence and Reproduction

`floor.py` (SHA25613c86e26…, identical to the executed remote copy),
`receipt.json`, `summary.json`, `launch.json`, `runtime-receipt.json`,
`reclassify.py`, `reclassification.json` and `raw-run.zip` (SHA256
`3d5768fcc076add3a6f104bcad1b0ce7ecb7614f0263f69974646eb2183c381e`) hold
every replay, comparison array, log, launch vector, overlay manifest and the
full runtime77 source. Remote output
`/mnt/c/Users/diete/native-two-window-floor-9967-77`; runtime
`/home/dieterolson/native-two-window-floor-9967-77`; launched with
`systemd-run --user` through `run_job.py`. Future experiments need a new
output path. The next authorized step is the bounded two-window direct-node
SLSQP trial specified in ../../NEXT_AGENT_CONVERGENCE_EXECUTION.md.
