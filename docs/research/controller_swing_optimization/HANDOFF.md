# Controller Swing Optimization Handoff

## Objective and Delivery

Epic #15493 requires five sequential, non-draft PRs targeting main, each
under 500 changed lines, with no auto-merge or merge without owner approval.
The working branch `codex/15493-baseline-audit` is a pushed research
checkpoint, not a PR-ready slice: its accumulated diff exceeds that limit.
Do not treat it as an accepted implementation or combine it wholesale into a
PR. Claims for #15493 and #15494 were posted on the current coordination
board, Repository_Management #2851. Verify current claims and bot identity
before any new GitHub write. The v1 specification is pinned and must not be
modified. Preserve private inputs and their identifying provenance outside git.

## Task Ledger

- [ ] **Slice 1:** Establish a qualified forward-match baseline on the pinned
  match and committed v2 model; verify prescribed-launch flight independently;
  establish bracketed, approaching on-face contact convergence; enforce hard
  ROM, speed, and total applied torque gates. The pinned embedded calibrated
  v1 model differs from the v1 source expected by the committed v2 migration;
  the guard now rejects that mismatch. The observed-marker refits and short
  constrained windows in the companion README are exploratory, not an
  admitted baseline. Actual post-saturation actuator evidence, full anatomy,
  a complete consistent reference and real matched-swing contact remain open.
- [ ] **Slice 2:** Versioned JSON and NPZ bundle, complete state and model
  identity, strict validation and migration, distinct playback and forward
  resimulation, and per-quantity replay tolerances.
- [ ] **Slice 3:** Fixed-gain tracking and bounded smooth feedforward spline;
  trust radius, sensitivity audit and total-applied-torque, ROM, speed,
  tracking, slew and work feasibility checks.
- [ ] **Slice 4:** Optimize power and straight flight through the existing
  pipeline, then independently replay and refine any qualified candidate.
- [ ] **Slice 5:** Comparison and analysis CLI with baseline and candidate
  metrics, feasibility margins, replay drift and explicit assumptions.
- [ ] Save and show occasional qualified optimization renders, with copies on
  the ControlTower Desktop. The existing straight-flight image is kinematic
  and is not an optimized or forward-validated result.
- [ ] Prepare compliant, reviewable slice delivery and run the required
  formatting, type and targeted test checks. No auto-merge or merge without
  explicit owner approval.

## Related Epic #15494

- [ ] **A:** Admit and pin the same two-handed baseline, controller, model,
  load policy, anatomy sources and replay contract; include near-rank-loss and
  infeasible-load fixtures. Six synthetic numerical-rank and inconsistent-load
  cases are pushed separately on `codex/15494-rank-fixtures` at `78929c2712`;
  25 focused tests passed. They do not establish a physical load failure, and
  the #15493 baseline is not yet admitted.
- [ ] **B:** Reuse the existing allocator to compare feasible fixed-motion
  strategies and component ranges, preserving net club wrench at a common
  reference; label allocation-only results when replay is unavailable.
- [ ] **C:** Export synchronized, common-scale replay videos and a manifest
  with frames, units, residuals, assumptions and constraint events.
- [ ] **D:** Independently qualify replay, render readback, negative controls
  and sensitivity to priors for panel handoff.

## Current Evidence and Next Work

The companion [research record](README.md) reports the numerical experiments
and their limits. A nine-frame constrained fit converged but required soft
boundary penalties; neighboring states still have closure error. A wider
17-frame dense solve did not converge in two 115-second budgets. Investigate
sparse temporal optimization and model lineage without changing the required
committed v2 baseline or inventing missing limits. Test full-trajectory joins,
recorded applied torque and independent forward behavior before acceptance.
A sparse `trust-constr` pilot did not converge on the nine-frame test and
degraded the fit from a converged SLSQP seed. It still assembled dense local
Jacobian blocks and used default BFGS, so it does not rule out a fully sparse
or better-conditioned trajectory formulation.
A block-sparse soft-weld pilot reached an `xtol` stop on 33 frames in 30.3 s:
maximum positional closure was 0.000606 mm, peak sampled angular speed was
16.75 rad/s, and join speeds were 7.34 and 15.76 rad/s. Its reported first-order
optimality was 0.276, so the stop is not a stationarity certificate. A 99-frame
trial reached its 90-second cap without convergence; its last saved point was
a function trial, not necessarily an accepted iterate. Soft weld penalties,
missing anatomical bounds and untested forward dynamics prevent acceptance.
Directional checks of the 33-frame Jacobian had relative errors of
1.4e-7--4.3e-7. A tighter, rescaled trust-region trial timed out and did not
improve the measured stationarity; its last function trial is not necessarily
an accepted iterate. A later 99-frame run from original seeds stopped on
`xtol` after 66.3 seconds, with first-order optimality 0.486. Its maximum soft
closure translation was 0.00172 mm and all supplied named bounds held, but
peak sampled angular speed remained 39.41 rad/s and acceleration remained
21,740 rad/s². It is not a stationary or physically admitted result.
A controlled 99-frame comparison with the same seeds and a tenfold larger
second-difference residual weight reached its 200-evaluation/300-second cap
without convergence (optimality 3.776). Peak sampled speed fell to 31.66 rad/s
and peak second difference to 984 rad/s²; maximum soft closure was 0.00628 mm,
maximum marker RMS 36.44 mm, and exit join speed 33.12 rad/s. All supplied
named bounds held. The altered objective, soft closure, unvalidated anatomy and
missing forward dynamics mean this is sensitivity evidence only.
Restarting that solver-returned point with the same objective stopped on
`xtol` after five evaluations, with independently recomputed unscaled
projected-gradient infinity norm 0.629 and exit join 33.12 rad/s. It is not
evidence of gradient convergence.
A full 477-frame, weight-10 sparse run reached its 300-second cap without
convergence. Forty accepted iterates were saved; the last had maximum soft
closure translation 0.0647 mm, maximum marker RMS 37.46 mm, peak angular
finite-difference speed 34.30 rad/s and peak second difference 5,840 rad/s².
All 29 supplied named bounds held. The endpoints were free, the weld remained
soft and 18 coordinate bounds were absent. This is not an admitted baseline.
Two subsequent 300-second restarts from separately saved accepted points also
timed out. The second ended with objective 7.9881, independently recomputed
projected-gradient infinity norm 4.989, maximum soft closure translation
0.00714 mm, maximum marker RMS 36.61 mm and peak sampled angular speed
33.33 rad/s. Named bounds held; the same physical limitations apply.
Direct derivation from the embedded calibrated model rejects an unknown pelvis
offset; relocating its pelvis is a physical model change, not a coordinate-only
migration. A three-state diagnostic of that changed-model candidate preserved
grip closure but did not preserve all original marker positions; copying its
coordinate values into the required committed v2 produced large closure and
marker errors. This does not prove that a valid mapping is impossible.

The contact prototype rejects duplicated faces, non-manifold edges and
inconsistent shared-edge winding. Its bounded root scan can still miss narrow
crossings, and its sampled miss penalty is not a global closest-approach
guarantee. An integrator audit found that repartitioning trial durations could
make a state jump at a substep boundary; the working correction anchors trial
and accepted segment partitions to a common hold lattice. This removes the
demonstrated numerical discontinuity but does not bound ODE integration error
or certify first contact. A focused contact, impact, stage-trace and MuJoCo
suite passed 89 tests with graphics disabled; a separate revision-2 flight
suite passed 16 tests with one strict expected failure, including a check that
an airborne endpoint cannot be reported as carry. Reference
flight rows include approximate inputs and are not independent flight
validation. Re-run relevant checks on the actual delivery commit and runtime.
In the MuJoCo parity plant the recorded held effort is a source command; the
native KKT solve zeroes passive shaft coordinates, and the forced wrapper
adds impact generalized force before that mask. The path has no actuator
control or saturation step. Audit actual accepted-stage ideal generalized
drive separately from impact, bias, passive, ground and weld forces, and do
not label it measured or post-saturation actuator torque.
An isolated research branch `codex/15493-ideal-drive-audit` at `d351e55f75`
captures the separated ideal drive and impact input at accepted RK4 stages.
Re-running the earlier 1.322-second rigid-v2 computed-torque diagnostic gave
42,304 of 42,304 expected stages; ideal drive equaled the root-zeroed source
at every stage and the impact component was zero. Peak non-root ideal drive
was 1,193.91 N m. This identifies the applied ideal input for that run only;
it is unsaturated, is not Option B, and is not anatomical certification.
At those recorded stages, 33 of 41 modeled non-root drive channels exceeded
their declared directional capacity thresholds and seven coordinate-ROM
slots exceeded their declared ranges; no generalized speed prior was exceeded.
Fourteen of the 33 exceeding capacity slots and all 41 speed priors were
assumption-marked. These are violations of this model's declared thresholds,
not established anatomical or clinical violations; complete feasibility is
indeterminate without validated mappings, saturation and between-sample bounds.
The flight suite was repeated in a fresh isolated Python 3.12 environment
after building the current Rust extension and verifying physics revision 2:
15 passed and one strict expected failure. The stale default revision 0 is
not a valid acceptance runtime.
