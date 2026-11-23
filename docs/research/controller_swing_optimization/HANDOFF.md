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
  resimulation, and per-quantity replay tolerances. The separate research
  branch `codex/15493-solution-bundle-prototype` at `d42adfc3f8` adds a
  restricted research-only JSON/NPZ codec and a real two-hold MuJoCo replay
  with ideal clipped input. It can replay an unverified prescribed latched
  force at an off-grid event and compare the actual split state; 100 focused
  tests passed. This is not contact detection and uses the legacy impact
  partition. It rejects unsupported policies and assets; its runtime
  fingerprint is partial while Git identity remains caller-asserted. Embedded assets,
  post-actuator evidence and full policy remain missing. A limited explicit
  research-v2/v3 to v4 no-impact fixed-grid migration is available; production
  schema migration remains missing. The oversized research branch is not a
  compliant or complete Slice 2 PR.
- [ ] **Slice 3:** Fixed-gain tracking and bounded smooth feedforward spline;
  trust radius, sensitivity audit and total-applied-torque, ROM, speed,
  tracking, slew and work feasibility checks. A separate research primitive
  on `codex/15493-option-b-prototype` at `6aad831418` covers fixed PD,
  clamped cubic residual splines, coefficient trust bound and ideal clipping
  of the total command; 14 focused tests passed. A later research checkpoint
  `473e4df628` adds a caller-attested total-input ZOH work/update-rate audit
  and distinguishes observed ideal input; 34 combined focused tests passed.
  Work assumes linearly interpolated endpoint
  velocities, the update-rate metric is not continuous ZOH slew, and prehistory
  is unknown. Neither checkpoint establishes physiological feasibility, an
  optimizer or actual actuator evidence. The combined branch exceeds the PR
  size limit and is not a complete Slice 3 delivery.
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
A further restart changed the objective only from 7.988055 to 7.987017 while
the projected-gradient norm increased to 5.597, so repeated identical
restarts were stopped. The largest sampled generalized speed occurs at the
free record endpoint; the largest second difference occurs near a marker
dropout where the corresponding marker-Jacobian sensitivity decreases.
This association does not prove causation or establish anatomical rates.
A matched 120-second sensitivity around that dropout compared an added
localized temporal penalty with an unchanged-objective control, starting from
the same accepted state. Neither converged. The penalty reduced peak sampled
second difference from 2,145 to 2,090 rad/s² but increased maximum soft closure
translation from 0.0094 to 0.0639 mm; marker and face-axis residuals were
nearly unchanged and the free-end speed remained about 33.3 rad/s. The
experiment does not justify adopting that penalty.
A matched full-record comparison increased the global second-difference
weight from 10 to 30, with the same seed and 180-second budget. Neither run
converged. Peak angular second difference fell from 2,158 to 2,042 rad/s²,
while maximum soft closure translation increased from 0.0091 to 0.0670 mm;
marker RMS and near-impact delivery proxies were nearly unchanged. This
tradeoff does not justify adopting the stronger weight.
A separate matched endpoint sensitivity used only a weak prior toward the
estimated earlier motion trend, without future observations. Both runs timed
out without convergence; endpoint speed changed from 33.325 to 33.306 rad/s,
with negligible change in the near-impact face-centre speed and direction
proxies. The trend is an unvalidated model assumption and is not adopted.
Direct derivation from the embedded calibrated model rejects an unknown pelvis
offset; relocating its pelvis is a physical model change, not a coordinate-only
migration. A three-state diagnostic of that changed-model candidate preserved
grip closure but did not preserve all original marker positions; copying its
coordinate values into the required committed v2 produced large closure and
marker errors. This does not prove that a valid mapping is impossible.
The pinned bundle's hash-bound dynamics record also contains the original
controller target, distinct from its saved feedback rollout. Source-model
sampled target states close at the grip; naive or provisional sign-mapped
transfer to committed v2 produces large marker and closure errors. An initial
200 ms source-model open-loop replay matched the saved rollout to near
roundoff. In a repeated full pre-impact source-v1 replay, maximum angular
coordinate disagreement first exceeded 1e-6 rad at 0.755 s and 0.001 rad at
0.920 s, growing to 65.83 rad by 1.322 s. Recorded impact begins later, so
its omission does not explain this pre-impact divergence. This historical
v1 result is not the required v2 audit or a feasibility certificate.

The contact prototype rejects duplicated faces, non-manifold edges and
inconsistent shared-edge winding. Its bounded root scan can still miss narrow
crossings, and its sampled miss penalty is not a global closest-approach
guarantee. An integrator audit found that repartitioning trial durations could
make a state jump at a substep boundary; the working correction anchors trial
and accepted segment partitions to a common hold lattice. This removes the
demonstrated numerical discontinuity but does not bound ODE integration error
or certify first contact. A later folded-face test showed that changing the
nearest triangle normal can create a false signed-gap root. The prototype now
brackets finite-surface sphere clearance, checks outward side and approach
separately, and requires a 1 µm numerical clearance residual. It retains a
bounded scan and a tessellation-dependent sampled miss surrogate. A focused
contact, impact, stage-trace and MuJoCo suite passed 105 tests with graphics
disabled; a separate revision-2 flight
suite passed 16 tests with one strict expected failure, including a check that
an airborne endpoint cannot be reported as carry. Reference
flight rows include approximate inputs and are not independent flight
validation on this research branch. Current `origin/main` has since replaced
those rows with source-linked tour-table values and changed the flight model;
the old branch's results do not validate that newer model. Rebuild and rerun
the exact delivery commit, recording its kernel and fixture versions. Tour
averages remain unpaired observations under venue conditions, not a validation
of an individual shot under standard air.
An isolated detached test of `origin/main` at `fa228f3a9374`, with its own
freshly built physics revision 3 extension, passed 30 tests with two strict
expected failures in wedge carry. The build resolved Cargo dependencies afresh;
pin and record the build environment before treating it as reproducible.
For a fixed ball centre and continuous rigid face with a proven material-point
speed bound V and exact endpoint distances d0 and d1 over duration dt, the interval distance
is at least (d0 + d1 - V dt)/2. A strict bound above ball radius can certify
clearance. The current arbitrary advance callback supplies neither a proven
interval speed bound nor numerical distance enclosures; sampled qd alone does
not bound the derivative of an RK4 trial-prefix map. Do not infer clearance
certification from the sampled scan.
An RK4 counterexample also showed that the trial-prefix clearance derivative
can report approach while the returned state velocity is separating. The
prototype now requires an event-state material-point velocity, checks it
against the latch Jacobian and leaves unavailable or ambiguous mappings
unresolved. MuJoCo uses its native point Jacobian; this does not certify
between-sample motion or first contact.
The renderer-pinned bundle has a ball centre from a legacy modeled-address
passage plan, not an independently measured placement. Against that unchanged
centre, 123 sampled pre-impact states of the exploratory v2 rollout had a
minimum sphere-to-face gap of 169.75 mm in the last 122 ms; its initial pose
overlapped the same ball by approximately 12.7 mm. The available rollout
ends before the source impact time. Those samples neither establish a valid
address nor prove absence of contact between samples, and no ball placement
was shifted to force a hit. On the final recorded 1 ms, a bracket search using
the real held-effort integrator reproduced the stored end state bit-for-bit and
returned unresolved with no accepted candidate; it is not a global miss proof.
An additional source-v1 check used its unchanged modeled-address ball, saved
state and held effort through the final partial millisecond up to its legacy
passage trigger. The physical-approach finder returned unresolved with no
accepted candidate; the minimum of 17 sampled clearances was 22.24 mm. This
is neither a measured-ball result nor proof that no crossing occurred between
samples, and it does not qualify the required v2 baseline. At the source
bundle's initial state that same modeled-address sphere overlaps the rendered
face by about 1.48 mm; it is not a verified nonpenetrating address.
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
An independent 200 ms computed-torque rollout of a saved, nonstationary
full-record fit completed all 6,400 expected stages without impact. Joint
tracking RMS was 0.00812 rad and root translation RMS 14.52 mm; peak
non-root ideal drive was 257.21 N m. Eleven of 41 modeled drive thresholds
and two coordinate-ROM slots were exceeded; none of the generalized speed
priors were exceeded. Five of those eleven capacity slots and all speed
priors were assumption-marked. This is not Option B, a post-saturation
measurement, anatomical clearance or an accepted trajectory.
A matched 200 ms trial smoothed that same saved reference with an 11-sample
cubic Savitzky--Golay filter followed by the existing weld projection. Peak
ideal drive fell from 257.21 to 238.94 N m, but joint tracking RMS rose from
0.00812 to 0.00913 rad, root translation RMS from 14.52 to 16.32 mm, and
modeled ROM threshold flags from two to four slots; 11 modeled drive thresholds
remained exceeded. This is not a physically accepted improvement.
Each run used the same initial-state preparation procedure but its own
reference's first pose, so this is not a same-state causal comparison.
The same saved fit was tracked over the full 1.322-second pre-impact interval:
42,304 of 42,304 accepted stages, no impact or numerical failure. Joint
tracking RMS was 0.01532 rad, root translation RMS 30.43 mm, and peak
unsaturated ideal drive 356.87 N m. Thirty of 41 modeled directional drive
thresholds and seven coordinate-ROM slots were exceeded; none of the
generalized speed priors were exceeded. Thirteen of the 30 exceeding drive
slots and all speed priors were assumption-marked. These are sampled model
thresholds, not clinical findings; the nonstationary fit, computed-torque
controller, missing saturation, anatomy and impact prevent admission.
A 200 ms fixed-gain sensitivity used the prior computed-torque history as
frozen feedforward, trial gains 3 N m/rad and 0.3 N m s/rad, and a smooth
2 N m elbow pulse. The pulse-response joint RMS was 0.15352 rad open-loop
and 0.02835 rad with fixed PD; baseline tracking RMS was 0.00812 and 0.01684
rad respectively. Each run recorded 6,400 stages without impact. All four
exceeded 11 of 41 declared modeled drive thresholds, and two to five ROM
slots. The gains, ideal unsaturated effort, exploratory reference and assumed
limits do not establish feasibility. A larger trial gain failed numerically.
For that pulse, the change in the same-axis PD term opposed the pulse on all
199 nonzero hold samples, peaking at 1.214 N m. This local sensitivity does
not by itself justify optimizing reference shifts. The larger 30/3 trial
failed with nonfinite acceleration after five completed 1 ms steps; that is
a numerical result for this setup, not a general physiological conclusion.
A 0.2 N m pulse under the same gains gave open-loop and PD response RMS of
0.02319 and 0.002842 rad, with same-axis PD opposition on 198 of 199 nonzero
holds. These exploratory responses are nonlinear across pulse amplitudes;
all runs still exceeded declared drive thresholds.
Over the full 1.322-second pre-impact interval with the same frozen history,
the 3/0.3 PD baseline failed numerically after 0.641 s and the 0.2 N m pulsed
case after 0.548 s. The corresponding open-loop baseline completed, but its
pulsed counterpart diverged to 16.33 rad joint RMS and exceeded 40 of 41
modeled ROM slots and all 41 generalized speed priors. These are sensitivity
and numerical results for this controller, time step and unadmitted reference;
no impact, saturation, anatomical or optimized-flight claim follows.
A further full-duration 0.3/0.03 fixed-gain trial completed its unpulsed run
but diverged to 5.78 rad tracking RMS, exceeding 40 of 41 declared model ROM
slots and all 41 speed priors. Its 0.2 N m pulsed run failed numerically at
0.522 s. This gain choice is not an accepted controller; no conclusion about
all possible gains or physical injury follows.
With the same 1 ms hold and 3/0.3 gains, increasing RK4 substeps from eight to
16 allowed the full 1.322-second run to finish, but joint tracking diverged
to 3.083 rad RMS and peak unsaturated ideal drive reached 963.95 N m. Matched
eight- and 16-substep states differed by at most 5.34e-7 rad at 0.4 s and
1.176 rad at 0.6 s. Numerical resolution changes this setup's failure mode;
the higher-resolution result is not an admitted controller.
An additional 700 ms sensitivity changed the feedback hold from 1 ms to
0.125 ms, using recorded RK4 substep states as the reference and the same
3/0.3 gains. Zero residual reproduced recorded grid states bit-for-bit; a
0.2 N m pulse gave 0.747 rad RMS deviation versus 0.751 for the earlier
1 ms hold, with the same 24 ROM, ten speed-prior and eleven held-effort model
threshold exceedances out of 41 slots. The continuous pulse was sampled at
different times in the two cases. Neither result establishes feasibility.
When the fixed-PD reference was changed to the recorded dynamic states that
produced the frozen feedforward, the zero-residual rollout reproduced those
states bit-for-bit over 1.322 s. This isolates a reference/feedforward mismatch
in the earlier nominal test; it does not establish a matched target. A 0.2 N m
elbow pulse completed numerically but grew to 3.088 rad RMS deviation over
the full interval, with peak ideal drive 640.15 N m and widespread declared
model-limit exceedances. These grid-sampled, unsaturated results remain
nonadmitted and do not prove anatomical or clinical limits.
With that same recorded-state reference, increasing the trial derivative gain
from 0.3 to 0.6 while retaining proportional gain 3 caused the 0.2 N m pulse
run to fail numerically after 45 ms; the unpulsed run reproduced the nominal
states for 700 ms. This is specific to the sampled controller and model.
At the initial recorded state, finite differences of the complete 1 ms
held-control and projection map gave local spectral radii about 1.034 and
2.179 for derivative gains 0.3 and 0.6 respectively, stable across tested
finite-difference steps. This supports local gain-dependent amplification,
not a global stability or physiological conclusion.
A predeclared per-coordinate sensitivity retained 3/0.6 gains except for
zero gains on both modeled toe coordinates. Over 700 ms, its 0.2 N m pulse
gave 0.403 rad joint RMS deviation versus 0.751 rad for uniform 3/0.3; uniform
3/0.6 failed numerically. The toe-zero trial still exceeded 20 of 41 modeled
ROM slots, four speed priors and 11 drive thresholds. It is an unvalidated,
unsaturated numerical experiment, not an accepted or optimized swing.
An independent 200 ms test of the separate Option B primitive used the same
recorded-state reference and zero spline residual, comparing an unclipped
control with declared directional model-capacity clipping. The control
replayed its recorded states exactly. Clipping affected 45 of 200 holds and
11 channels; recorded ideal drive matched the clipped total at all 6,400
stages, but tracking RMS rose to 0.44482 rad and 17 modeled ROM and three
generalized-rate slots were exceeded. Sixteen capacity priors were assumed.
This is not post-actuator evidence or physiological clearance.
A matched 200 ms clipping ablation used the same state, reference and gains.
Joint tracking RMS was 0, 0.44482, 0.04142 and 0.43842 rad for unclipped,
all-cap, assumption-marked-only and other-registry-only clipping respectively.
Relaxing either group still violated some declared caps. This identifies a
model sensitivity, not validation of the registry values as anatomical limits.
An additional same-state ablation isolated one modeled knee cap: clipping that
cap alone gave 0.44928 rad tracking RMS, while applying the other caps but
relaxing it gave 0.09199 rad RMS and still violated that declared cap. This
does not establish that the cap is valid or invalid and does not authorize
relaxing it for acceptance.
The registry's cited knee capacity derives from an isometric group parameter
in [Anderson et al.](https://stacks.cdc.gov/view/cdc/188754/cdc_188754_DS1.pdf),
whose model explicitly varies strength with joint angle and velocity. The
clipped, divergent diagnostic included eccentric modeled motion beyond the
study's reported eccentric test speed. The corresponding over-cap hold-start
speeds in the unclipped nominal diagnostic were within that speed interval,
but the subject and posture are not validated matches. The static cap is a
declared model gate, not a validated subject-specific dynamic limit. Preserve
it until a justified policy and mapping are available.
As a sensitivity only, evaluating the published population-mean active
angle/velocity model at the 13 unclipped over-cap hold-start states gave
115.88--121.39 N m; 12 ideal commands exceeded those point estimates. The
study's seated, right-leg cohort and population parameters are not validated
against this modeled left-leg stance; passive torque, individual uncertainty
and between-stage behavior remain unverified. This did not alter the gate.
Bit-identical replays of both 200 ms cases recovered pre-projection hold-end
velocities. Under left-ZOH ideal input and linear velocity interpolation,
unclipped versus clipped modeled positive work was 32.48 versus 327.54 J;
signed work was -5.40 versus 151.50 J. These are numerical quadratures, not
measured actuator or clinical work. Work, effort and slew limits were not
available; the clipped run remains indeterminate even though its modeled
torque caps were met.
A predeclared 200 ms three-point B-spline sensitivity tested one coefficient
at 0 and ±1 N m, within a unit scaled coefficient trust ball, with the same
state, reference, gains and total-command clipping. All three completed
6,400 accepted stages and met the declared sampled drive caps, but each
exceeded 17 of 41 modeled ROM slots and three generalized-speed priors.
Tracking RMS was 0.4448--0.4450 rad. None is admissible; three samples are
not evidence that no feasible improvement exists. Work was evaluated only
as ideal-input ZOH quadrature and anatomical mapping remains incomplete.
A subsequent three-parameter constrained SLSQP search hit its time limit
without convergence. All 12 evaluated candidates were within the coefficient
trust ball and met the sampled clipped-drive caps, but each exceeded 17 modeled
ROM slots and three generalized-speed priors. This limited search does not
prove infeasibility and lacks complete anatomical, actuator, work/slew,
between-sample and impact evidence.
A matched 32-substep run completed 0.7 s with approximately the same 1.431 rad
tracking RMS as the 16-substep run. Their maximum coordinate difference grew
from 1.13e-7 rad at 0.4 s to 0.109 rad at 0.7 s; resolution convergence is
not established.
The flight suite was repeated in a fresh isolated Python 3.12 environment
after building the current Rust extension and verifying physics revision 2:
15 passed and one strict expected failure. The stale default revision 0 is
not a valid acceptance runtime.
