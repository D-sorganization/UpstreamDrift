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
  branch `codex/15493-solution-bundle-prototype` at `09e5a6647a` adds a
  restricted research-only JSON/NPZ codec and a real two-hold MuJoCo replay
  with ideal clipped input. It can replay an unverified prescribed latched
  force at an off-grid event and compare the actual split state; 106 focused
  tests passed. Its restricted runner now uses the fixed held-step lattice and
  can bind and compare ideal source and external accepted-stage records in a
  typed NPZ archive. A fresh 200 ms exploratory v2 recording, save/load and
  same-runtime traced rerun passed with 6,400 actual stages. This does not
  establish independent runtime or matched-target validation. Full-horizon
  evidence is not yet qualified. This is not contact detection. It rejects
  unsupported policies and assets; its runtime
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
A retrospective local KKT coupling direction was then tested with a single
cubic residual coefficient at 0 and ±10 N m L2, keeping total-command
clipping and all declared caps. The positive trial reduced 200 ms joint
tracking RMS from 0.4448 to 0.4181 rad and modeled ROM flags from 17 to 16;
all three still exceeded modeled motion limits. Fresh-plant same-runtime
replays of the zero and positive trials matched saved q, v and inputs
bit-for-bit. The direction was selected from one state; neither independent
runtime replay nor physical qualification was performed, and this is not a
power or carry improvement.
A retrospective scale sweep along that same direction, using a larger
declared coefficient trust radius up to 20 N m, reduced 200 ms tracking RMS
to 0.3926 rad at the largest tested scale, but still exceeded 15 modeled ROM
slots and four speed priors. It is not a feasible candidate; the arbitrary
regularized diagnostic score did not improve monotonically with scale.
A separate unit-L2-trust local acceleration-matching direction reduced the
same-state modeled acceleration mismatch from 397.19 to 220.75 rad/s², but
its 200 ms rollout changed tracking RMS only from 0.444824 to 0.444684 rad
and still exceeded 17 modeled ROM and three speed slots. It did not produce
an admissible candidate or establish infeasibility of the full search.
A predeclared five-point combination of the two local directions stayed
within a 10 N m coefficient trust ball. The best sampled tracking RMS was
0.4330 versus 0.4448 rad for zero residual, but every candidate exceeded
modeled motion limits. This small grid does not establish full-search
infeasibility or a power improvement. A fresh same-runtime rerun of the
lowest-tracking-RMS candidate, selected after the scan, matched saved state,
command and ideal-input arrays bit-for-bit. Zero residual had the lower
predeclared penalized score. This is not an independent runtime or a Solution
Bundle replay.
A full-horizon check of zero residual and the selected two-direction trial
with the same clipped controller became nonfinite at 0.402 and 0.583 s,
respectively. Neither reached the 1.322 s endpoint; a later failure is not a
feasibility improvement, and two trials do not prove infeasibility. A
trajectory-wide solver with continuity constraints and uninterrupted replay
remains to be implemented.
A matched 32-substep run completed 0.7 s with approximately the same 1.431 rad
tracking RMS as the 16-substep run. Their maximum coordinate difference grew
from 1.13e-7 rad at 0.4 s to 0.109 rad at 0.7 s; resolution convergence is
not established.
The flight suite was repeated in a fresh isolated Python 3.12 environment
after building the current Rust extension and verifying physics revision 2:
15 passed and one strict expected failure. The stale default revision 0 is
not a valid acceptance runtime.

A fresh recorder-generated exploratory v2 no-impact run covered the full
1.322 s horizon and 42,304 integration stages. Typed bundle save/load and
same-runtime traced forward re-simulation reproduced the recorded stages.
Recording took 430.68 s; save 6.71 s, load 8.72 s, and verification including
rerun 569.59 s. The NPZ was 30,188,964 bytes; traced Python peak was 414.6 MiB
and observed process working set approximately 598 MB. This is not an admitted
best-match baseline or an independent-runtime replay; the fingerprint is
partial. It establishes neither actuator torque nor impact/contact or flight
validation. Bundle research checkpoint: `09e5a6647a`.

Numerical fitting research branch `codex/15493-infinite-bounds-research`
checkpoint `9802796bd9` supports correctly signed infinite bounds in the
existing equality least-squares solver without inventing finite limits.
It rejects invalid infinities and overflow; unknown bounds remain unknown and
are not a physical feasibility pass. Fifty-five focused cases, Ruff, Black,
scoped mypy and the normal headless pre-push hooks passed. This is a research
checkpoint, not a Slice 1 PR.

Slice 1 delivery preparation: branch `codex/15493-slice1-extraction` at
`6985e6c2e1` contains 93 changed lines against then-current main. It rejects
migration from a mismatched embedded model and prevents an unlanded flight
fixture from reporting carry. Both defects were demonstrated by failing tests
before the fixes. Focused tests: 39 passed, two strict expected failures,
using verified physics revision 3; Ruff, Black, scoped mypy and normal headless
pre-push hooks passed. This is a partial extraction, not a completed Slice 1
or a PR. The admitted pinned-v2 forward match, physical gates and bracketed
on-face contact evidence remain open; reconcile against current main before
continuing and retain the five-PR and under-500-lines constraints.

A private 477-frame fit probe replaced local finite-difference grip/foot
Jacobians with native derivatives. At three sampled states the directional
errors were 2.3e-10 to 1.7e-9, while the original and native whole-objective
values at the seed agreed within 5e-15. A whole-record directional Jacobian
check had relative error 1.45e-8. One same-process full assembly benchmark
measured 9.147 s for finite differences and 0.781 s for native derivatives
(11.71 times faster); total-objective gradient relative difference was
5.11e-7. These measurements are not an independent implementation audit.
A 120.47 s full-record trial saved 81 accepted iterates and did not converge:
marker RMS changed from 36.6081 to 36.6079 mm, maximum soft closure translation
from 0.00714 to 0.00971 mm, peak sampled speed from 33.326 to 33.325 rad/s,
and peak sampled acceleration from 2173 to 2126 rad/s². The native unscaled
projected-gradient infinity norm was 5.37. These are exploratory kinematic
results, not a stationary fit, an admitted baseline, or physical validation.
Further work must verify the native derivative mapping across the complete
feasible domain and use hard closure and defensible physical limits.

A same-seed 60 s LSMR sensitivity on the native-derivative soft fit compared
inner tolerance 1e-6/max 200 against 1e-8/max 1000. Both timed out without
stationarity; respectively they accepted 35 and 14 steps, with objectives
7.983665 and 7.984408 and projected-gradient infinity norms 5.2031 and 5.2290.
The tighter settings did not help in this limited comparison.

A nine-frame hard-equality pilot used the existing SLSQP equality solver with
all 54 native grip rows. It terminated successfully after 193 iterations in
34.44 s; whole-residual directional derivative relative error was 1.46e-9.
Independent original-frame checks found maximum closure translation
6.73e-8 mm and rotation 3.35e-10 rad; the equality Jacobian had full row rank
54 and condition number 9.69. The objective was 0.32457088834 versus
0.32457088852 for the earlier finite-difference run from the same seed.
The free-coordinate Lagrangian-gradient infinity norm was 0.00845; it omits
active-bound multiplier signs and is not a KKT certificate. This window has
35 active supplied bounds, free endpoints, assumed foot anchors, incomplete
anatomy and no qualifying forward replay. It is not an admitted baseline.
Test overlapping-window joins and a genuinely sparse full-trajectory constrained
method before considering this result transferable.

Research checkpoint `codex/15493-contact-root-kernel` at `810991550b` adds a
bracket refinement kernel with approaching and face-eligibility checks, with
10 focused tests including a MuJoCo sliding-face fixture; normal headless
push checks passed. Its 369 changed lines include the earlier extraction.
It does not yet integrate the real finite face or fixed-hold evaluator,
certify first-contact scan coverage, or implement a smooth miss penalty.
Research checkpoint `codex/15493-replay-horizon` at `12a806ea78` corrects a
same-input parity score that hid coordinate excursions between sparse frame
samples; 15 focused tests and normal headless push checks passed. Its frame
checks remain sampled, and it is not the controller-based physical admission
gate. Neither checkpoint is a PR or a complete Slice 1.

An adjacent nine-frame hard-closure pilot fixed the two shared states to the
first window, making their positions and shared discrete interval velocity
identical by construction. After one restart it reached native grip residual
infinity norm 7.75e-9 and independently evaluated original-frame closure
translation 5.09e-6 mm and rotation-vector norm 9.42e-9 rad, but the solver
hit its iteration limit. The reduced equality Jacobian had rank 42/42; the
combined peak sampled speed was 16.155 rad/s and the join second difference
1259.77 rad/s². Acceleration was not constrained. A bounded-multiplier
stationarity estimate was 0.0181; neither stationarity nor continuous-time
feasibility is established. The first-window multiplier audit likewise found
an estimated gradient infinity norm 0.00845. Assumed anchors, incomplete
anatomy, free ends and lack of a qualifying forward run remain limitations.

Native MuJoCo geometry inspection of the committed v2 plant found six foot
geometries and no club-body geometry, so its built-in geom distance cannot
query the actual driver face without adding geometry. The research rendered
face patch has 576 triangles, 64 boundary edges and approximately 20.04°
maximum triangle-normal spread; substituting a plane or rectangle would alter
the geometry. The compact root kernel must be connected to a validated
finite-patch evaluator and continuous fixed-hold advancement. The current
369-line extraction leaves about 130 changed lines under the requested PR
limit, while the existing research mesh handling alone is larger; find a
reuse-based implementation and recheck exact diff rather than dropping the
geometry or convergence requirements.

A three-state overlapping hard-closure window converged after one restart:
maximum native equality residual was 6.93e-10, original-frame closure
translation 3.57e-7 mm and rotation 7.63e-10 rad, with reduced Jacobian rank
36/36. Shared positions, both discrete velocities and one shared second
difference agreed exactly by construction. The immediately following second
difference changed by 1098.75 rad/s²; peak was 1266.10 rad/s². Estimated
bounded-multiplier Lagrangian gradient infinity norm was 0.01153 with 24
active bounds. This does not certify stationarity, acceleration continuity,
anatomical feasibility or forward tracking. With those three states fixed,
exactly matching the next discrete acceleration uniquely predicts the next
state; that state violated hard grip closure by 0.0274 mm and 1.21e-4 rad.
This local incompatibility does not rule out a joint window solve. Next test
a coupled sparse constrained trajectory with explicit continuity, and source
actual acceleration limits separately.

A private reuse probe compared installed optional trimesh 4.12.2 against the
research finite-face nearest-point implementation on 500 deterministic points
of the 576-triangle v2 patch: maximum per-triangle difference was 8.37e-16 m
and no nearest-triangle indices differed. One same-process 1000-point timing
was 0.168 s versus 0.280 s. This is sampled evidence, not equivalence proof;
trimesh is optional and its default vertex merging tolerance differs from the
research topology rules. Boundary semantics, continuous fixed-hold state,
approach velocity, scan coverage and smooth miss penalty remain open.

Direct model inspection found 47 degrees of freedom and no native actuators.
A private one-state plumbing probe added 41 synthetic unit-gear motors in a
copy of the compiled model, with arbitrary ±1 test-only limits. MuJoCo clipped
controls and mapped measured actuator forces to generalized coordinates with
zero maximum residual, while six root inputs remained zero; feeding those
forces to the existing constrained solver yielded finite acceleration. This
is not a physiological actuator model, a trajectory or a physical limit.
The existing solver consumes its supplied effort directly, so any future seam
must explicitly avoid double counting; physical actuator provenance and
post-saturation trajectory evidence remain missing and the gate stays closed.

Research branch `codex/15493-actuator-seam-research` at `6d934c440d`
adds a stateless point-evaluation seam using caller-supplied joint motors on
a copy of the base model. It passes MuJoCo's post-saturation generalized
actuator force exactly once to the constrained solve and rejects missing or
stateful actuators and passive-shaft drive. Five focused tests cover synthetic
control and force clipping, a non-unit transmission and no repeated-call
accumulation; Ruff, Black, scoped mypy, LaTeX compilation and normal headless
push hooks passed. This is not integrated with stage recording and supplies
neither physiological motor limits nor a validated actuator model. Its branch
is based on oversized research and is not a Slice 1 PR or physical admission.

A coupled 15-frame sparse trust-constr pilot optimized the shared window
states together, enforced all native grip rows and equated one pair of
sampled join accelerations. The first run stopped at 150 iterations without
convergence: native closure residual 5.41e-8, independently measured maximum
closure 1.78e-5 mm and 7.38e-8 rad, sampled acceleration-join mismatch
1.15e-10 rad/s², and reported optimality 0.00857. Supplied bounds held.
A same-objective restart reached 1000 iterations in 149.04 s without
convergence. Objective fell 0.54449 to 0.49525 and peak sampled second
difference 1410 to 1094 rad/s², while native closure residual rose to
2.17e-7 (original-frame 0.000113 mm, 2.95e-7 rad); sampled join mismatch
was 1.73e-10 rad/s² and reported optimality 0.00433. This local test uses
shared states and an explicit discrete join constraint, not global C2
interpolation; a Gauss–Newton objective and zero constraint-Hessian
approximation, assumed anchors, partial physical limits and no forward replay
prevent physical admission or a stationarity certificate.

A private synthetic-motion contact probe used the actual v2 576-triangle face,
fixed-lattice forward dynamics and point-Jacobian approach velocity. With
nonzero root rotation and 250/125 microsecond maximum substeps, both candidate
roots were at 0.000459850311 s, with q difference 2.22e-16, qd difference
1.57e-16, clearance 2.27e-9 m and approach speed 1.089914 m/s. Receding and
0.2 m lateral-offset controls returned no accepted candidate and finite
sampled penalties. The initial state was an exploratory fit, while velocity
and ball placement were synthetic; the one-millisecond fixture does not
establish pinned-match contact, complete scan coverage or a physical shot.

Reproducible research branch `codex/15493-realface-fixture` at `36a15de7c2`
adds a synthetic v2 finite-face regression with a closure state solved from
zero, no private fit input, fixed-lattice integration, step halving, receding
and off-face controls. One focused test and normal headless push hooks passed.
The 1 microsecond synthetic horizon does not certify actual-match contact,
first-contact coverage or physical gates. It is based on oversized research,
not a PR.

Research branch `codex/15493-actuator-rollout-research` at `0db6c92d5f`
adds a separately labeled post-saturation motor-drive recording mode for RK4
stages without silently changing ideal-drive semantics. Thirty-one focused
actuator, ideal-drive and stage-trace tests and normal headless push hooks
passed. A real-v2 synthetic two-step fixture required 0.1 microsecond steps;
a 1 ms trial diverged. This is short plumbing evidence only, with synthetic
motor limits, no admitted actuator model, and no impact or projection support
in that mode. It is not a qualified trajectory or compliant Slice 1 PR.

The unchanged pinned modeled-address ball was checked against the exploratory
v2 computed-torque rollout. Its only 1 ms sampled clearance sign change was
at 0.099–0.100 s and was separation after initial overlap. Re-integrating
the held input at 125 and 62.5 microsecond maximum substeps located the
separating root at 0.0990658750665 s with 1.37e-15 s between estimates and
gap-increase speed 0.272676745 m/s; the approaching-contact finder rejected
it. In the final 1 ms, 33 samples per resolution found no accepted candidate
and sampled clearances 0.169752–0.210776 m. Both checks are exploratory,
sampled evidence and cannot rule out unsampled crossings or establish contact
for an admitted match. The ball placement was not altered.

An independent non-draft PR #15552 at head `138feba99e37a9e03d19951a321ba30353924532`
was opened by another lane for a sampled total-applied-torque and adjacent
sample slew audit. Its stated scope does not include inter-sample behavior,
work, ROM, tracking or the matched-swing experiment. It is unarmed and has not
been integrated into this research; inspect its current state before reuse.
It is not evidence that the required five sequential slices are complete.

A dense SLSQP polish of the coupled 15-frame solution hit its 400-iteration
limit after 98.66 s. Objective changed 0.495250 to 0.492709; native grip
residual was 4.26e-9 and independently measured closure 1.22e-6 mm and
4.30e-9 rad. Sampled join-acceleration mismatch was 1.44e-10 rad/s², peak
sampled second difference 1049.50 rad/s², and supplied bounds held. A
bounded-multiplier estimate gave Lagrangian-gradient infinity norm 0.00345;
55 bounds were active at 1e-7, and equality Jacobian rank was 131/131
including all coordinates but 128/131 on free coordinates. This is neither a
converged or stationary fit nor physical or full-trajectory admission.

A matched-policy 200 ms exploratory v2 computed-torque comparison used zero
KKT regularization for both paths, no projection or impact, and synthetic
unit-gear motors. With ±1000 N m test-only caps, ideal and motor paths each
recorded 6400 stages, did not clip, and matched q, velocity and held commands
bit-for-bit; motor-stage force matched command. Both had 0.00812 rad joint
tracking RMS and 257.21 N m sampled peak command. With arbitrary ±30 N m
caps, all accepted motor stages clipped and the run hit a numerical guard at
0.189 s after divergent states. These are exploratory computed-torque results,
not Option B, physiological limits, or admission of the refitted baseline.

The matched-policy full pre-impact exploratory v2 run completed 1.322 s,
1322 holds and 42,304 actual RK4 stages for both ideal input and synthetic
±1000 N m unit-gear motors. Within this runtime, saved q, velocity, held
commands and every recorded stage q, velocity, command and applied drive were
bit-identical; no stage clipped and peak command/applied magnitude was
356.8718 N m. Joint target RMS error was 0.01532 rad, maximum 0.31992 rad.
Each pass took approximately 41 s. The run used computed torque, an
exploratory nonstationary refit, no projection or impact, zero KKT
regularization and arbitrary motor caps; it is not independent-runtime
replay, Option B, physical admission, or contact evidence.

The coupled 15-frame fit's active-set sensitivity found the same 55 active
bounds at thresholds 1e-9, 1e-7 and 1e-5, with full equality rank 131 but
free-coordinate rank 128. Treating these bounds as active makes equality and
active-bound gradients dependent, so multiplier estimates can be non-unique.
This local finding does not prove infeasibility or stationarity; actual
inequality complementarity, complete physical limits and forward tracking
remain open.

A full-horizon sensitivity mapped the existing directional model torque
priors to synthetic motor control ranges for all 41 non-root v2 coordinates.
Sixteen slots are explicitly assumed or unverified; subject and actuator
inventory remain unverified. With the same computed-torque controller,
1 ms hold, eight substeps, zero KKT regularization, no projection and no
impact, this run reached 1.250 s then hit a numerical guard after 40,000
accepted stages. Applied motor force matched directional clipping at each
recorded stage; 37 coordinates clipped in 5,792 stages (14.48%). Against
the corresponding ideal run, maximum coordinate difference first exceeded
0.01 rad at 0.008 s, 0.1 at 1.139 s and 1 rad at 1.166 s; speed exceeded
100 rad/s at 1.215 s. The divergent tail makes whole-partial-run RMS
misleading. This is model-envelope sensitivity, not a physical gate pass,
physiological limit, Option B result or proof of infeasibility.

A fixed-PD Option B motor-path sensitivity used gains 3/0.3, 1 ms ZOH,
eight RK4 substeps, the exploratory recorded-dynamics reference, no
projection or impact, and existing directional model-envelope capacities
(16 of 41 slots unverified). The zero-residual candidate hit a numerical
guard at 0.333 s after 10,656 stages; a preselected cubic residual with
1 N m coefficient norm failed at 0.265 s after 8,480 stages. Actual recorded
motor force matched directional clipping. On their common 250 ms prefix,
zero versus residual tracking RMS was 0.54085 versus 0.53655 rad; both had
20 modeled ROM flags, speed flags were 6 versus 5, and clipped stages were
1600 versus 1632 of 8000. Maximum held applied-input update rates were
111,246 versus 116,200 N m/s and stage-based positive work estimates were
435.87 versus 433.50 J. Both trajectories failed well before the endpoint;
this limited diagnostic is not a physical acceptance, complete optimization,
carry result or proof that no feasible improvement exists.

A separate 15-frame hard-closure trial fitted v2 FK markers to markers
computed from the original controller target `q_track`; the source dynamics
file hash matched the pinned bundle's dynamics-record hash. The 250-iteration
trial did not converge. Source-target marker RMS changed 19.04 to 15.30 mm,
while observed-marker RMS changed 31.01 to 32.91 mm and maximum coordinate
change was 0.439 rad. Native grip residual was 1.03e-5; original-frame
closure was 0.00371 mm and 1.27e-5 rad. Supplied bounds held, sampled join
acceleration mismatch was 1.44e-10 rad/s² and peak second difference was
714 rad/s². This is a short marker-space target proxy, not direct migration,
whole-swing tracking or physical admission.

An independent original-source-model check found `q_track` FK markers already
differed from observed markers by 28.34 mm RMS over those 15 frames and 487
valid marker samples. Thus a target-observation mismatch exists before v2
transfer, but this check does not apportion the v2 fitting tradeoff. The
private data, scripts and hash-bound receipts remain outside git.

Research delivery checkpoint `codex/15493-delivery-audit4` at `19523f3f3b`
adds separately labeled per-coordinate sampled controller-position-target RMS
and maximum errors to the existing same-input replay audit, preserving native
coordinate units. It binds the source dynamics record to the bundle hash
before simulation; a regression shows zero saved-feedback-rollout parity can
coexist with nonzero controller-target error. Twenty-two focused tests,
Ruff, Black, scoped mypy, LaTeX compilation and normal headless push hooks
passed. This is 208 changed lines against main `34c0d41e4a`; it is not a
qualified v2 baseline, physical gate, contact evidence or a PR. Combining it
wholesale with the current contact extraction exceeds the 500-line limit;
reconcile and reuse rather than claim a completed Slice 1.

A controlled zero-residual Option B gain sensitivity used fixed gains
1/0.1, 3/0.3 and 10/1 with the same exploratory recorded-dynamics reference,
1 ms ZOH/eight RK4 substeps and actual post-saturation force under the
assumption-marked directional model envelope. Numerical guards stopped the
runs at 0.399, 0.333 and 0.313 s. Over their shared first 250 ms, joint RMS
errors were 1.1783, 0.54085 and 0.31659 rad; modeled ROM flags were 26, 20
and 14, speed flags 12, 6 and 2, and clipped stages 1408, 1600 and 7328 of
8000. A repeat of the 3/0.3 control reproduced saved states and stage arrays
bit-for-bit in this runtime. None completed or qualified; the model limits
are not verified physiological limits, and this is not gain co-optimization
or proof of infeasibility.

A 33-frame sparse hard-closure fit to FK markers from the hash-verified
controller target hit its 350-evaluation limit after 125.36 s. Objective
changed 2.49949 to 0.34634; native grip residual was 1.33e-13 and independent
closure was 5.28e-11 mm / 1.61e-13 rad. One discrete acceleration-join
mismatch was 1.15e-10 rad/s², supplied bounds held, and reported optimality
was 7.99e-4. Euclidean target-marker RMS changed 19.08 to 15.93 mm while
observed-marker RMS changed 30.30 to 31.51 mm. Peak sampled speed was
32.68 rad/s and second difference 886.89 rad/s². This is not a converged or
admitted full-swing trajectory: anchors are assumed, anatomy is incomplete,
global continuity is unconstrained and forward dynamics were not verified.

A 99-frame target-marker-proxy sparse constrained run reached its 240.68 s
callback budget after 198 iterations without convergence. Objective decreased
7.80094 to 2.76869, but Euclidean target-marker RMS worsened 19.99 to
24.57 mm and observed-marker RMS 33.97 to 41.78 mm. The objective includes
strong assumed foot-anchor residuals, so the lower value does not establish
an improved match. Native closure residual was 9.29e-4, independent closure
0.3750 mm / 0.001134 rad; supplied bounds held, sampled acceleration join
mismatch was 5.76e-11 rad/s², peak speed 44.22 rad/s, second difference
2360.93 rad/s² and reported optimality 20.88. This is neither a physical or
stationary fit nor proof of infeasibility. Revise formulation and residual
weighting before spending more time on identical restarts; private receipts and
input hashes remain outside git.

The controller-target audit was exercised on the actual hash-verified pinned
source-v1 bundle. Its initial run exposed an endpoint defect: rounding to a
fixed 1 ms grid extends the final time from 1.8138889 to 1.814 s, and the
audit rejected it even though the controller clamps the reference. Research
branch `codex/15493-target-endpoint-fix` at `4754344ff0` handles only the
fixed-step rounding allowance, reports the clamped duration and rejects longer
extensions. A failing-first regression, 23 focused tests, Ruff, Black, scoped
mypy, LaTeX compilation and normal headless push hooks passed.

The corrected full source-v1 CLI run completed 1.814 s. Recorded-rollout
maximum coordinate parity error was 4.24e-10 rad and maximum frame error
1.49e-10 m, while controller-target RFInput had sampled RMS error 0.15628 rad
and maximum 0.84263 rad. The final target clamp was 0.000111111 s. This
shows recorded-rollout parity does not establish target tracking. It is not
the required committed-v2 forward match, a physical acceptance or a PR.

A revised 99-frame target-marker-proxy sensitivity used the same initial
seed, sampled hard grip rows and supplied bounds while changing the assumed
foot-anchor residual weight from 100 to 10. It stopped at its callback budget
(183.19 s, 92 iterations), not converged. Native grip residual was 1.36e-10,
independently measured maximum closure 5.95e-8 mm / 1.38e-10 rad, and the
one sampled acceleration join mismatch 1.73e-10 rad/s². Supplied bounds held.
Target marker component RMS changed 11.5407 to 9.1115 mm (Euclidean about
19.99 to 15.78 mm), observed component RMS 19.6095 to 19.9910 mm (about
33.97 to 34.63 mm), and assumed foot-anchor component RMS 0.1744 to
0.3460 mm. Peak sampled speed was 35.69 rad/s, second difference 957.43
rad/s² and reported optimality 1.79e-4. The objective differs from the
previous weight-100 trial and objective values are not directly comparable.
Assumed anchors, incomplete anatomy, only one discrete join, no forward
tracking and lack of a stationarity certificate prevent admission.

For the exploratory fixed-PD Option B reference, a loose synthetic ±1000 N m
motor-cap diagnostic completed 1.322 s and 42,304 stages with zero clipping
and exactly matched reference q and v, but still had seven modeled ROM flags.
A preselected 1 N m spline perturbation under the same loose caps hit a guard
at 0.515 s, showing that clipping is not the sole source of local sensitivity.
For that one perturbation, PD feedback dotted with the residual was negative
at 200 of the first 201 held samples; this is one unadmitted direction, not
proof that reference shifts are safe or generally beneficial. Neither run is
physical feasibility, optimization or carry evidence.

Under the original assumption-marked model envelope, the change in PD feedback
between the 1 N m residual and zero-residual runs opposed the spline residual
at 200 of 201 held samples through 0.2 s (mean incremental dot product
−0.03265 N²m²). Total feedback itself was mostly aligned because the
baseline tracking error was already large; these are different quantities.
A predeclared diagnostic then applied ±0.01 rad coefficient-L2 reference
shifts in the residual spline direction, with analytic velocity derivatives
from the same spline, fixed gains 3/0.3 and unchanged caps. Both shifted runs
hit the numerical guard at 0.265 s, as did the unshifted residual. On the
common 250 ms prefix, tracking RMS against the original reference was
0.536550, 0.536446 and 0.536655 rad for unshifted, positive and negative
shift; all had 20 modeled ROM flags, five speed flags and 1632 clipped stages
of 8000. This single-direction diagnostic establishes neither meaningful
feasibility nor power improvement and does not qualify the reference.

Research branch `codex/15493-derived-velocity-audit` at `c4324734b5` adds
per-coordinate controller-derived velocity-target errors to the position audit.
The pinned source record contains `q_track` and `track_time_s`, not a stored
velocity target; the implementation reuses the controller's `reference_rates`
and split-aware sampling, including the bounded fixed-step endpoint clamp.
It labels the rates as derived and preserves native units and provenance hash
checks. Twenty-five focused tests, Ruff, Black, scoped mypy, LaTeX compilation
and normal headless push hooks passed. The branch adds 131 changed lines past
endpoint-fix, approximately 382 against its original main; refresh before
review. It is partial audit infrastructure, not a complete Slice 1 or PR.

On stored source-v1 feedback states, the derived target comparison found
RFInput velocity RMS 4.87718 rad/s and maximum absolute residual 67.13571
rad/s at 1.326 s, about 3 ms after the recorded impact split. Sampled RMS was
0.08242 rad/s pre-impact and 9.36654 rad/s post-impact. This compares stored
simulation with controller-derived finite-difference rates, not measured
subject velocity, independent forward replay, or the committed-v2 baseline.

A predeclared five-point, two-direction Option B motor-path grid used fixed
gains 3/0.3, 200 ms, a 10 N m coefficient trust ball and actual stage-level
post-saturation forces under assumption-marked model limits. All five
candidates exceeded sampled modeled ROM and speed limits. Zero residual had
the best predeclared regularized diagnostic score (7.4117); the best tracking
sample improved RMS from 0.44482 to 0.43305 rad but still had 16 ROM and
three speed flags and 1440 clipped stages of 6400. A fresh-plant same-runtime
rerun reproduced its arrays bit-for-bit. Extended towards the full horizon,
it hit the numerical guard at 0.339 s; later failure than zero does not show
feasibility or shot improvement. These are not physiological or carry claims.

A 199-frame target-marker-proxy continuation, seeded partly by the revised
99-frame fit, stopped after 248 s and 84 iterations without convergence.
Component target RMS changed 11.686 to 9.307 mm (Euclidean approximately
20.24 to 16.12 mm), observed RMS 18.087 to 18.834 mm (about 31.33 to
32.62 mm). Native grip residual was 5.42e-8, independent closure
3.38e-5 mm / 8.37e-8 rad, and supplied bounds held. One sampled join
mismatch was 1.15e-10 rad/s²; sampled peak speed was 35.67 rad/s, second
difference 959.39 rad/s² and reported optimality 0.00306. A fresh 200 ms
computed-torque forward diagnostic of this refitted proxy completed 200
holds and 6400 ideal-drive stages with projection and no impact. Joint RMS
tracking error was 0.01764 rad, maximum 0.17582 rad; velocity RMS 0.4665
rad/s, maximum 3.8873 rad/s. Peak ideal generalized drive was 195.86 N m;
one named model ROM slot exceeded by 0.518° and modeled foot-sphere height
reached −9.72 mm. Preloading changed initial q by up to 0.00357 in native
coordinates. Assumed anchors, incomplete physical limits, projection, lack
of independent replay and tracking a proxy rather than original q_track mean
this is not an admitted baseline.

The complete 0.550 s extent of the exploratory 199-frame proxy was tracked
with computed torque, projection and no impact: 550 holds and 17,600
recorded ideal-drive stages. Joint q RMS error was 0.016995 rad, maximum
0.17582 rad; velocity RMS 0.4293 rad/s, maximum 4.0859 rad/s. Peak ideal
generalized drive was 395.04 N m and peak stage speed 34.94 rad/s. Two named
subtalar model ROM slots exceeded by 0.518° and 1.785°, and the lowest modeled
foot sphere reached −15.18 mm. Preload changed initial q up to 0.00357 in
native coordinates. This is not Option B, actuator evidence, original-target
full pre-impact tracking, physical admission or independent replay.

Research branch `codex/15493-boundary-vertex-research` at `08e2da96e8`
fixes a finite-face classification defect: an outer corner was accepted as
interior when an incident interior triangle won the nearest-triangle tie.
An end-to-end failing-first regression showed the old implementation accepting
a synthetic 0.5 s corner root; the fix rejects it while retaining an interior
vertex control. Thirty-nine focused contact tests, Ruff, Black, LaTeX
compilation and normal headless push hooks passed. It remains an oversized
research branch, not a PR or proof of first contact on the pinned swing.

A bounded local two-direction Option B quadratic surrogate used four central
probes and a 10 N m coefficient trust ball with actual post-saturation stage
forces under assumption-marked model priors. Its selected 2.301 N m candidate
changed the 200 ms regularized diagnostic score from 7.41166 to 7.35845 and
tracking RMS from 0.44482 to 0.43876 rad, but retained 17 modeled ROM flags,
three speed flags and 1440 clipped stages of 6400. Extended toward 1.322 s,
it reproduced its first 200 ms bit-for-bit but hit a numerical guard at
0.265 s. This is not a feasible candidate, physiological result or shot
improvement.

The 199-frame target proxy's fitting objective constrained horizontal foot
anchors but had no height row. Kinematic evaluation found modeled foot-sphere
gap below the assumed ground in 25 of 199 frames, worst −0.453 mm. A revised
sampled six-sphere nonpenetration-inequality sensitivity stopped at its
180 s callback budget after 194 s without convergence; minimum sampled gap
became +4.390 mm, but grip closure worsened to 0.274 mm / 0.001004 rad and
target component RMS changed 9.307 to 9.366 mm. In a like-for-like 0.55 s
projected computed-torque replay, minimum modeled foot height was −16.57 mm
(previous proxy −15.18 mm), despite positive sampled fit gaps. Preload
changed initial q by up to 0.00777 native units (previous 0.00357); joint RMS
was 0.01276 rad, peak ideal drive 304.56 N m and a modeled subtalar flag
reached 3.04°. These are assumed-ground diagnostics with nonconverged,
non-admitted fits and ideal unsaturated drive. Sampled position constraints
alone did not prevent dynamic penetration under these preload/support
semantics; they do not establish measured subject contact or feasibility.

Research branch `codex/15493-contact-coverage-research` at `8cc3d9de0f`
adds a fail-closed adaptive clearance scan prototype (213 added lines beyond
the boundary fix). Seven new and 39 existing contact tests, scoped
NumPy-aware mypy, Ruff, LaTeX compilation and normal headless push hooks
passed. A narrow real-triangle crossing missed by coarse endpoints is detected;
earlier ambiguous intervals, initial overlap and budget exhaustion remain
unresolved rather than certifying a later hit. The scanner is not wired into
the event finder, relies on a caller-attested continuous whole-surface speed
bound that samples cannot validate, and has no interval rounding. It does not
certify first contact, the pinned swing or a physical shot. A separate broad
mypy invocation reported unrelated Qt document-reader errors; normal
repository mypy and the genuinely scoped check passed.

The assumed v2 stance uses compliant Hunt–Crossley sphere contact with
50,000 N/m per sphere, not a rigid nonpenetration constraint. `preload_feet`
sets nominal equal-load static compression of 2.584 mm for the modeled
79.06 kg mass and six spheres, then closure projection can alter q. At the
worst sampled 0.516 s state, the previous and ground-inequality forward runs
had minimum gaps −15.185 / −16.568 mm and modeled total normal force
1528.5 / 1668.0 N (1.97 / 2.15 bodyweights); their sampled maximum totals
were 1758.4 / 1942.5 N. At that time their interpolated kinematic references
had positive modeled clearance +2.78 / +5.39 mm. Across hold-end grip
projections, minimum gaps were unchanged to printed precision; maximum q and
v projection changes were 3.65e-12 and 4.20e-10. Compression is expected in
this compliant model and is not automatically a rigid-contact violation;
these model-dependent forces are not measured ground forces or acceptance.
The reference/forward difference calls for analysis of support dynamics,
root motion and preload rather than assuming the sampled fit alone enforces
stance.

A targeted knee/right-hip Option B local surrogate selected a candidate with
7.1586 N m coefficient norm inside the 10 N m trust ball. Over 200 ms it
changed tracking RMS 0.44482 to 0.38316 rad and the regularized diagnostic
score 7.41166 to 6.82228, but had 17 modeled ROM flags, three speed flags
and 1440 clipped stages of 6400 under assumption-marked priors. Its extended
run reproduced the first 200 ms stage arrays bit-for-bit then hit a numerical
guard at 0.266 s, with 22 ROM and seven speed flags. This is not a feasible
candidate, physical acceptance, or a carry/power improvement.

Research branch `codex/15493-coverage-event-research` at `e2860862d1`
integrates the optional conditional adaptive scan with the existing finite-face
event finder. It rejects a later hit when an earlier interval is unresolved,
and detects a narrow crossing missed by coarse endpoints. Synthetic fixed-
lattice tests include the actual committed v2 face and step halving. Fifty-one
focused contact/coverage/v2-face tests, scoped NumPy-aware mypy, Ruff, Black,
LaTeX compilation and normal headless push hooks passed. The continuous speed
bound remains caller-attested, floating-point evaluation is not interval
arithmetic, and the higher-level ImpactForce trigger does not opt in. This is
not a first-contact certificate, pinned-swing result, complete Slice 1 or PR.

The separate broad Python 3.12 mypy invocation for the preceding coverage
checkpoint eventually exited 1 with 96 errors in 34 unrelated imported files;
no changed contact-module error appeared. The normal repository mypy hook and
scoped NumPy-aware check passed.

A vertical force-balance audit of the 199-frame exploratory v2 target proxy
used modeled foot support and COM acceleration estimated from Jacobian
velocity. Mean absolute vertical residual was 1.016 bodyweights (RMS 1.143),
with zero modeled normal force in 92.3% of interior samples. Applying the
same audit to the corresponding 0.55 s forward rollout gave mean 0.000829
and RMS 0.001074 bodyweights. Target acceleration from position second
differences versus Jacobian-velocity derivatives differed by at most
0.044 m/s² on interior samples. Uniform downward offsets of nominal static
compression 2.584 mm and 15.185 mm gave mean mismatch 0.911 and 1.524
bodyweights; neither is a dynamic correction. A per-frame root-translation
force match found roots for 197/199 samples, but when velocity and
acceleration were recomputed its mean mismatch became 3.97 bodyweights,
peak modeled normal force 18.73 bodyweights and peak offset second derivative
714.6 m/s². These are model-dependent, assumed-ground diagnostics, not
measured forces, physiological impossibility or an admitted trajectory. They
support using coupled smooth dynamics and support constraints rather than
independent pose adjustments.

A 41-coordinate saturation-deficit Option B direction with 10 N m coefficient
trust lowered clipped stages from 1440 to 1376 over 200 ms, but tracking RMS
worsened 0.44482 to 0.51337 rad, modeled ROM flags 17 to 18, speed flags
3 to 4 and the regularized score 7.41166 to 9.66887. Extension reproduced
the first 200 ms stage arrays bit-for-bit then hit a numerical guard at
0.437 s with 24 ROM and eight speed flags. The later failure is not a feasible
or improved shot. Limits remain assumption-marked, and no physical or
infeasibility claim follows.

A coupled 199-frame vertical-offset diagnostic used sparse least squares,
fixed joint coordinates, the same assumed compliant ground, and explicit
numerical offset/speed regularizers. It reduced sampled COM vertical force-
balance residual RMS from 1.052 to 0.004536 bodyweights, independently
recomputed with COM Jacobian/velocity, but stopped at 30 evaluations with
optimality 0.295 and did not converge. Offsets ranged −16.13 to +0.23 mm,
RMS 9.33 mm; peak offset acceleration was 150.7 m/s² and modeled normal
force 2.65 bodyweights. The 0.550 s projected computed-torque forward replay
completed 550 holds / 17,600 ideal-drive stages, no impact. Compared with the
prior proxy, lowest modeled foot sphere was −13.59 versus −15.18 mm, joint
RMS tracking 0.01854 versus 0.016995 rad, peak ideal drive 381.52 versus
395.04 N m, and worst modeled subtalar ROM exceedance 7.40° versus 1.785°.
Root translation RMS was 19.72 mm; preload changed initial q up to 0.00612
native units. The sampled balance reduction did not produce an admitted or
better rollout: the offset was not jointly optimized, regularizers and ground
are assumptions, forces are unmeasured and drive is ideal unsaturated.

Research branch `codex/15493-impact-coverage10` at `178b591940` integrates
an opt-in conditional coverage policy into ImpactForce. Its chronological
coverage clock and serialized policy fail closed on missing or previously
unresolved intervals; a synthetic actual-v2-face fixture exercises the
higher-level schedule. Ninety-nine focused tests, scoped NumPy-aware mypy,
Ruff, Black, LaTeX compilation and normal headless push hooks passed.
Coverage starts at arming time and assumes a caller-attested continuous
whole-surface speed bound and physically consistent contiguous advance states;
timestamp continuity alone does not prove the latter. Floating-point
calculations are not interval arithmetic. The result is conditional, not a
first-contact certificate, pinned-swing contact, completed Slice 1 or PR.
The branch is based on oversized research.

The opt-in ImpactForce policy was exercised with the unchanged pinned
modeled-address ball and exploratory v2 trajectory. Initial finite-face gap
was −12.704 mm (overlap); with illustrative caller bounds 10, 100 and
1000 m/s, the policy remained unresolved and did not schedule an impact.
Skipping ahead stayed blocked. Across 1323 stored grid states and 42,304
RK4 stage states, maximum sampled speed of the 321 face vertices was
52.174 m/s at grid time 1.317 s; a point-Jacobian vector agreed with finite
difference there to 3.3e-8 m/s. This contradicts the illustrative 10 m/s
bound at a sample but does not prove 100 or 1000 m/s bounds; RK4 stages are
not necessarily physical path states, and no continuous global bound or
pinned-swing first contact is established.

A feasibility-prioritized two-direction Option B surrogate confirmed zero
sampled modeled ROM/speed flags at its initial state. With a 10 N m trust
ball and assumption-marked motor limits, its selected 3.3045 N m candidate
changed the 200 ms diagnostic score 2.95782 to 2.84732 and tracking RMS
0.44482 to 0.41842 rad, but retained 17 modeled ROM and three speed flags and
1440 clipped stages of 6400. The extended run reproduced the first 200 ms
stage arrays bit-for-bit in this runtime then hit a guard at 0.263 s, with
22 ROM and nine speed flags. It is not a feasible full-horizon candidate,
physiological assessment or shot improvement.

A coupled vertical plus bilateral ankle support sensitivity, initialized
from the vertical-only diagnostic, stopped after 12 evaluations / 32.7 s
without convergence. Its regularized objective changed 0.87069 to 0.48761
while sampled support-balance RMS worsened 0.00454 to 0.01173 bodyweights;
maximum ankle adjustment was 2.272° and grip closure remained 3.38e-5 mm /
8.37e-8 rad. A like-for-like projected 0.55 s computed-torque replay
completed 550 holds / 17,600 stages. Compared with vertical-only, q RMS
was 0.02196 versus 0.01854 rad, peak ideal drive 449.85 versus 381.52 N m,
worst named ROM exceedance 9.36° versus 7.40°, and minimum modeled sphere
height −13.98 versus −13.59 mm. Initial preload shift was 0.00429 native
units. This assumption-dependent, unsaturated ideal-drive proxy is not a
qualified baseline or an improvement; numerical regularizers and trust
bounds are not physical limits.

A 0.100–0.110 s post-separation ImpactForce test explicitly rearmed the
conditional scanner and used illustrative 10, 100 and 1000 m/s bounds.
All ten 1 ms intervals returned conditionally clear with no event for each
bound; end-state q differences versus the saved states were zero and maximum
v difference 5.94e-14. Runtime was 106.5 s. Rearming discards the known
unresolved initial overlap; the bounds are not verified continuous bounds,
and 10 m/s is contradicted by a later sampled face speed. This is neither a
first-contact result nor a global no-contact proof.

At the same 199-frame samples, Euclidean target-marker RMS was 16.120 mm for
the original proxy, 18.629 mm for the vertical-only support adjustment and
17.236 mm for the coupled vertical/ankle adjustment; observed-marker RMS was
32.621, 33.738 and 33.066 mm respectively. The coupled adjustment is closer
to markers than vertical-only but farther than the original proxy, while its
forward tracking, modeled ROM and ideal drive worsened. None is admitted.

A targeted ankle/lumbar Option B surrogate used fixed gains 3/0.3, the
assumption-marked motor envelope and 10 N m coefficient trust. Its 7.828 N m
candidate changed 200 ms joint RMS 0.44482 to 0.26871 rad, modeled ROM/speed
flags 17/3 to 11/0 and modeled ROM RMS 0.29398 to 0.15878 rad; 1440 of 6400
stages still clipped. The 1.322 s extension reproduced the first 200 ms stage
arrays bit-for-bit in this runtime, then hit a guard at 0.531 s with 24 ROM
and eight speed flags. Retrospective 0.5× and 1.2× scales stayed inside trust
but still failed ROM. This is not a feasible or physically qualified swing,
independent replay, carry or power improvement; 16/41 capacities are
unverified.

A coupled 199-frame root-height, bilateral ankle and knee sensitivity stopped
at 20 evaluations / 98.18 s without convergence. Modeled support-balance RMS
changed 0.01173 to 0.001582 bodyweights; grip closure stayed 3.38e-5 mm /
8.37e-8 rad, with max ankle/knee shifts 2.60° / 1.75°. Target/observed marker
RMS were 16.584 / 32.773 mm. A projected 0.55 s computed-torque replay
completed 550 holds / 17,600 ideal-drive stages, but relative to the prior
ankle/root adjustment q RMS worsened 0.02196 to 0.02630 rad, peak ideal
drive 449.85 to 507.91 N m, worst model ROM excess 9.36° to 13.47°, and
lowest modeled foot height −13.98 to −15.30 mm. It is an assumption-dependent
nonconverged proxy, not a physical or actuator-qualified baseline.

Research branch `codex/15493-model-binding12` at `b3a2ce208e` addresses a
proven target-audit provenance gap: the dynamics-record digest could match
while a same-width run model differed from the bundle model. The guard
reconstructs the recorded rigid/chain shaft configuration and compares
canonical model JSON before simulation. Failing-first different-model and
chain-policy regressions, 27 focused tests, Ruff, Black, scoped mypy, LaTeX
compilation and normal headless push hooks passed. Run/bundle equality does
not independently prove the source record was generated by that model. The
whole audit branch is 466 changed lines against fetched main, without real
contact or qualified v2 baseline; it is not a completed Slice 1 or PR.

A full 477-frame target-marker-proxy sparse fit used hard sampled grip rows,
assumed foot anchors, supplied bounds and one discrete join. It stopped at
its 242 s callback budget after 55 iterations without convergence. Target
marker component RMS changed 13.906 to 11.327 mm (Euclidean approximately
24.09 to 19.62 mm); observed component RMS 17.647 to 16.170 mm (about
30.57 to 28.01 mm). Native equality infinity residual was 5.49e-6;
independently measured max closure 0.00445 mm / 6.97e-6 rad, sampled join
mismatch 8.63e-11 rad/s², supplied bounds held, peak sampled speed 35.78
rad/s, second difference 1026.44 rad/s², reported optimality 0.00183.

A full 1.322 s projected, no-impact computed-torque forward diagnostic of
that nonconverged proxy completed 1322 holds / 42,304 actual ideal-drive
stages. Joint q RMS error was 0.016355 rad, maximum 0.17509 rad; velocity
RMS 0.37287 rad/s, maximum 4.593 rad/s. Root translation RMS was 0.10572 m,
root rotation RMS 0.14282 rad, peak ideal drive 383.77 N m and peak speed
35.07 rad/s. Two modeled subtalar ROM slots exceeded by 5.06° / 3.68°;
minimum modeled foot-sphere height was −23.94 mm under compliant ground and
modeled support ranged 0.165–1.989 bodyweights. Initial preload changed q by
up to 0.003817 native units. This is not Option B, actual actuator force,
original-coordinate migration, measured stance, actual contact or an admitted
baseline; fitting did not establish global continuity or complete anatomy.

The unchanged modeled-address ball overlaps the actual finite face at the
hash-verified source controller target initial pose by 1.975 mm and at the
source saved-feedback initial pose by 1.483 mm, compared with 12.704 mm in
the exploratory v2 initial pose. The placement derives from observed
address-face geometry, not an independently measured ball. On the 1815
source-v1 saved 1 ms states the only sampled overlap transition was separation
at 0.054–0.055 s; minimum sampled clearance from 0.100 through 1.322 s was
6.904 mm. A late re-armed conditional scan 1.320 s to the legacy force
onset at 1.3246166437 s found no candidate under an illustrative unverified
100 m/s bound. Neither scan covers earlier overlap or proves first contact;
no ball placement was changed.

A predeclared two-phase Option B test retained the 7.828 N m ankle/lumbar
coefficient and added ±3 N m later spline coefficients, total norm within
10 N m. Five 400 ms trials completed, zero late coefficient had the lowest
predeclared score, and all had 23 modeled ROM / seven speed flags. A selected
+3 N m hip extension reproduced its first 400 ms stage arrays in the same
runtime then hit a guard at 0.531 s. Model limits remain partly unverified;
none is a feasible or improved shot.

An independent 477-frame marker-space audit used 15,733 valid samples. The
fitted proxy versus source target had 19.62 mm Euclidean RMS, whereas the
projected ideal-drive forward rollout versus source target had 269.65 mm;
fit versus observations was 28.01 mm, forward versus observations 270.35 mm,
and forward versus fit 268.56 mm. This exposes substantial world/root drift
that the approximately 0.016 rad joint-coordinate RMS alone concealed. It
confirms the exploratory full-horizon rollout is not a forward-matched
baseline, independent of its incomplete physical gates.

The unchanged modeled-address ball was reconstructed with
`ImpactForce.at_address` at the saved source initial pose: centre and passage
point matched the pinned record exactly. This supports a modeled-address
origin but does not independently prove historical generation chronology.
The convention offsets by ball radius along the 3D face normal then resets
ball height to face-centre height. In a planar diagnostic this loses normal
clearance by r*n_z²: 1.326 mm at source pose and 0.680 mm at exploratory v2.
Actual finite-mesh own-address overlaps were 1.483 and 0.852 mm; the source
face-centre landmark was 0.210 mm from the rendered mesh. Keeping the full
normal offset is a counterfactual, not authority to move the pinned ball;
no placement was changed and no contact is certified.

A targeted initial-saturation Option B diagnostic added −5 N m on the
assumed shoulder-blade cap deficit to the prior ankle/lumbar direction,
coefficient norm 9.28875 N m within trust. Initial command changed 29.779 to
24.779 N m against an assumed 25 N m cap; over 200 ms clipped stages fell
1440 to 1376 but tracking RMS worsened 0.26871 to 0.27204 rad and 11 ROM /
zero speed flags were unchanged. It hit a guard at 0.585 s with 25 ROM and
17 speed flags. This is not feasibility, physiological evidence or an
improved shot.

The existing lower-limb planted-coupling root regulator was tested with fixed
gains (20,5) on the full nonconverged 477-frame proxy, using zero direct root
effort in both runs, projected ideal unsaturated computed torque, no impact.
Root translation component RMS changed 0.10572 to 0.08516 m; root rotation
RMS 0.14282 to 0.14295 rad and joint RMS 0.016355 to 0.01731 rad. Using the
same 15,733 valid samples, source-target marker RMS changed 269.65 to
234.60 mm and observed-marker RMS 270.35 to 235.38 mm. Both remain far too
large for a match. Held joint effort peak changed 383.77 to 359.62 N m;
hold-sampled named ROM flags changed from two (5.055° / 3.676°) to one
(4.080°), and minimum compliant foot-sphere height from −23.94 to −24.87 mm.
These are unadmitted model-dependent diagnostics, not Option B, actual
actuator force, measured support or physical feasibility.

For the root-regulation diagnostic, root-translation error norm against the
fitted proxy first exceeded 50 mm at 0.438 s versus 0.417 s unregulated,
and 100 mm at 0.626 versus 0.601 s; final norm was 0.288 versus 0.424 m.
A fresh-process repeat under the same runtime reproduced time, q, velocity
and held effort arrays bit-for-bit. No accepted-stage trace or independent
runtime was checked. This remains substantially off the target and is not a
qualified baseline, physical feasibility or Option B result.

A fixed-step resolution sensitivity repeated the 0.585 s Option B candidate
at 8, 16 and 32 RK4 substeps per 1 ms hold under the same model-envelope
motors. All hit the guard at 0.585 s with 25 modeled ROM / 17 speed flags.
Maximum |q16−q32| at 0.2 / 0.5 / 0.55 / 0.58 s was 1.42e-7 / 5.73e-7 /
2.05e-4 / 3.82e-4 rad; maximum |v16−v32| at 0.55 / 0.58 s was 0.00720 /
0.00890 rad/s. Finer differences decreased, but a shared guard time is not
whole-trajectory convergence or physical qualification.

A subsequent within-trust earliest-limit perturbation scan extended three
candidates toward the full horizon. Wrist, subtalar and combined candidates
hit guards at 0.521, 0.506 and 0.516 s versus 0.585 s for the prior
candidate; they had 23 modeled ROM and 8 / 11 / 10 speed flags. A short
tracking benefit did not survive full-horizon validation. Model priors remain
partly unverified; this is not proof of infeasibility or an improved shot.

Two predeclared full-horizon lower-limb root-regulator sensitivities used no
direct root torque, projected ideal unsaturated computed torque and no impact.
Gains (50,10) gave root translation RMS 0.08616 m, root rotation 0.17569 rad,
joint 0.03703 rad, peak held effort 393.16 N m and source-target marker RMS
254.75 mm. Gains (100,20) gave 0.09314 m, 0.12405 rad, 0.03617 rad,
800.95 N m and 234.22 mm. The prior (20,5) gave 0.08516 m, 0.14295 rad,
0.01731 rad, 359.62 N m and 234.60 mm. All remain far from a match and
retain a modeled ROM flag; the tiny 100/20 marker reduction comes with more
than twice the effort and worse joint tracking. These are exploratory, not
Option B, measured support or physical acceptance.

Research branch `codex/15493-leadin-clock15` at `68c6efe35b` fixes a
confirmed target-audit alignment defect for generated lead-in bundles: the
bundle's recorded trajectory is trimmed and clock-shifted, but the prior
closed-loop audit started and scored at original time zero. The correction
repeats the existing no-impact fixed-step lead-in, scores the trimmed window
against position and derived-velocity targets at the elapsed clock, reports
the offset, and rejects inconsistent metadata before simulation. A failing-
first regression, 37 focused CLI/lead-in tests, scoped mypy, Ruff, Black,
LaTeX compilation and normal headless push hooks passed. The branch remains
oversized research and does not establish the v2 baseline, physical gates,
contact, a completed Slice 1 or a PR.

The existing all-six-spheres planted-coupling assumption was audited at 134
equally spaced saved hold states. Only 1–5 spheres in the baseline and 1–6
in the regulated run carried >1 N modeled load, with means 2.84 / 2.77;
all-sphere centre RMS speed was 0.430 / 0.716 m/s and loaded-sphere horizontal
RMS speed 0.123 / 0.076 m/s. These sampled model values show that all-six
stationarity is not satisfied, not measured physiology or causal proof.
A load-weighted coupling diagnostic used square-root modeled normal-force
weights and a >1 N activity threshold, fixed gains 20/5, no direct root torque,
projection, ideal unsaturated computed torque and no impact. At 1.322 s,
source-target marker RMS changed 234.60 to 212.74 mm, root translation RMS
0.08516 to 0.07436 m and rotation 0.14295 to 0.13873 rad. Joint RMS worsened
0.01731 to 0.03885 rad, peak held effort 359.62 to 790.28 N m, model ROM
flags one to three; minimum modeled compliant sphere height changed −24.87
to −18.92 mm. The experimental threshold and weights are not physical
limits, and the result is neither an admitted baseline nor Option B.

A 41-coordinate local 20 ms trust-region feasibility probe used fixed gains,
actual post-saturation motor forces and assumption-marked model limits. Its
9.3058 N m coefficient candidate had zero modeled ROM/speed violations at
1 ms hold samples, but an RK4 stage first exceeded a modeled ROM limit at
13.625 ms; peak stage excess was 1.05e-5 rad. At 200 ms it retained 11 ROM
flags and extended only to a numerical guard at 0.593 s with 26 ROM / 23
speed flags. Integration-stage values are not independently certified
continuous-path extrema; the result nevertheless disproves a hold-sample-only
pass and is not a feasible or physical swing.

Research branch `codex/15493-delivery16` at `c02b391c13` fixes a
same-input replay-scoring defect: shifted, nonmonotonic or nonfinite clocks
and nonfinite q could previously be accepted while assigning a fixed-grid
horizon. The guard requires finite state and the bundle time grid, with
bounded roundoff for lead-in subtraction. Failing-first regressions, 18
focused tests, Ruff, Black, scoped NumPy-aware mypy and normal headless push
hooks passed. It is 78 changed lines against then-current main, partial audit
infrastructure rather than a Slice 1 PR. It overlaps scoring files with
`codex/15493-replay-horizon` and must be combined and retested deliberately.
The legacy `max_coordinate_error_rad` scalar still mixes translation metres
and rotational radians, and frame checks remain sampled; neither is fixed by
this checkpoint.

As observed at the time of the delivery review, #15552, #15564 and #15573
were open and unarmed. The latter two independently address portions of a
bundle schema and optimization runner but explicitly omit required replay /
migration or force-attestation / comparison work. Inspect current PR heads
and status before reuse; they do not establish the requested sequential five
slices as complete.

A 41-coordinate 200 ms Option B local linearization used fixed gains, actual
post-saturation forces, a 10 N m coefficient trust ball and hold-sampled
modeled ROM/speed constraints under partly unverified capacities. The solver's
initial output exceeded trust by 3e-12 N m and was scaled inward before
nonlinear validation. At recorded integration stages, peak modeled ROM excess
changed 1.319 to 0.602 rad and ROM flags 11 to six; joint RMS tracking error
changed 0.2720 to 0.1651 rad. The candidate still violated a modeled ROM
limit from 9.94 ms, clipped in 1440 of 6400 stages and hit a numerical guard
at 0.288 s on extension, with 19 ROM / three speed flags. This is a one-sided
local diagnostic, not a feasible full swing, continuous-time certificate,
physiological assessment or power improvement. Private hash-bound receipts
and arrays remain outside git.

Research branch `codex/15493-score-units16b` at `28f34a9f91` fixes the
legacy mixed-unit replay score and invalid-clock/state acceptance. It classifies
coordinates from model joint primitives, fails closed for missing or ambiguous
metadata, reports rotation in radians and translation in metres separately,
and applies numerical-parity thresholds 1e-6 rad and 1e-6 m to every recorded
q sample. Frame checks remain sampled. Replay and closed-loop receipts use v2
schemas to distinguish corrected semantics from historical mixed-unit v1
receipts. Failing-first regressions, 61 focused tests, Ruff, Black, scoped and
broad mypy, LaTeX compilation and normal headless push hooks passed. The
254-line branch overlaps earlier scoring/clock branches and needs reconciliation;
it is not a physical gate, continuous-time certificate or completed Slice 1.

A 500 ms 41-coordinate Option B linearized continuation predicted reduced
hold-sampled modeled violation, but nonlinear validation hit a numerical guard
at 0.277 s versus 0.585 s for its starting candidate. At 200 ms recorded stage
ROM peak worsened 1.319 to 2.317 rad, speed flags zero to three and tracking
RMS 0.2720 to 0.3113 rad. The complete sensitivity matrix and candidate were
saved before a partial-horizon comparison harness error; limits remain partly
unverified. This is not feasible, a full-horizon result or an improved shot.

Upstream #15564 is now merged at `675d34feec3436e003e2363cd5ced6757dfa80e6`.
Its limited bundle-schema scope alone does not establish the required complete
Solution Bundle, migration or independent forward replay; check current main
and remaining requirements before reuse.

A predeclared nonlinear backtracking check along the 500 ms 41-coordinate
Option B surrogate direction used fractions 0, 0.125, 0.25 and 0.5, fixed
gains and the same assumption-marked motor caps. Zero and 0.125 completed
500 ms but had peak modeled ROM excess 8.062 / 8.032 rad and speed excess
92.00 / 129.91 rad/s; 0.25 and 0.5 hit guards at 0.349 / 0.467 s. On the
common 349 ms prefix the one-sided linear prediction's max q errors were
0.0395 / 0.5364 / 1.4464 rad for nonzero fractions, with max velocity errors
1.22 / 19.48 / 48.58 rad/s. This local sensitivity is severely infeasible
and shows the surrogate is unreliable over these steps; it does not prove
there is no feasible improvement.

An alternative root diagnostic used a weighted least-squares inverse of the
existing compliant-contact/weld affine dynamics, actuated joint torques only
and no direct root torque. At 200 ms, strong numerical root weight 10,000
reduced translation RMS to 4.34 mm and rotation RMS to 0.00118 rad, versus
8.74 mm / 0.00953 rad for the prior planted regulator, but peak ideal held
effort rose to 504 N m and two modeled ROM flags remained. Its full-horizon
extension hit a numerical guard at 0.805 s with partial joint RMS 0.212 rad,
peak ideal held effort 64,437 N m and 11 modeled ROM flags. This is an
unconstrained, projected computed-torque diagnostic, not Option B, physical
feasibility or a baseline improvement. Private receipts remain outside git.

Research branch `codex/15493-main-bundle17` at `84a8c9f155` bridges the
merged bundle codec and fresh closed-loop replay runner, with separate recorded
playback and explicit per-quantity boundary q/v/command/time comparisons. A
serialized two-link ODE fixture reruns with fresh factories in the same
runtime. It fails closed outside a no-impact, no-saturation, stateless fixed-
ZOH subset without external assets. Factories, model and runtime are caller-
attested; actual applied torque, impact and independent runtime are unverified,
and full migration, actual v2 and complete replay remain missing. A tiny-dt
fixed-grid tolerance defect was reproduced and fixed test-first. Thirteen
focused bridge tests, 115 combined neighboring tests before that additional
case, scoped mypy, Ruff, Black, LaTeX compilation and normal headless push
hooks passed. The branch has 499 insertions against fetched main `d73e7e933e`;
refresh main and reconcile before reuse. It is not a completed Slice 2 or PR.

Research branch `codex/15493-bundle-attest18` at `1bb66374f5` adds
measured-local checks of the supplied model file bytes and a declared partial
runtime fingerprint before replay factory construction. Ninety-six focused
bridge/codec tests, scoped mypy, Ruff, Black, LaTeX compilation and normal
headless push hooks passed. This does not prove a factory consumed that model,
cover all backend/controller/native dependencies, verify post-actuator torque
or impact, or establish independent replay. The initial test-first attempt
failed at collection due to an uninitialized pinned dependency; a functional
red phase was not observed. The 172-line increment is atop the 499-line
research bridge and is not a complete Slice 2 or PR.

Sequential local Option B relinearization used fixed gains, actual motor
clipping, partly unverified priors and a 10 N m global coefficient trust.
The strict first 0.5 N m local step changed 200 ms stage peak modeled ROM
excess 0.60196 to 0.22128 rad and joint RMS 0.16512 to 0.09674 rad, but
five ROM slots still failed and 1440/6400 stages clipped; extension hit a
guard at 0.631 s. A second local step lowered 200 ms peak to 0.10405 rad
and joint RMS to 0.08979 but still had five ROM flags and hit a guard at
0.594 s. A later 500 ms relinearization reduced stage ROM peak 6.220 to
4.572 rad but increased speed flags seven to 13 and clipped stages 1440 to
1504 of 16,000; its full extension failed at 0.568 s. These remain infeasible
model diagnostics, not physical or shot improvements.

The existing contact-wrench QP was exercised through a private Clarabel
adapter because its Drake wrapper was unavailable; three toy scenarios and
residual checks do not prove equivalence to Drake. The full 1.322 s
projected ideal computed-torque run made 1322 controller calls with zero
direct root effort; maximum reported normalized cone slack was 0.535,
source-target marker RMS 272.77 mm versus 269.65 mm unregulated, root
translation RMS 0.10534 m, joint RMS 0.01650 rad and peak held ideal effort
384.28 N m. Existing controller fallback can silently handle solver failure
or missing support; solve status was not captured for every call, so this is
not evidence that every QP solved, nor physical or Option B qualification.

Research branch `codex/15493-wrench-audit18` at `4c270bd466` adds an
opt-in strict contact-wrench audit with caller-selected finite normalized
slack tolerance. It fails closed on missing support, failed or malformed QP
solutions and excessive slack; default fallback remains available and is
counted. Twenty-six focused tests passed with six Drake-dependent tests
skipped on this host; scoped mypy, Ruff, Black, LaTeX compilation and normal
headless push hooks passed. In a private v2 probe using the direct Clarabel
adapter, strict mode stopped at controller call 18 because the adapter
classified `AlmostSolved` as unsuccessful. The prior 1322-call diagnostic
therefore must not be described as 1322 successful QP solves. The numeric
slack tolerance used there was test-only, not a physiological gate. This is
not a complete baseline or PR; a separate broad-import mypy was still running
when the checkpoint was recorded.
The separate follow-imports mypy for this checkpoint subsequently completed
successfully for the one source file. The private QP adapter was modified for
logging after the initial full run, so its current hash is not that run's
original script hash; its solve-success count remains unknown.

A full 477-frame 3D support audit used native COM Jacobian velocity,
finite-difference acceleration, and the six modeled normal/friction forces,
excluding two endpoints. The kinematic target proxy had vertical RMS force
residual 1.067 bodyweights, horizontal vector RMS 0.354 and combined 1.124;
modeled support was zero for 93.66% of sampled interior states. Applying the
same audit to its 1.322 s forward rollout gave vertical 0.000897, horizontal
0.00714 and combined 0.00719 bodyweights RMS. These are assumed-ground model
and finite-difference diagnostics, not measured forces or proof of physical
impossibility.

A coupled root-translation-only 3D support correction with numerical ±0.05 m
bounds and smoothing stopped after 15 evaluations without convergence; its
sampled force-balance RMS decreased 1.0097 to 0.00557 bodyweights, but
optimality was 3.76. Its projected ideal-drive 1.322 s forward test completed
42,304 stages yet source-target marker RMS worsened 269.65 to 293.27 mm and
observed-marker RMS 270.35 to 293.90 mm. Root translation RMS against the
original proxy rose 0.10587 to 0.11720 m; joint RMS was 0.01764 rad, modeled
ROM excesses 3.87° / 8.98°, peak ideal drive 336.92 N m, minimum compliant
foot sphere −22.72 mm. No impact, measured actuator force or physical
admission was established; sampled COM balance alone did not fix the mismatch.

A stage-constrained 41-coordinate Option B refinement under 10 N m trust and
assumption-marked motor limits reduced a 20 ms RK4-stage modeled ROM overshoot
from about 1.047e-5 to 2.16e-9 rad; a further numerical margin gave zero
modeled ROM/speed flags at 640 recorded stages, with 608 stages clipped.
The same candidate had 11 ROM flags and 1.2492 rad peak excess by 200 ms,
and hit a guard at 1.050 s with divergent states. The later guard is not
feasibility or improved carry. RK4 evaluation stages are not a continuous
trajectory certificate, and 16/41 capacities remain unverified.

Research branch `codex/15493-motor-bundle19` at `5b3b0fdf21` combines the
existing typed-stage and motor-stage research components. A force-only
verifier reconstructs a fresh MuJoCo model and motor XML from externally
hash-bound bytes, decodes explicitly labeled post-saturation numerical stages
and re-evaluates saved stage forces. A synthetic v2 16-stage test rejects
altered force even when the archive digest is recomputed; ideal v5/v6
envelopes reject motor traces. Sixty-three focused tests, scoped mypy, Ruff,
Black, LaTeX compilation and normal headless push hooks passed. This verifies
forces at stored numerical states only: it does not bind to the merged bundle
schema, freshly replay the controller, attest an independent runtime or prove
physical motor parameters. The combined branch is oversized research, not a
complete Slice 2 or PR.

A 41-coordinate stage-aware Option B continuation attained a 50 ms prefix
with zero sampled modeled ROM/speed flags over 1600 RK4 stages, though 608
stages clipped under partly unverified capacities. At 200 ms it still had
11 ROM flags and 1.249 rad peak excess; extension failed at 0.729 s. A 100 ms
continuation from that candidate moved the first sampled violation back to
about 29 ms, had 11 ROM flags / 0.7982 rad excess at 200 ms, and failed at
0.458 s. These are numerical stage samples, not continuous-path or
physiological feasibility, a full-horizon swing or power improvement.

A prefix-preserving nonlinear Option B search retained zero modeled ROM/speed
flags in its first 1600 recorded RK4 stages and moved the first sampled flag
from stage 1833 to 1969 (about 61 ms), using a 9.31983 N m coefficient norm
under partly unverified model limits. At 200 ms it still had 11 ROM flags,
1.1050 rad peak excess and 1376 clipped stages; its full extension failed at
0.346 s. Common-prefix q/v/command/applied arrays matched in the same runtime.
This is not a continuous, physiological or full-swing feasibility result.

A coupled 199-frame 3D-root and bilateral knee/ankle support sensitivity
stopped after 12 evaluations without convergence; support residual RMS
changed 1.258 to 0.693 bodyweights, target-marker RMS 16.120 to 17.311 mm and
observed RMS 32.621 to 33.325 mm. Its 0.55 s projected, no-impact ideal-drive
replay completed 17,600 stages. Over the same 6559 valid marker samples,
source-target marker RMS changed 72.94 to 66.53 mm, but peak ideal drive
rose 395.04 to 1187.20 N m, joint q/v tracking worsened and three modeled
ROM flags remained. The numerical bounds/weights and compliant support are
assumptions, not physical gates; this is not an admitted improvement.

A fresh same-runtime v2 fixed-gain motor replay reproduced 6400 recorded
stages over 200 ms, including post-saturation generalized motor forces, using
partly unverified model-prior caps. A separate full 1.322 s recording and
fresh replay reproduced all 42,304 numerical stages exactly, including
post-saturation force, under loose synthetic ±1000 N m caps, no impact or
projection and an exploratory dynamics-derived reference. The two passes took
78.02 s. These are same-runtime numerical replay results, not independent
runtime, physical actuator qualification, admitted match or improved shot.

Research branch `codex/15493-motor-forward20` at `cd6752e605` implements a
fresh fixed-PD controller and MuJoCo motor re-simulation against separately
hash-bound controller, model, motor XML, archive and partial measured runtime.
It compares numerical stage q/v/command and actual post-saturation generalized
motor force, with explicit KKT regularization. Functional failing-first API
check, 54 focused actuator/archive/bundle tests, scoped mypy, Ruff, Black,
LaTeX compilation and clean normal headless push hooks passed. Initial push
caught two real type/protocol issues; after correction both empirical runs were
repeated under the changed fingerprint before the successful clean push.
The 200 ms exploratory v2 run matched all 6400 stages exactly in the same
runtime, with 1440 clipped stages under partly unverified model priors; a
separate 1.322 s run matched all 42,304 stages with no clipping under synthetic
±1000 N m caps. The contract rejects impact, projection and passive shafts;
it is not bound to the merged Solution Bundle, a complete runtime lock,
independent-runtime replay, physiological actuator evidence, admitted match
or improved shot. The branch is oversized research, not a PR.

A predeclared Option B 100 ms relinearization retained the first 1600 clean
recorded RK4 stages and moved the first sampled modeled ROM/speed flag from
stage 1969 to 2067 (about 64.6 ms), with 100 ms modeled ROM peak changing
0.27452 to 0.25141 rad and coefficient norm 9.3316 N m. Its full extension
reproduced the first 100 ms arrays in the same runtime but hit a guard at
0.523 s with 23 ROM / ten speed flags and 8.21 rad peak excess. Capacities are
partly unverified; this is not continuous or physiological feasibility, an
admitted swing or a power improvement.

An initial root translational velocity sensitivity used six central-difference
probes and a numerical 0.1 m/s L2 bound, unchanged compliant ground/model and
no direct root torque. The selected unmeasured initial-velocity perturbation
changed sampled root translation component RMS from 23.59 to 12.96 mm over
the predeclared 50–550 ms window. Its full 1.322 s projected, unsaturated
computed-torque replay with no impact changed source-target marker RMS 234.60
to 186.74 mm and observed-marker RMS 235.38 to 187.77 mm (15,733 valid
samples); root translation component RMS changed 0.08516 to 0.06761 m and
rotation RMS 0.14295 to 0.11509 rad. Joint RMS worsened 0.01731 to 0.01939
rad, peak held ideal torque changed 359.62 to 340.39 N m, one modeled ROM
flag increased 4.080° to 4.675°, and minimum compliant sphere height changed
−24.87 to −25.50 mm. Initial closure projection was reapplied. The initial
velocity is not measured or biomechanically sourced; this is sensitivity on a
nonconverged proxy, not an admitted baseline, physical claim or Option B
improvement. Common first-550-ms arrays matched previous probes in the same
runtime.

Research branch `codex/15493-motor-bundle21` at `1ba9012db7` integrates an
explicit research schema-v2 no-impact representation and v1 migration with the
actual JSON+NPZ codec and motor-stage replay. It preserves separate reference
and rollout grids, binds externally trusted manifest/model/runtime hashes and
typed motor-stage/controller/XML payloads, and checks boundary states,
commands, applied inputs and controller references against the stage archive
before fresh replay. Adversarial tests reject duplicate JSON/NPZ names,
nonfinite extra arrays, unsupported impact status, wrong hashes and mismatched
boundary/target data. One hundred forty-two targeted tests, scoped mypy, Ruff,
Black, LaTeX compilation and clean normal headless push hooks passed. An
initial push caught an Optional-drive typing defect; it was fixed fail-closed,
tests repeated and the empirical run repeated after the final commit.

The actual codec saved and loaded the exploratory v2 no-impact bundle and a
fresh motor replay matched all 42,304 recorded numerical stages exactly in the
same runtime (38.96 s after final commit). Loose synthetic caps, partial runtime
fingerprint, no impact/projection/passive shaft, and an unadmitted fitted
reference prevent physical, first-contact or independent-runtime claims. The
branch is oversized research, copies codec files already on main, and requires
reconciliation with current main; it is not a completed Slice 2 or PR.

A predeclared stage-aware 100 ms Option B relinearization preserved the first
2067 previously clean numerical stages and moved the first modeled ROM/speed
flag to stage 2087 (about 65.2 ms); 100 ms peak modeled ROM excess changed
0.2514 to 0.2468 rad and coefficient norm was 9.3341 N m within the 10 N m
trust. The full extension reproduced the first 100 ms state/command/applied
stage arrays in the same runtime then hit a guard at 0.509 s with 23 ROM /
eight speed flags. At the first flagged stage left subtalar q was 0.349158 rad
versus modeled upper 0.349066 rad and its 5.520 N m command equaled applied
force. These are sampled numerical stages under partly unverified priors, not
continuous, physiological, full-swing or power evidence.

Initial-root-velocity provenance was examined using the hash-bound source
controller target and stored feedback. Source `q_track` initial translation
gradient norm was 0.02741 m/s, agreeing with source initial feedback v0 to
5.97e-7 m/s; v2 proxy initial gradient norm was 0.01786 m/s and raw coordinate
components differed by 0.01012 m/s. Coordinate equivalence across model frames
is not established. Four pelvis-attached marker secants over four frames
produced source-target-derived / observation-derived correction norms of
0.01474 / 0.01819 m/s, with opposite vertical signs; these use fixed proxy
joint rates and are not directly measured root velocity. Full projected,
unsaturated computed-torque replays with no impact or root torque gave target /
observed marker RMS 238.49 / 239.25 mm and 238.02 / 238.78 mm respectively,
versus unperturbed 234.60 / 235.38 mm. Both worsened drift; they do not
support adopting the earlier numerically selected 0.1 m/s correction as a
measured state or establish a v2 baseline.

Research branch `codex/15493-motor-impact22` at `cde946327e` extends the
codec-backed motor replay with optional existing closure projection and a
separately applied prescribed latched force. It records/recomputes saturated
motor input and external force distinctly, verifies the actual split-event
state, and preserves `contact_verified=False`. Synthetic committed-v2 bundle
round trips cover projected/unprojected × no-impact/prescribed-force cases,
including tampered-event and false-contact-certification rejection. One
hundred seventy-seven focused tests, Ruff, Black, LaTeX compilation and normal
headless push hooks passed. Replaying a supplied force does not establish
bracketed physical contact, physiological feasibility, independent runtime
or a complete Slice 2. This is an oversized research branch, not a PR. An
additional broad mypy was still running when this checkpoint was recorded.

A direct raw-coordinate diagnostic matched the source `q_track` initial
translation-gradient components with a 0.01012 m/s perturbation; cross-model
root-frame equivalence remains unverified. The full projected ideal-drive
replay changed target/observed marker RMS from 234.60/235.38 to 233.35/234.13
mm, with one modeled ROM flag, so the small reduction does not qualify the
baseline. Four-pelvis-marker source-target and observation estimates each
improved its own initial marker-velocity comparison but worsened the other;
finite secants and fixed attachments are not measured root velocity.
The additional broad follow-imports mypy subsequently completed successfully
for the four changed source files.

A six-component source-derived initial pelvis-twist diagnostic used four
pelvis-attached markers. At six sampled source/v2 states their translational
root Jacobian columns matched exactly; rotational columns differed in norm by
up to 0.0272–0.0284. At the initial projected v2 state, source-v2 marker
velocity residual had comparable translational and rotational components.
A rank-six least-squares correction had magnitudes 0.01359 m/s translation and
0.06637 rad/s angular. After closure projection it changed initial source-
target pelvis marker velocity component RMS 0.00956 to 0.001315 m/s but
worsened observed secant RMS 0.01355 to 0.02141 m/s. Its full 1.322 s
projected, no-impact, unsaturated computed-torque replay used zero direct
root torque; source-target marker RMS worsened 234.60 to 244.13 mm, observed
235.38 to 244.87 mm, root translation component RMS 0.08516 to 0.08962 m,
root rotation RMS 0.14295 to 0.14745 rad. Joint RMS changed 0.01731 to
0.01643 rad; one modeled ROM excess was 3.67° and peak held ideal effort
359.75 N m. This source-derived model/attachment estimate is not a measured
initial root velocity or a qualified baseline; it did not fix full drift.

A fresh Python process verified an actual serialized 1.322 s exploratory v2
motor bundle using the pushed `codex/15493-motor-impact22` implementation.
Closure projection was enabled and an explicitly synthetic off-grid 2 N
prescribed force was latched at 1.0005 s. All 42,312 numerical stages and
the split-event state reproduced exactly in the same runtime; the separately
recorded generalized external force was nonzero at 36 stages. Recording took
183.09 s and fresh-process verification 167.62 s. The receipt explicitly
sets contact, physical feasibility and independent-runtime verification
false. Loose synthetic motor caps, prescribed rather than detected contact,
a partial runtime fingerprint and an unadmitted reference remain; this does
not complete Slice 2 or establish a physical shot. Private bundle and hash-
bound receipt remain outside git.

Research branch `codex/15493-coverage-continuity23` at `e06fe08a75`
fixes two conditional-contact coverage defects found by failing-first tests:
a different q/v state could be spliced at the same timestamp, and an advance
callback could start from a state different from its supplied q/v. The policy
now stores and serializes the prior evaluated endpoint and fails closed on
state mismatch, malformed or absent historical endpoint, and inconsistent
callback origin. Exact equality intentionally may reject projected or
roundoff-different states. One hundred four focused contact/coverage/MuJoCo
tests, scoped NumPy-aware mypy, Ruff, Black, LaTeX compilation and normal
headless push hooks passed. The unchanged pinned modeled-address ball still
starts 12.704 mm overlapped in exploratory v2 and remains unresolved for
illustrative 10/100/1000 m/s bounds. This does not verify the bound, continuous
trajectory or first contact; the oversized research branch is not a PR.

A post-hoc FK decomposition on 15,733 valid marker samples replaced root
components in saved full v2 rollout states with the fitted proxy's values.
Unregulated source-target marker RMS was 269.65 mm; replacing only root
translation gave 158.98 mm, only rotation 182.90 mm, both 20.52 mm, while
the fitted proxy itself was 19.62 mm. For the existing regulator the same
values were 234.60, 150.29, 147.56 and 20.33 mm. These counterfactuals are
not additive, dynamically achieved trajectories or acceptance evidence;
they isolate root/world motion as a dominant residual.

A matched-grid source-v1 support audit used its own ground/gravity and COM
Jacobian, finite differences at 473 interior 360 Hz samples, no impact.
The hash-bound controller target had vector COM residual 1.013 bodyweights
RMS (vertical 0.970, horizontal 0.291) and modeled normal-support median
0.0807 bodyweights versus required-normal median 1.003. Saved feedback q/v
interpolated to that same grid had vector residual 0.00525 bodyweights RMS.
Source and v2 ground planes differ by about 0.258 mm; these are retrospective,
model-dependent force estimates, not measured GRF, independent replay or
proof of physiological impossibility. They indicate force inconsistency in
the source target under its own model before v2 transfer.

A two-basis compact-support Option B spline diagnostic used a 10.1 ms minimum
break spacing and combined coefficient L2 9.804 N m under the same assumption-
marked caps. A −3 N m subtalar coefficient starting after 50 ms preserved
the first 1600 stage q/v exactly and changed first sampled modeled flag from
stage 2087 to 2097, but retained ROM flags and clipping. Its extension hit a
guard at 0.643 s with 26 ROM / 22 speed flags. The two-basis coefficient norm
is parameterization-dependent and not equivalent to canonical knot insertion;
no physical feasibility, continuous-path or improved-shot claim follows.

A separately installed Python environment reproduced the original serialized
projected 1.322 s exploratory v2 motor bundle with synthetic off-grid
prescribed force: all 42,312 saved numerical stages and the event matched
exactly in 167.71 s. It used the same host and CPython build, matching package
versions and sampled NumPy/SciPy/MuJoCo extension binary hashes; this is
isolated-environment repeatability, not independent runtime/hardware,
physical feasibility or detected contact. The receipt keeps those verification
flags false and the runtime fingerprint remains partial. A separate checkout
at the same Git commit failed the byte fingerprint because four source files
had CRLF instead of the original LF bytes; the guard was not bypassed. This
shows Git SHA alone does not identify working-tree source bytes. Private
hash-bound receipts and bundle remain outside git.

A 477-frame v2 hard-closure fit to FK markers from the source saved-feedback
trajectory (kept distinct from true `q_track`) stopped after 241 s / 46
iterations without convergence. Fit Euclidean marker RMS was 20.96 mm versus
feedback, 32.43 mm versus `q_track`, and 38.58 mm versus observations;
independent max closure was 0.000973 mm / 2.37e-6 rad and supplied bounds
held. A fresh 1.322 s projected, unsaturated ideal computed-torque replay
completed 1322 holds / 42,304 stages with no impact. On 15,733 valid marker
samples forward RMS was 260.82 mm versus feedback, 274.38 mm versus `q_track`
and 275.11 mm versus observations; root translation component RMS was
0.1037 m, joint q RMS 0.01696 rad, peak ideal drive 375.63 N m, one named
model ROM excess 1.685° and minimum compliant sphere height −21.64 mm.
Using the source's dynamically consistent saved feedback as a kinematic target
alone did not resolve v2 world/root drift. This nonconverged proxy is not an
admitted baseline, Option B, actual-actuator or contact evidence.

A coordinated 41-coordinate late-spline diagnostic preserved its first 50 ms
and reduced 100 ms sampled model-ROM peak 0.24675 to 0.20172 rad; three ROM
flags and 1376/3200 clipped stages remained. Its extension reproduced the
first 100 ms numerical stage arrays in the same runtime but hit a guard at
0.504 s with 23 modeled ROM / 12 speed flags. The two-basis coefficient
norm is parameterization-dependent and 16/41 capacities remain unverified;
this is not physical feasibility or a shot improvement.

A model-dependent support audit of the full 477-frame v2 saved-feedback
marker proxy gave combined 3D COM force-balance residual 0.9242 bodyweights
RMS (vertical 0.8614, horizontal 0.3348), versus 0.00729 for its v2 forward
rollout under the same compliant-ground audit. The source-v1 saved feedback
was about 0.00525 on matched sampling in the earlier audit. This suggests the
kinematic transfer loses modeled support consistency, but uses estimated
forces and finite differences, not measured GRF or proof of physical
impossibility. No baseline is admitted.

The original hash-bound 1.322 s projected exploratory v2 motor bundle with
synthetic off-grid prescribed force was replayed on a separate user-owned
Windows host using an isolated portable copy of the trusted interpreter,
packages and exact source bytes. The strict partial runtime fingerprint
matched without bypass; local/remote trusted JSON, manifest and array archive
hashes matched. The verifier completed 42,312 stages in 593.85 s and returned
forward replay verified. Its comparisons use exact array equality for each
stored/replayed stage array and event q/v, and exact event time; mismatch
raises. This is cross-host repeatability with the same OS build, CPython and
copied runtime/native binaries, not an independently built runtime. Contact,
physical feasibility and independent-runtime flags remained false. The force
was prescribed, motor caps synthetic, and the reference unadmitted. A private
hash-bound receipt preserves the result; no private identifiers are in git.

A fixed-basis local Option B convex minimax model over the 0.5 s hold grid
used the existing finite-difference trajectory sensitivity, the 10 N m global
coefficient ball and local step radii 0.25 / 0.5 / 1 N m. Numerical
primal/dual worst-violation values were approximately 7.22275 / 6.93559 /
6.57204 rad-equivalent, with solver residuals around 1e-10. These are only
approximate bounds for that linearized finite-grid surrogate and partly
unverified model priors, not nonlinear or physiological infeasibility proofs.
The 0.25 candidate's actual motor rollout hit a guard at 0.357 s while its
baseline reached 0.5 s; prediction errors grew rapidly and it was rejected.

A separate multiple-shooting seam prerequisite split an actual fixed-gain
0.5 s motor rollout at 0.250 s. Restarting a fresh plant with exact full q/v
seam state and absolute controller clock reproduced the uninterrupted q/v,
RK4 stage states, commands and post-saturation drive bit-for-bit in the same
runtime, for both baseline and a nonzero coefficient perturbation within the
10 N m trust ball. This validates the numerical handoff only; it is not a
lifted optimizer with seam-defect constraints, a feasible trajectory or
independent runtime replay. Private scripts and hash-bound receipts remain
outside git.

Current-main codec coordination finding: open PR #15600 at inspected head
`9de2563904b026eac983227c2112f84eabaadf5b` requires the impact event time to
lie within the recorded interval. The prior restricted no-impact bridge
`codex/15493-main-bundle17` requires its v1 placeholder event after the final
sample, so these contracts would be mutually unsatisfiable if combined
unchanged. Explicit versioned no-event semantics and migration are required;
recheck the current PR head/state before acting. Do not invent an event or
bypass its validation.

After correcting a sparse-Jacobian row-pattern error and checking randomized
whole-residual directional derivatives (relative errors 7.2e-7–1.6e-6), a
33-frame six-DOF root soft centroidal-momentum fit reached its 20-evaluation
limit without convergence. The earlier result using the incorrect pattern is
invalid and must not be used. A 477-frame extension stopped at its 240 s
callback, not converged (reported optimality 278): sampled modeled linear
balance RMS changed 1.1243 to 0.8637 bodyweights and angular RMS 104.01 to
35.21 N m. Root adjustments were numerical bounded diagnostics, not physical
limits. A matched-initial-state 1.322 s projected ideal computed-torque
comparison gave original-root translation component RMS 105.716 / 105.727 mm
and original-root rotation RMS 0.142821 / 0.142826 rad baseline/candidate.
On 15,700 valid marker samples, fitted source-target RMS worsened 19.619 to
20.363 mm and forward source-target RMS 268.434 to 268.463 mm. The candidate's
own-reference metric was misleading because its reference changed. This soft,
nonconverged model diagnostic is not hard collocation, physical admission or
an improved baseline.

Research branch `codex/15493-schema-v2-26` at `f965b29e51` implements an
isolated explicit v2 no-impact representation, separate reference and rollout
grids and event-preserving v1 migration without editing the active codec PR's
files. Ninety-seven focused v1/v2 tests and normal headless push hooks passed.
It is not integrated motor replay, provenance attestation, physical acceptance
or a complete Slice 2; reconcile with current canonical codec and #15600 state
before use. No PR was opened.

A genuine two-window lifted Option B local feasibility diagnostic used an
independent 47-q/47-v seam, 41 coefficients, fixed gains, real motor forces
and the original 10 N m coefficient ball. Two local linearized iterations
reduced 20 ms modeled stage excess 0.01104 to 0.00778 rad-equivalent, but
nonlinear seam defects remained (second iteration 3.01e-5 native q /
0.002699 native v). Replacing the seam with the actual first endpoint made
segmented and uninterrupted first 640 stages exactly equal in this runtime,
but did not remove violations; full extension hit a guard at 0.528 s with
23 modeled ROM / eight speed flags. Only near-bound stage constraints were
linearized; work/slew, continuous behavior and 16 unverified capacities remain.
This is not a feasible swing, complete optimizer or physical conclusion.

A 477-frame coupled root/ankle/knee soft momentum and target-marker fit,
with randomized sparse-Jacobian checks 7.6e-8–1.03e-7 relative, stopped at
its 240 s callback after eight evaluations without convergence (optimality
1209.83). Under identical baseline-derived closure-projected initial q/v,
its 1.322 s projected ideal-drive replay worsened original-reference root
translation RMS 105.72 to 107.89 mm, rotation 0.14282 to 0.15646 rad and
peak ideal drive 383.77 to 761.81 N m; two model ROM flags remained. Across
15,700 valid markers, forward source-target RMS worsened 268.43 to 280.62 mm
and observed RMS 269.13 to 281.27 mm. It is not a qualified baseline or
forward improvement; ground and numerical regularizers remain assumptions.

Research branch `codex/15493-motor-v2-reconcile27` at `ad8a78319b` connects
explicit no-impact v2 replay to the original reference grid, without resampling
controller input. Four synthetic motor cases, including projection and legacy
prescribed force, passed; altered reference values and times are rejected.
Normal headless push hooks passed. This 354-line increment is based on the
oversized motor research branch, not a complete Slice 2, independent-runtime
or physical certification, and not a PR. Reconcile against current main and
current #15600 state before use.

A condensed two-window Option B local restoration initialized the second
window from the actual first endpoint, making the nonlinear seam exact by
construction. It removed sampled modeled ROM/speed flags over its first
20 ms / 640 RK4 stages while checking actual clipped total motor force.
Uninterrupted validation nevertheless first violated modeled ROM at stage 755
(about 23.6 ms) and hit a numerical guard at 0.528 s, with 23 ROM and eight
speed coordinates flagged. Sixteen torque capacities are assumption-marked;
work/slew and continuous physical gates remain unverified. This is not a
feasible full-horizon swing or improved shot.

A full-horizon contact-wrench QP diagnostic used a private Clarabel adapter
with numerical solver tolerances 1e-6 and independently checked equality
residual <=1e-5 and inequality <=1e-6. All 1322 calls completed without
fallback and direct root effort was zero. Target/observed marker RMS were
272.70 / 273.39 mm, root translation component RMS 0.1055 m and rotation
RMS 0.1408 rad; peak absolute generalized actuated effort was 384.09 N m.
Maximum normalized soft contact-cone slack was 0.5337 and its threshold was
deliberately nonbinding to observe solver behavior. This shows fallback alone
did not explain prior poor tracking, but does not establish admissible
contact, anatomy, actuator limits, physical baseline or Option B success.
The adapter is not independently proven equivalent to unavailable Drake.

Research branch `codex/15493-motor-evidence28` at `bcd082ce4d` fixes a
false-positive replay evidence path in the research motor runner: constructing
`MotorForwardCheck` could set `forward_replay_verified=True` with zero stages
before any integration. The default is now false and true is set only after
archive-versus-fresh-stage equality. The regression failed before the fix;
14 focused MuJoCo motor replay tests and normal headless push hooks passed.
This does not establish the private bundle's validity, physical feasibility,
contact or independent runtime, and the branch remains oversized research,
not a complete slice or PR.

A three-window condensed Option B restoration propagated the actual endpoint
between each 10 ms window. Over 30 ms / 960 RK4 stages it had exact q/v
seams and no sampled modeled ROM or speed flags, with coefficient norm
9.43249 within the 10 N m diagnostic radius. There were 672 clipped stages;
signed stage-quadrature work was -6.245 J, positive power integral 0.00463 J,
and the largest 1 ms applied-drive update was 23.2964 N m. Work and slew
limits are not sourced, and 16 of 41 torque capacities remain unverified.
The uninterrupted continuation first flagged ROM at stage 961 and hit a guard
at 0.723 s with 27 ROM and 17 speed coordinates flagged. This is sampled,
short-window numerical evidence, not full-horizon feasibility or a shot.

Production contact and flight paths still need integration: the existing
impact scheduler uses a face-center proximity / linear-extrapolation heuristic
and clamps negative lead to immediate firing; the flight pipeline accepts a
supplied impact state. Neither independently proves an approaching finite-face
root. A separate research surface trigger and an open generic root PR do not
by themselves establish this end-to-end contact gate. Verify the first root,
finite-face geometry, bracket confinement and numerical convergence before
using contact or carry as evidence.

Next priorities: complete the support-preloaded full-trajectory fit and an
independent forward audit; establish sourced hard anatomical, speed, actuator,
work and slew limits; resolve the initial ball/face overlap with an authorized
physical placement; and extract small, sequential Slice 1 changes against the
current main and existing open work. Keep the prescribed-launch flight tests
separate from the unqualified swing-to-impact path. Related Slice A remains
blocked on admission of the same physically qualified two-handed baseline.

A 477-frame support-preloaded fit used the existing six compliant ground
spheres and no direct root drive. Sparse directional Jacobian errors were
5.6e-7 to 6.7e-7 relative. At its 254 s budget it had not converged: linear
balance residual decreased from 1.010 to 0.671 body weights RMS and angular
residual from 109.6 to 60.2 N m RMS relative to the preloaded initial guess,
while proxy marker RMS increased from 19.62 to 20.18 mm. An independent 1.322 s
ideal-drive rollout under the identical baseline-derived initial state finished
without a numerical failure but worsened source-target marker RMS from 268.43
to 284.68 mm, observed-marker RMS from 269.13 to 285.32 mm, root translation
RMS from 0.1057 to 0.1101 m and root rotation RMS from 0.1428 to 0.1557 rad.
Peak unsaturated ideal drive rose from 383.8 to 556.6 N m; both rollouts had
two modeled ROM flags. No measured ground reaction, motor saturation, impact,
or verified speed and torque gates were included. Reject this fit as a
baseline improvement; improved fitted balance alone did not predict improved
forward tracking.

Research branch `codex/15493-slice-audit29` at `0d1437e5fc` corrects two
legacy passage-trigger timing cases: receding closest approach and an event
before arming. Both regressions failed first; 20 focused tests passed, as did
normal commit and push hooks including mypy, Bandit and the unit-test hook.
This is a small pushed checkpoint, not a PR or the required finite-face
contact gate. It overlaps the impact-force files of the larger real-face
research branch and must be reconciled before integration.

A four-window exact-seam Option B continuation removed sampled modeled ROM
and speed flags over 40 ms / 1,280 RK4 stages, with coefficient norm 9.43307
inside the 10 N m diagnostic ball and 672 clipped stages. Signed applied-stage
work was -5.947 J, positive-power integral 0.303 J, and peak 1 ms drive
update 23.2964 N m. Uninterrupted replay first flagged ROM at stage 1843
(approximately 57.6 ms), then hit a numerical guard at 1.270 s with 38 ROM
and 41 speed coordinates flagged. The run's post-violation outputs are not
physical metrics. Sixteen torque capacities and the work/slew bounds remain
unverified; sampled short-window success does not admit this swing.

A focused reconciliation of `0d1437e5fc` with real-face research commit
`36a15de7c2` passed 84 impact-force, contact-event, MuJoCo surface and
synthetic v2 finite-face tests (three warnings). Keep the research version of
`_point_and_normal` and both sets of tests when resolving the two textual
conflicts; the legacy timing guard applies cleanly and surface dispatch bypasses
it. This isolated test changes neither branch and does not resolve first-root
coverage, initial overlap or physical admission.

A provenance review found that the v2 spec defines coordinate and soft ROM,
but no torque, rate or work capacities. The existing registry and physiological
envelope leave 16 of 41 non-root torque slots assumption-marked: bilateral
shoulder X/Y/Z, scapula X/Y, forearm rotation, wrist radial/ulnar and MTP.
Some individual directions have static literature entries, but their opposite
directions remain unverified. Of 41 speed slots, 33 are unverified and eight
are sourced with assumed margin; none is an established subject-specific hard
limit. Existing partial lower-limb static and force-velocity sources are
population- and posture-dependent. Work and slew limits remain unsourced.
Use these as research priors, not certified physical gates.

A 60 ms exact-seam Option B continuation reduced the original candidate's
0.004591 rad modeled ROM excess to zero sampled ROM/speed flags across 1,920
stages. Coefficient norm was 9.437298 within the 10 N m diagnostic ball, with
960 clipped stages, signed work -4.4830 J, positive-power integral 1.76935 J,
and peak 1 ms applied-drive update 23.2962 N m. Independent uninterrupted
replay first flagged ROM at stage 1921 (about 60.03 ms) and stopped at a
numerical guard at 0.665 s with 27 ROM and 15 speed coordinates flagged.
Compared with the 40 ms candidate, the first flag moved later but the guard
occurred sooner. Neither demonstrates full-horizon or physical feasibility.

Windowed support diagnosis found zero modeled contact in the original
reference over 0–1.1 s, and only 0.32–0.43 body weights RMS modeled contact
force in the preloaded fit over that interval. A 21-frame post-filter reduced
correction roughness but worsened force and moment residuals; it is not an
admissible fix. A coupled acceleration-regularized fit stopped unconverged
after 265 s. Its sampled linear and angular residuals fell from 0.671 to
0.527 body weights RMS and 60.2 to 43.4 N m RMS relative to its seeded start,
but its weak numerical regularizer barely changed correction roughness.
Independent 1.322 s ideal-drive rollout source-target marker RMS worsened
from 268.43 mm baseline and 284.68 mm seeded to 291.77 mm regularized;
observed-marker RMS was 269.13 / 285.32 / 292.39 mm. All retained two modeled
ROM flags. Reject both fitted candidates: lower sampled balance residuals
have not produced forward tracking improvement. No measured ground forces,
motor saturation, impact or verified speed and torque gates were included.

A foot-geometry diagnosis compared the actual embedded matched model with the
pinned v1/v2 models. Sphere definitions, contact parameters and ground normals
match; the embedded ground height is 0.258 mm above the v2 default plane.
The embedded source has different lower-chain transforms from both pinned
models: hip translations differ by 61/103 mm and rotations by 11.1/28.8 deg,
with 10 deg ankle rotation differences. Direct joint-name copying is invalid;
source versus v2 sphere-height discrepancy under that copy is 69.3 mm RMS.
The migration helper correctly rejects the differing embedded source spec.

Over 476 samples, the source target's median deepest sphere gap is -0.415 mm
and modeled normal force is about 0.081 body weights median; the fitted v2
target's median deepest gap is +0.502 mm and 93.7% of samples have zero
modeled normal force. These statistics do not establish that the small ground
offset caused the mismatch. The current fit tracks source-model markers and
anchors only horizontal foot projections. Against time-varying source-model
sphere centers, horizontal RMS is 0.77 mm but vertical RMS is 14.23 mm
(left heel 28.31 mm). These are model-derived targets, not measured foot
positions or measured ground forces.

Three independent representative frame fits retained v2 geometry, modeled
hand/club closure and available named anatomical bounds while fitting 3D source
sphere centers and markers. Sphere RMS decreased from 14.08/14.94/15.67 mm
to 0.57/0.63/0.54 mm; marker RMS increased from 25.14/17.38/19.37 mm to
27.21/20.92/22.49 mm and closure residual stayed below 1.7e-11. Solver success
for isolated poses does not establish temporal continuity, physical limits or
forward tracking. Test a continuous fit and independent rollout before use.

Current-main prescribed-launch fixtures call the flight simulator directly
using ten published aggregate launch rows, separately from swing-to-impact.
An isolated Python 3.12 build of the current native kernel reported physics
revision 3 and yielded 30 passed / two expected failures. Running the two
expected failures as ordinary assertions confirmed actual carry-band misses:
127.481 yd versus a 129.2 yd lower bound and 99.266 yd versus 101.65 yd.
Published aggregates are comparisons, not individually matched controlled
shots. The default Python environment contained an importable stale revision
and produced setup failures; the broad expected-failure marks also masked two
of those errors. Future fixture work must distinguish an unavailable or stale
kernel from a numerical model discrepancy. Neither result qualifies the
unverified swing-to-impact path.

A 200 ms late-violation sensitivity and early-restoration experiment reduced
maximum modeled ROM excess at 200 ms from 1.15669 to 1.03005 rad while keeping
zero sampled flags in the first 60 ms and staying in the diagnostic trust
ball. The independent full rollout first flagged ROM around 62.8 ms and hit a
guard at 0.552 s, earlier than the previous 0.665 s. Reject this as evidence
of full-horizon improvement. Small coefficient changes produced large later
joint differences even without late-stage clipping; causality remains unproven.

Research branch `codex/15493-flight-evidence31` at `592b9ffd7b` makes
prescribed-flight fixture failures distinguishable from unavailable or stale
native physics: expected physical misses accept assertion failures only, and
an unusable kernel fails explicitly. Failing-first regression, isolated
revision-3 run (32 passed, two expected numerical failures), stale-revision
negative control, scoped mypy, Ruff, Black and normal push hooks passed. It is
not a PR. Reconciliation with the earlier Slice 1 extraction must retain the
aggregate-data caveat and both sets of tests, and mock kernel availability in
the extraction's mocked unlanded-trajectory test.

A 476-frame sparse 3D-foot/marker/closure fit on the pinned v2 model stopped
unconverged after 244 s. Source-model sphere error fell from 8.23 to 0.440 mm
per Cartesian component while target-marker component RMS rose from 11.33 to
15.81 mm; sampled closure residual was 1.8e-9 and no named-angle excess was
reported. It used soft temporal regularization and model-derived foot targets,
not measured forces. Every sampled frame had modeled contact but median normal
force was only 0.0286 body weights. Peak target joint rate was 34.5 rad/s;
no verified speed gate passed.

An ideal unsaturated projected-drive rollout of that target completed 1.319 s
without numerical failure. Against exactly the same source-FK markers, the
same 475 comparison times and array-equal initial q/v as the baseline rollout,
source-marker RMS improved 267.24 to 37.32 mm and observed-marker RMS 267.94
to 43.07 mm. Source-marker RMS in the last 1.1–1.317 s improved 514.4 to
59.3 mm; the first 0.1 s changed 28.9 to 30.0 mm. Against the original
reference, root translation vector RMS improved 0.1817 to 0.0201 m and root
rotation vector RMS 0.2436 to 0.1116 rad. Peak held ideal unsaturated effort
rose from 383.77 to 524.20 N m; peak sampled joint speed was 35.03 / 34.91
rad/s. RK4 stages were not retained. The fitted target's first pose differs
from the shared rollout start by up to 0.335 rad in a shoulder coordinate;
its initial state was not constrained in the fit. This is an encouraging
tracking lead, not an admitted baseline: the fit is unconverged, and actual
saturated total torque, stage-level ROM/speed, work, impact and measured
ground forces remain unverified. Independently check actual motor-drive
replay and consistent initialization before any physical or carry claim.

The 3D-foot candidate was also tested with actual modeled motor saturation,
fixed gains 3/0.3 and target-state computed feedforward sampled at 1 ms, from
identical model hash and recorded initial q/v. The first 200 ms / 6,400 RK4
stages had no sampled modeled ROM/speed flags or clipping, with root
translation/rotation and joint RMS versus its own target 0.0307 m / 0.0837
rad / 0.1397 rad. The full requested 1.319 s run first flagged modeled ROM
at stage 9399 (about 0.294 s), then stopped at a numerical guard at 0.885 s,
with 32 ROM and 25 speed slots flagged. Peak total command was 272.94 N m,
peak actual modeled drive 72.4 N m and 2,560 of 28,320 stages clipped. The
ideal comparison used projected computed torque, while this run used an
unprojected fixed-gain controller, so this does not isolate saturation as the
cause. Sixteen capacity priors and all speed/work/slew admissions remain
unverified; this is not an admitted baseline or feasible swing.

A controlled fixed-gain Option B ablation on the foot-target candidate used
the same v2 plant, reference, initial state and command with projection on/off
and manual preclip on/off. Through 350 ms / 11,200 RK stages, all four direct
input cases were identical to roundoff (maximum q difference 3.36e-13,
velocity 1.51e-12): six modeled ROM slots failed, first at stage 9399, and no
speed slots or commands clipped. At 750 ms, unprojected clipping first
occurred near 0.717 s, after the first ROM failure; the two unprojected cases
had 20 ROM and 11 versus 12 speed slots flagged. Preclipped direct input
matched the actual motor run bit-for-bit in states, stage states and applied
drive through 750 ms. Projected preclip is a direct-input surrogate: the
motor plant does not support projection. These comparisons narrow the first
failure but do not establish its cause or physical feasibility.

The ideal foot-target replay and low-gain Option B use different controllers.
The ideal recomputes constrained inverse dynamics at the actual state with
acceleration gains 900/60 and COM balance gains 60/15; Option B freezes
reference-state feedforward and uses direct torque gains 3/0.3. At the shared
start, their peak commands were 65.65 and 3.19 N m respectively. With the
identical actual-state computed-torque controller, the unprojected direct
plant reproduced the saved projected ideal through 350 ms to maximum q/v
errors 1.22e-11 / 1.05e-10 and effort error 1.89e-8, with no modeled ROM or
speed flags. Projection therefore did not explain the ideal tracking in that
interval. On the same controller, motor saturation first clipped at about
62 ms and clipped 8,352 of 11,200 stages; eight modeled ROM slots failed.
Joint RMS versus target was 0.1966 rad, compared with 0.0436 rad for ideal;
root translation RMS was 0.0174 versus 0.00563 m. Peak command / actual
modeled drive were 563.8 / 200.3 N m. Saturation changes this controlled
comparison under assumption-marked capacities, but does not establish a
physiological solution. Actual-state computed torque is a diagnostic, not
the required fixed-gain Option B architecture.

An initial-constrained temporal fit fixed q0 and a three-point initial
velocity and stopped unconverged at 242 s. Although that derivative matched
v0 to 1.3e-13 and closure was 5.9e-10, the controller's actual first-order
initial velocity differed by 6.93 rad/s and peak sampled acceleration was
4,992 rad/s². Reject it as an admitted reference. Closure-projecting a
one-step Euler pose instead left only 0.000337 mixed-coordinate units/s
maximum initial velocity discrepancy; it is approximate, and an explicit
manifold-consistent velocity state is the principled follow-up.

The initial-pose-constrained fit's independent ideal unsaturated rollout
completed 1.319 s without numerical failure, with source/observed marker RMS
36.83 / 42.58 mm against the same 475 samples (original baseline 267.24 /
267.94 mm) and no reported sampled modeled ROM flags. It nevertheless needed
901.3 N m peak ideal drive and retained the 6.93 rad/s controller initial-rate
mismatch and 4,992 rad/s² peak sampled target acceleration. No motor
saturation, stage gates, verified speed/torque or impact were checked. Reject
it for admission despite the marker improvement. A bounded follow-up pins the
closure-projected first step to align with the existing controller's finite
difference and tests stronger numerical acceleration regularization.

On the 3D-foot reference, a bounded four-coefficient fixed-gain Option B
restoration removed sampled modeled ROM/speed flags over 320 ms / 10,240
actual motor RK4 stages; the unmodified run had three ROM slots with maximum
0.139206 rad excess. Coefficient norm was 1.23574 within the 10 N m ball,
and no stage clipped in the optimized window. Joint RMS versus target
worsened 0.2435 to 0.2471 rad. An independent uninterrupted replay first
flagged ROM at about 0.321 s versus 0.294 s baseline, but hit a guard earlier
at 0.797 versus 0.885 s, with 21 ROM and 17 speed slots flagged. This is
prefix restoration, not full-horizon improvement or physical feasibility;
work/slew limits and several capacity and speed priors remain unverified.

The projected-Euler q1, 10-times numerical acceleration-regularization fit
stopped unconverged after 250 s with exact q0 and 0.000337 mixed-coordinate
units/s maximum controller initial-rate mismatch. Its sampled hand/club
closure was unacceptable (15.94 mm / 0.0368 rad, constraint residual 0.0329)
despite no named-angle bound excess. Do not forward-test or admit this
iterate; restart only to investigate whether closure can recover. The
regularizer is numerical, not a physiological acceleration limit.

A four-window, exact-seam 400 ms Option B diagnostic kept the first 320 ms
sampled modeled gates clear and reduced maximum 400 ms ROM excess from
0.54359 to 0.43246 rad; seven ROM slots still failed, no speed slots failed,
and coefficient norm was 2.741 within the 10 N m diagnostic ball. The
independent motor replay first flagged ROM at about 0.324 s and reached a
numerical guard at 1.204 s, with 34 ROM and 30 speed slots flagged. Longer
numerical survival is not feasibility or qualified tracking. The larger
linearized step was rejected because actual early sampled ROM failed.

A restart of the projected-q1 reference fit stopped unconverged after another
255 s. Sampled hand/club closure improved from 15.94 mm / 0.0368 rad to
12.39 mm / 0.0315 rad, still unacceptable; controller initial-rate residual
remained 0.000337 mixed-coordinate units/s, and peak sampled target speed and
acceleration were 23.86 rad/s and 1,134 rad/s². No forward run was warranted.
Do not admit either iterate as a closure-consistent reference.

A framewise closure-projection pilot on the smoother initial-constrained fit
succeeded on all 476 frames: sampled closure inf 8.87e-14, no violations in
its available named-ROM subset, maximum coordinate correction 0.0121, exact
q0 and 0.000337 mixed-coordinate units/s controller initial-rate discrepancy.
Source-foot RMS stayed 2.229 mm, source-marker RMS changed 29.966 to 30.052
mm, and peak sampled joint acceleration stayed about 1,134 rad/s². This is
post-processing, not a closure-constrained optimization or physical proof.

Its independent 1.319 s ideal-drive rollout used the same initial state and
475 marker samples: source-marker RMS improved from 267.24 to 45.73 mm and
observed-marker RMS from 267.94 to 50.83 mm versus the original baseline;
peak held unsaturated drive was 464.49 N m. A broader registry audit revealed
lumbar axial rotation outside its modeled envelope in 590 sampled states,
peak 19.94 degrees, although the older HUMAN_RANGES_DEG subset reported zero.
The ideal command exceeded registry torque capacities at 320 samples, peak
ratio 13.56. No sampled speed flag appeared in the assumption-limited registry
(maximum ratio 0.726). All speed capacities and 16 torque slots remain
assumption-marked; no stage-level or actual saturated torque, work or impact
was verified. Do not infer a complete ROM pass from the narrower reporter or
admit this reference as physically feasible.

An eight-direction 400 ms motor-spline continuation reduced worst modeled
ROM excess from 0.43246 to 0.31034 rad but still failed six ROM slots while
preserving the first 320 ms sampled gates. Clustered active-row sensitivities
were poorly conditioned; this is not proof of global uncontrollability. The
independent replay first flagged ROM around 0.321 s and stopped at 0.807 s,
earlier than its predecessor. It is not a full-horizon improvement.

The closure-projected reference was also replayed with actual modeled motors,
zero spline and fixed gains 3/0.3, from the same saved q0/v0. The full registry
covers 41 non-root slots versus 29 in HUMAN_RANGES_DEG. Over the first 200 ms,
both tables flagged right-wrist range starting at stage 151 (about 4.7 ms),
with 1,146 violating stages, no modeled speed flags and 800 clipped stages;
the previous foot-target candidate had no sampled flags or clipping in that
window. At equal 27,232-stage prefix the registry flagged 29 projected-run
slots versus 23 in the narrower table. The motor run stopped at 0.851 s.
Over 200 ms, own-target translation / rotation / joint RMS changed from
0.03068 m / 0.08367 rad / 0.13974 rad for the earlier candidate to 0.02945 m /
0.11694 rad / 0.12259 rad for the projected one. Peak command / applied
modeled drive were 260.09 / 189.97 N m and clipping began at the first stage.
Target and feedforward both changed, so projection alone is not isolated.
Registry limits are model assumptions, not verified anatomical capacities;
this actual motor result does not admit either reference.

A registry-bounded closure pilot solved all 476 frames independently with q0
and q1 retained: sampled closure residual <=5.88e-11 and zero excess against
modeled registry ranges; controller initial-rate discrepancy remained 0.000337
mixed-coordinate units/s. Source-foot RMS improved 2.229 to 0.943 mm and
source-marker RMS 30.05 to 24.27 mm. It is not a usable trajectory: joint
acceleration RMS rose from 47.9 to 430.6 rad/s², peak 20,851 rad/s², and
three actual-interval speed ratios exceeded the assumption-limited registry
(maximum 1.243). Centered differences missed those interval violations.
No forward test was warranted. Time coupling and interval-aware auditing are
needed; registry soft and assumed values are not verified anatomical limits.

Research branch `codex/15493-rom-coverage32` at `273e8dc414` makes legacy
range-table coverage explicit in IK and dynamics receipts and summaries:
checked/unchecked coordinates and covered/incomplete/invalid status are
reported, while old receipts show unknown. Forty-six focused tests, Ruff,
Black and normal push hooks passed. Covered means table coverage only, not a
complete anatomical gate; no PR was opened.

A full 476-frame smoothed, registry-bounded closure reference retained exact
q0 and 0.000337 mixed-coordinate units/s initial-rate residual; sampled
closure inf was 7.31e-12, no modeled ROM excess, interval/centered speed
ratios at most 0.9809/0.9715, acceleration RMS 55.98 and peak 1,010.72 rad/s².
Source-model foot/marker RMS were 2.51/24.09 mm. One frame reached its solver
iteration limit despite independently satisfying sampled geometric gates;
optimality was not established. All registry limits remain soft or assumed.

The independent 1.319 s ideal-drive rollout from the same initial state
completed without numerical failure and achieved source/observed marker RMS
35.72/41.20 mm versus 267.24/267.94 mm baseline, peak unsaturated held
command 314.95 N m. Forward state registry ROM failed in 816 samples (worst
right subtalar excess 11.14 degrees); no sampled speed flag appeared under
assumed limits, while ideal commands exceeded modeled capacity in 280 samples
(peak ratio 7.19). Passing reference-level gates does not establish rollout
feasibility. Stage-level motor torque, work and contact remain unverified;
this is not an admitted physical baseline or optimized shot.

Actual fixed-gain Option B motor replay of the continuous reference used
identical saved q0/v0 and v2 spec, zero spline and gains 3/0.3. Its first
50 ms / 1,600 stages had no modeled registry ROM/speed flags or clipping,
with joint RMS 0.00484 rad and maximum actual drive 35.07 N m. Right subtalar
ROM first failed at stage 2007 (about 62.7 ms); through 200 ms, three ROM
slots failed at 4,393 stages, with no speed flags or clipping and joint RMS
0.11739 rad. The earlier foot-temporal reference had zero sampled ROM/speed
flags and no clipping through 200 ms, with joint RMS 0.13974 rad. The new
run stopped at a guard at 0.552 s, with 17 ROM and four speed slots flagged;
no stages clipped, and stage drive matched total-command clipping exactly.
References and feedforward differ, so this is not a single-variable comparison.
All speed and 16 torque capacities remain assumption-marked; work/slew limits
and physical feasibility remain unverified.

An interior-margin numerical pilot used 3 degrees for both modeled subtalar
ranges, 0.5 degrees for wrist ranges and 0.25 degrees for lumbar ranges,
ramped in over 18 frames to retain q0/q1. These are robustness parameters,
not physiological limits. The full 476-frame reference had all pose solvers
report success, closure <=7.30e-12, no modeled ROM/margin excess and no
interval or centered speed flags (maximum assumed ratio 0.968). Foot/marker
RMS changed from 2.51/24.09 mm to 3.45/24.22 mm versus the unmodified
continuous reference.

The identical-initial-state 1.319 s ideal rollout completed without a
numerical failure. Compared with no margin, sampled forward ROM-violating
states fell from 816 to 338, but right subtalar remained out of range from
0.085 s and peak excess worsened 11.14 to 12.48 degrees. Source/observed
marker RMS worsened 35.72/41.20 to 40.48/45.38 mm; held ideal torque
capacity exceedances increased 280 to 313 samples, peak ratio 7.19 to 8.60,
and peak absolute command 314.95 to 372.30 N m. No sampled assumed speed
flag appeared. Reject this tradeoff as an admitted or improved baseline;
actual saturated torque, stages, work and impact were not established.

A bounded continuous-reference Option B spline correction cleared modeled
registry ROM/speed flags over 100 ms / 3,200 actual motor stages with no
clipping and norm 2.93171 inside the 10 N m diagnostic ball. Independent
replay first flagged ROM at about 100.3 ms versus 62.7 ms uncorrected, then
stopped at 0.869 s with 30 ROM and 25 speed slots flagged; native overflow and
invalid-matmul warnings occurred before the guarded stop. Tracking over the
first 100 ms worsened slightly. This is not full-horizon or physical proof.

A separate 120 ms ideal-controller contact sensitivity found a material
integration-resolution issue. At the same 1 ms hold, the nominal eight-RK4-
substep run first crossed the modeled right-subtalar bound at 79 ms; with 16
substeps it did not cross and peaked at 19.319535 degrees. At 32 substeps
it peaked at 19.319552 degrees, with maximum 16-to-32 q/v differences
4.62e-7 / 7.53e-5 and held effort difference 0.00118 in native units. The
eight-substep trajectory differed by up to 0.0376 in q. Thus the early event
timing and other earlier eight-substep comparisons are not established as
numerically converged; recheck conclusions at a suitable step size. This
local 120 ms comparison does not prove convergence over the full trajectory.

For the corrected continuous-reference Option B motor trajectory, a local
16-versus-32 RK4-substep comparison at the same 1 ms hold agreed over 120 ms:
maximum matched-hold q/v differences were 1.01e-7 / 8.86e-5 in native units,
and command/applied-drive difference 2.67e-5 N m. Both had zero modeled
registry ROM/speed flags for 100 ms, no clipping, and the same first modeled
ROM flag at 100.3125 ms. This is local numerical agreement, not continuous,
full-swing or physiological validation.

Hypothetical contact-parameter perturbations did not all converge: the -10%
damping case changed from a modeled breach at 81 ms with 16 substeps to no
breach over 120 ms with 32; the -10% stiffness case was also unstable across
refinements. Do not infer measured force, parameter robustness or contact
dominance from those variants. The nominal 16/32 ideal-controller agreement
is only established over 120 ms; the pinned baseline settings were unchanged.

A main-based partial integration checkpoint `codex/15493-slice1-integrate33`
at `ee07ab7fc6` combines fail-closed prescribed-flight fixture evidence,
landing verification and a concise v2 forward-tracking diagnostic, 347 changed
lines across five files. Failing-first regressions, 33 flight tests plus two
expected numeric failures with revision-3 physics, four baseline tests
including a closure-consistent synthetic 2 ms MuJoCo run, stale-kernel
negative control, scoped mypy, Ruff, Black, LaTeX and normal push hooks passed.
The diagnostic reports missing provenance and physical gates as unverified,
writes an unqualified result and exits 2. No eligible original-target bundle
bound to canonical v2 was found locally; the embedded calibrated source model
differs and must not be silently accepted. This branch is not a PR or complete
Slice 1: actual best-match forward evidence, full applied-stage physical gates,
first finite-face contact and contact-to-flight qualification remain open.

The continuous-reference ideal-controller convergence comparison was extended
through 0.5 s, holding the 1 ms command lattice fixed. Sixteen versus 32 RK4
substeps agreed across five saved windows to maximum q/v/held-command
differences 4.62e-7 / 7.54e-5 / 0.00118 in native units, with no numerical
failure. The earlier eight-substep trajectory differed from the 32-substep
trajectory by up to 0.0502 in q and 1.40 in v over this interval. At 32
substeps, the only sampled modeled ROM failure through 0.5 s was RWInputY,
first at 0.201 s, peak 0.00711 degrees; no sampled speed flag appeared.
Assumed ideal-command capacity was exceeded from 0.068 s. This invalidates
using the earlier eight-substep early ankle timing as a converged result and
requires reevaluation of its full-run gate counts. It does not establish
full-horizon, actual-motor or physiological feasibility.

The continuous-reference ideal-controller convergence comparison was extended
through the full 1.319 s on the same 1 ms hold lattice. Across 1,320 common
sampled states, 16 versus 32 RK4 substeps differed by at most 2.14e-6 in q,
8.42e-5 in v and 0.00481 N m in held command, with no numerical failure.
The old eight-substep result differed from the 32-substep run by up to 0.183
in q and 1.93 in v, so its full-run state and gate counts are not reliable.
Both refined runs had 712 sampled states with modeled ROM excess (peak right
subtalar 1.106, left subtalar 3.561, soft lumbar 0.082 and wrist 0.447
degrees), no sampled speed flags, and 236 held commands exceeding assumed ideal
command capacities (peak ratio 7.249, peak absolute command about 425.6 N m).
This establishes a close sampled numerical comparison for this ideal controller
and model, not actual saturated-motor, integration-stage, physical or contact
feasibility. Registry limits retain their soft/assumed provenance.

A corrected per-hold decomposition of the refined ideal controller rebuilt
commands at the actual state to <=3.7e-12 N m. The largest assumed-capacity
ratio was 7.25 at a wrist coordinate at 1.313 s: 72.5 N m total, comprising
68.8 state-conditioned reference dynamics, 3.9 PD increment and -0.2 balance;
the corresponding target-state controller value was 65.8 N m. That wrist
capacity is unverified. The earliest exceedance, right hip adduction at
0.068 s, was -144.1 N m versus -16.0 N m evaluated at target state. The
state-conditioned term includes actual-state bias and contact, so neither it
nor the target-state difference is pure open-loop feedforward or isolated
feedback. The results motivate checking both reference dynamics and state/
contact coupling; they do not identify a single cause or justify changing
physiological capacities.

Research checkpoint `codex/15493-face-geometry34` at `287d644490` extracts
finite-face triangles using the existing rendered mesh mask, seating and
frame transforms. It rejects invalid or unrelated geometry and makes no
contact/admission decision. Thirty-nine focused tests and normal push hooks
passed; 149 lines were added and no PR was opened. Initial ball/face overlap,
approaching first-root coverage and end-to-end event verification remain open.

A bounded +2 ms local timing warp preserved q0/q1, sampled closure, modeled
ROM and interval-speed gates, and changed source-model reference marker RMS
24.092 to 24.463 mm. Offline fixed-recorded-state evaluation suggested the
maximum assumed command-capacity ratio would fall from 7.249 to 6.861.
Independent 32-substep ideal continuation from the identical 1.1 s q/v did
not confirm that gain: full comparison source/observed marker RMS improved
35.279/40.911 to 34.829/40.514 mm and peak absolute command 425.60 to
391.68 N m, but maximum modeled capacity ratio worsened 7.249 to 7.528,
236 held commands still exceeded capacities, and peak left subtalar excess
worsened 3.561 to 3.748 degrees. Both had 712 sampled ROM-flagged states and
no sampled speed flags. Reject this as a gate-regressing candidate; modeled
limits remain soft/assumed and actual motor, stage, work and impact were not
verified. Offline fixed-state improvements are not forward guarantees.

A controlled 200 ms fixed-gain Option B motor sensitivity used the identical
v2 spec, target-state feedforward and initial q/v. The baseline 3/0.3 gains
reproduced prior recorded states, commands and drive bit-for-bit: joint RMS
0.1174 rad, three modeled ROM slots, no speed slots or clipping among 6,400
stages. Gains 6/0.6 yielded 0.1901 rad RMS, five ROM and two speed slots,
with 4,416 clipped stages; gains 12/1.2 yielded 0.2676 rad RMS, three ROM
and two speed slots, with 5,728 clipped stages. Each case held its gains
fixed; the higher-gain cases worsened joint tracking and added clipping in
this modeled interval despite lower root-rotation RMS. This is sensitivity
analysis, not a physiological gain recommendation or feasible-swing proof;
16 torque and all speed capacities remain assumption-limited.

Static finite-face clearance research checkpoint `codex/15493-clearance35`
at `34489362a8` builds on face geometry and adds 285 lines (434 relative to
its main base). It computes instantaneous sphere-to-patch gap and rejects
initial nonpositive or insufficient caller-margin clearance. Fifteen focused
tests, including independent constrained-minimization comparisons, and normal
push hooks passed. It cannot certify outside-solid placement, approach, interval
coverage or first root; no PR was opened and address overlap remains open.

In the fixed-gain sweep above, the higher-gain cases first clipped at right
toe actuators after 12 ms (6/0.6) or 6 ms (12/1.2); left/right toe actuators
were dominant clipped slots and their capacities are unverified. The cases
also had toe ROM/speed flags under the model. These results do not establish
verified physiological restrictions or justify escalating gains in place of
addressing the coupled reference and dynamics.

An early local -4 ms reference timing warp preserved q0/q1, sampled closure,
modeled registry ROM and interval-speed bounds before replay. Independent
16/32-substep ideal runs over 200 ms agreed to maximum q/v/held-command
differences 2.61e-7 / 1.70e-5 / 0.00114 in native units. First-100-ms modeled
capacity peak ratio improved 1.608 to 1.477, with violating held commands 27
to 18, but a wrist range first exceeded its modeled bound at 0.195 s (six
samples, peak 0.000388 degrees); the original refined run remained inside
through 200 ms. Reject the candidate under the zero-tolerance modeled gate.
This is ideal-drive numerical evidence, not applied-motor or physiological
acceptance; capacity priors remain assumption-limited.

An early -4 ms timing warp with a declared 0.005 degree right-wrist interior
margin retained q0/q1, closure <=7.31e-12 and modeled reference ROM/interval
speed checks. Independent 16/32-substep ideal replays through 200 ms had no
sampled ROM/speed flags and agreed to maximum q/v/held-command differences
2.61e-7 / 1.70e-5 / 0.00114 in native units. First-100-ms modeled capacity
peak improved 1.608 to 1.477 times the assumed limit and violating commands
27 to 18. It still fails modeled torque capacity, so it is not admitted; the
margin is numerical, not a verified physical threshold. Full-horizon replay
and actual motor stages remain necessary.

A 150 ms, three-window actual motor Option B continuation preserved the first
100 ms sampled gates and reduced maximum 150 ms modeled ROM excess from
0.009191 to 0.004383 rad, but two ROM slots still failed and tracking RMS
rose 0.07643 to 0.07765 rad. No stages clipped in the 150 ms comparison.
Independent full replay first flagged ROM at about 101.5 ms and hit a guard
at 1.110 s with 31 ROM and 31 speed slots flagged. Sixteen/32-substep motor
checks through 150 ms agreed closely on the violations (peak difference
<4e-9 rad), not on full-horizon feasibility. Work and slew limits and the
physical provenance of several capacities remain unverified.

The 32-substep early-warp/wrist-margin ideal continuation completed 1.319 s
without numerical failure. Against the refined original it changed source/
observed marker RMS from 35.279/40.911 to 36.198/41.703 mm and sampled
modeled ROM-flagged states from 712 to 614, but right subtalar peak excess
worsened 1.1058 to 1.1118 degrees and left wrist 0.4471 to 0.4487 degrees.
Modeled speed flags remained zero; held ideal commands above assumed capacity
changed 236 to 234, peak ratio 7.24873 to 7.25312, and peak absolute command
425.6 to 389.7 N m. Reject as gate-regressing. Sixteen/32 agreement for this
candidate was checked only through 200 ms; actual motor stages, work, impact
and physical feasibility remain unverified.

A further 150 ms exact-seam motor restoration reduced worst sampled modeled
ROM excess 0.004383 to 0.001266 rad but left two coordinates failing and
increased tracking RMS. The full replay first flagged ROM around 102.7 ms and
stopped at 0.861 s with 32 ROM and 26 speed slots flagged. Ill-conditioned
selected sensitivities and this local failure do not establish global
infeasibility; no physical acceptance is implied.

Advancing finite-face diagnostic checkpoint `codex/15493-contact-diagnostic36`
at `5963d3cad4` checks sampled positive-to-nonpositive gaps with finite states,
front side, strict triangle interior and material-point approach, and rejects
initial overlap. Twenty-three geometry/clearance/diagnostic tests and normal
push hooks passed, including hidden crossings, discontinuous gaps, back side,
edge, endpoint and a synthetic pinned-v2 MuJoCo step-halving case. Both hit
and miss remain unqualified and coverage uncertified: sampled scans do not
exclude an earlier crossing. This 353-line increment has a 787-line research
ancestry, so it is not a <500-line PR or a completed Slice 1.

A 16/32-substep motor-state KKT acceleration audit of the continuous
reference exposed a support/reference mismatch before large tracking drift.
At the initial reference state, target-controller root translation acceleration
differed from the finite-difference reference by 8.27 m/s² while modeled
support was 0.165 body weights; at 75 ms the defect was 9.27 m/s² with
0.073 body weights of modeled support. At the actual motor state at 75 ms,
modeled support was 1.97 body weights and root acceleration mismatch was
28.2 m/s². The target-state inverse solve applies no direct root actuation and
cannot generally force the prescribed root acceleration. These are instantaneous
model calculations under assumed compliant contact, not measured ground
forces or a causal proof. They motivate a closure- and continuity-constrained
support-aware reference fit as well as controller work.

The sampled face diagnostic's trial-prefix advance repartitions its RK4
steps as query time changes; the existing integrator has no dense-output/error
enclosure and applies nonlinear closure projection at step boundaries. Sampled
velocities or observed maxima therefore do not establish a continuous
trajectory bound or first-root certificate. A fixed-lattice diagnostic can
remove that partition seam, but remains conditional without a validated
continuous error enclosure and projection-continuity treatment.

At the continuous-reference motor state, finite-difference joint-to-root
translation acceleration sensitivity had rank three at the sampled early
states; no direct root effort was applied. A -0.05 N m hip spline change
slightly reduced the initial translation defect while retaining 100 ms
sampled modeled ROM/speed gates and no clipping; 16/32 motor substeps agreed
locally. It did not resolve the roughly 8.1 m/s² defect.

A local least-norm linear estimate to eliminate that initial defect required
20.85 N m perturbation norm and 21.61 N m total spline coefficient norm,
above the existing 10 N m diagnostic trust radius. A bounded radius-10
estimate reduced the instantaneous modeled defect to 4.392 m/s², but its
actual 100 ms motor replay failed three modeled ROM slots and worsened joint
tracking. Even the smallest of five tested steps failed early ROM gates.
Reject all tested candidates; the linear estimate addresses only initial
translation and does not prove global infeasibility, physiology, work or
contact. Torque and speed priors retain their documented assumptions.
