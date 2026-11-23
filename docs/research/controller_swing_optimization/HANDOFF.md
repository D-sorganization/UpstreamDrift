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
