# Controller-Based Swing Optimization

This research record tracks the evidence for UpstreamDrift #15493. The
calculation-level reference is [swing optimization](swing_optimization.tex).
It is an editable research reference, not the engineering design manual.
The initial source compiled with `pdflatex --disable-installer
-interaction=nonstopmode -halt-on-error` on this host; compilation does not
establish scientific acceptance.

## Baseline Evidence

The recent Brick straight-flight showpiece uses a kinematic fit. Its reported
12.2 mm whole-swing marker RMS is not a forward-dynamics tracking error, and
its bundle has no recorded efforts. It uses the fitted trajectory with the
last two frames repaired and a replay-time club-roll adjustment. Its documented
flight prediction is approximately 211 yd carry and zero lateral displacement
after rounding; it is not an optimized forward-dynamics result. The linked
fit, model and source receipt have been copied to private scratch with matching
hashes. The saved feedback rollout is filtered relative to the video target.

The requested pinned v2 baseline is a separate artifact. Its ledger describes
an older v1 feedback fit rejected on closure and support criteria; the render
entrypoint migrates the v1 fit to v2 for replay rather than refitting it. The
Brick model shares its coordinate order but differs from the pinned v2 spec.
Do not conflate the two or claim either is an admitted forward baseline.
The private input matching the renderer pin has now been located. Its embedded
calibrated v1 model differs materially from the committed v1 migration model;
the existing migration attempts to track the prior rollout, not the original
tracking target. In a local attempt, closure projection failed with maximum
residual 0.8982 before a v2 rollout was produced. Coordinate-name agreement
alone does not establish geometry, calibration or closure equivalence.
The new migration precondition rejects the mismatched embedded model before
generating a bundle. It was checked against the actual pinned input; the input
and both committed model files remained unchanged and no output was written.

An exploratory first-frame fit on the unchanged committed v2 spec reached the
60-iteration cap. Using 25 valid markers with available offsets, it yielded
53.39 mm marker RMS, 1.018 mm closure translation norm and 0.000253 rad closure
rotation; its lowest foot sphere remained 67.1 mm above the inferred ground.
Eight marker offsets were unset and no planted-foot condition was imposed.
This is a diagnostic of initialization, not an accepted fit or full swing.
A subsequent bounded first-frame solve grounded the feet and anchored their
in-plane locations to an earlier fitted pose, not measured stance anchors. It
converged in 45 evaluations with 49.34 mm marker RMS on the same 25 markers,
0.715 mm closure translation norm and 0.0000401 rad closure rotation. The
lowest sphere was 0.377 mm below the inferred ground. These first-frame results
do not demonstrate trajectory, force, or full-marker feasibility.
Over ten frames, warm-starting from the preceding fitted pose produced a peak
coordinate speed of 54.02 rad/s, including an 8.6-degree forearm change between
the first two frames; it was not a whole-turn wrap. A separate run seeded from
the migrated historical rollout at each frame had a 2.30 rad/s peak and
58.19--58.22 mm marker RMS on the 25 available markers. Both are exploratory,
use assumed stance anchors and lack full anatomical speed and torque checks.
The source receipt supplies the eight missing offsets as a separate observation
map. A provisional transfer to the unchanged v2 model gave 96.37 mm RMS on 33
valid markers and stopped at a bound. The body-frame conventions agree, but a
six-state native-FK comparison showed 99.7--218.8 mm RMS world-position
differences for the transferred markers. Different pose and model geometry
contribute, so an identity offset transfer is not yet a validated calibration.
A four-seed, fixed-v2 first-frame sweep using that provisional observation map,
assumed anchors and the available zero-tolerance ROM bounds did not converge:
marker RMS was 107.70--112.14 mm on 33 valid markers and closure translation
was 3.16--3.61 mm. The available ROM table omits shoulder Euler components;
this bounded experiment does not prove that no feasible fit exists.
A subsequent 24-case first-frame sensitivity sweep varied four seeds, solver
and assumed stance anchoring without changing the committed model. No case was
classified converged by the repository: 12 reached the LM iteration cap and
12 were classified at a bound, although SciPy reported success for 11 of the
12 bounded TRF cases. The best unanchored TRF case had 69.41 mm RMS on 33
valid markers, 3.69 mm closure translation, and foot-sphere heights from
0.69 mm below to 2.11 mm above the inferred ground. The alternative anchors
were assumptions, and the finite sweep does not prove infeasibility. The LM
path clips trial states to named bounds, which is not a constrained-optimality
certificate; shoulder Euler anatomy remains unassessed.
In a separate first-frame control using the same 33-marker observation map,
common named bounds and solver settings, the source v1 model gave 3.11 mm RMS
and the committed v2 model 60.24 mm RMS without foot anchors. With the same
assumed source-pose anchors, the figures were 3.42 and 67.56 mm respectively;
the source solve converged and the v2 solve was classified at bounds. This
supports a model/seed-geometry mismatch under these assumptions, not a unique
cause or proof that a valid v2 fit cannot exist. The source model was calibrated
to these observations, while transferring its offsets to v2 is provisional.
In the unanchored v2 control, residual RMS was largest for the trail arm
(117.5 mm) and club (76.1 mm), with pelvis and trunk at 43.0 and 24.7 mm.
Five v2 coordinates were at bounds; closure translation was 2.85 mm versus
0.047 mm for the source control. Grip, wrist and shoulder geometry and the
initial pose differ, so the group errors do not identify a single cause.
In a six-state FK diagnostic, corresponding local closure-site transforms
matched, while some upper-body joint-frame orientations differed. At the
initial migrated pose, wrist-associated body positions differed by roughly
295--582 mm. Those comparisons combine frame convention, trunk geometry and
closure mismatch; they are not anatomical-angle errors or a unique diagnosis.
Local FK probes found that five right-side scapula, shoulder and wrist
coordinates need a sign reversal to match source-model motion increments in
the tested joint contexts; a separate FK check on the same model pair agreed
on the isolated and combined-angle probes. Applying that provisional map at eight saved
source-rollout states reduced the worst sampled closure translation from
1,029 mm to 39.6 mm, but rotational closure still reached 179.9 degrees and
marker discrepancies remained large. The map has not been validated as a
full-swing anatomical mapping, and wrist neutral frames remain different.
A first-frame fixed-prior fit using the map as an initialization found a lower
basin: 30.35 mm marker RMS and 0.818 mm closure translation, versus 60.24 mm
and 2.853 mm from the original initialization, on the same 33 valid markers
and with the same unanchored objective. Both were classified at bounds; this
does not isolate the effect of individual sign changes or establish an
accepted trajectory. The transferred marker map remains provisional.
Over the first 25 ms, a fixed-prior continuation from that candidate produced
28.69--28.72 mm marker RMS and a 0.096 rad/s peak finite-difference coordinate
speed in the sampled frames; a changing-prior variant produced a 34.08 rad/s
peak. Both used assumed anchors, were classified at bounds, and cover only the
named ROM ranges. These kinematic diagnostics are not a dynamic rollout or an
anatomical-speed assessment.
An exploratory 200 ms continuation compared the fixed prior with a
time-aligned, sign-mapped source-rollout prior. Marker RMS ranged from 27.8 to
30.3 mm and peak finite-difference coordinate speed was 0.58 and 1.82 rad/s,
respectively. All 146 fitted frames were classified at bounds. Both runs used
assumed fixed foot anchors and the provisional observation map; they do not
establish full-swing tracking or direct anatomical rates.
Two exploratory continuations then reached the sample immediately before the
recorded impact split using either the original tracking prior or saved-rollout
prior. Marker RMS ranged 25.4--38.0 mm and closure translation stayed below
1.3 mm, but every frame was classified at bounds. Both showed an 84.5--84.7
rad/s finite-difference forearm-coordinate spike near impact while the source
prior remained smooth and the valid marker count briefly fell. Dropout is a
plausible contributor, not an established cause. These remain kinematic fits
with provisional offsets and assumed anchors, not physical acceptance.
A local three-frame check using only the observed markers reduced the largest
finite-difference coordinate jump from 84.67 to 8.22 rad/s with a stronger
previous-fit continuity prior; the dropout-frame marker RMS rose from 25.58 to
33.85 mm. All solutions were classified at bounds, and the changed prior is a
different objective, not an anatomical-speed guarantee. Raising a source
prior instead worsened fit and closure in the tested cases. This supports
investigating underdetermination, without establishing dropout as its cause.
Across all 477 pre-impact frames, previous-fit continuity weights 0.1 and 1.0
reduced the maximum sampled coordinate rate from 84.67 to 43.88 and
25.00 rad/s respectively. Maximum marker RMS was 38.00, 36.49 and 66.44 mm;
maximum closure translation was 1.281, 1.093 and 4.790 mm. All frames were
classified at bounds. The stronger prior substantially trades fit for
smoothness; neither rate is an anatomical or physical acceptance result.
Intermediate weights 0.2, 0.35, 0.5 and 0.75 also reached the pre-impact
window. Their peak coordinate rates were 32.21, 29.49, 28.02 and 26.38 rad/s;
maximum marker RMS was 37.07, 42.78, 49.26 and 58.59 mm, respectively. Every
frame was still classified at bounds. These runs use the same provisional
observation map, named bounds, assumed foot anchors and unfilled marker masks;
they characterize a fitting tradeoff, not an optimized or feasible shot.
The rate calculation excludes all six root coordinates; the remaining
compiled coordinates are hinges. The largest rate at weights 0.2--1.0 moved
to a wrist coordinate near impact, where all 33 markers were available, rather
than the dropout frame. At weight 0.2, impact-window marker RMS had median
31.50 mm and maximum 35.80 mm, compared with 29.61 and 30.72 mm for the
source-prior baseline. These windows were examined in the same data used for
the sweep, not held-out validation.
At one common pre-impact sample, v2 FK face-centre speed was 50.68 m/s for
the source-prior fit and 49.39 m/s at continuity weight 0.2; the unsigned
horizontal angle between face normal and velocity was 1.3 and 11.6 degrees.
These are time- and frame-sensitive kinematic proxies, not target-relative
face-to-path measurements or evidence of contact, ballflight, or carry. A
different finite-difference stencil changed velocity by 0.31--0.74 m/s.
A 200 ms forward-tracking diagnostic used the exact committed v2 model and
the exploratory continuity 0.1 / face 0.1 fit, with the existing computed-
torque controller, 1 ms zero-order hold and eight RK4 substeps. It completed
200 steps with 6,400 expected stage records and no ball force. Joint-position
tracking RMS was 0.007994 rad, root translation RMS 19.21 mm, and peak held
source effort 303.05 N m. A zero-tolerance check of the available named ROM
table flagged four coordinates (largest excess 0.742 degrees). The recorded
effort is not established as post-saturation torque; complete anatomical,
speed and torque gates remain indeterminate. This is neither Option B nor an
admitted baseline.
The same diagnostic was extended to 1.322 s, immediately before the recorded
impact split. It completed 1,322 steps and 42,304 expected RK stages. Joint
position tracking RMS was 0.005049 rad, root translation RMS 47.90 mm, and
peak held root-zeroed source effort 1,193.91 N m. Six coordinates exceeded
the available named ROM ranges in stage or grid samples; the largest excess was
0.742 degrees. The sampled support model reported its center of pressure inside
the polygon in 62.89% of grid samples. No impact force or Option B controller
was used, and missing full anatomy, speed and torque bounds leave hard
feasibility indeterminate. Stage points are numerical evaluation states, not
independent physical observations.
A sensitivity check of those 42,304 stage rows against the existing coordinate
envelope found directional held-effort exceedances in 33 of 41 non-root slots,
coordinate-ROM exceedances in seven, and no generalized-speed-prior
exceedances. The largest effort ratio was 62.85 against an assumed capacity;
the largest coordinate-ROM excess was 13.44 degrees. Six soft ROM slots,
16 assumed torque slots and all 41 assumption-marked speed limits apply to
that inventory. The held source command has not been proven to be
post-saturation actuator torque, and coordinate mappings are not a complete
anatomical model. These are model-assumption sensitivities, not clinical or
physical certification.
At four sampled hold states, fixed-state inverse-dynamics differences
reconstructed the existing controller's commands within 4.55e-13 N m. At
the peak thoracic axial command of -1,193.91 N m, the acceleration-reference
term contributed -1,178.81 N m; feedback, balance and bias made up the
remainder. This is local to those states, uses the existing computed-torque
controller, and does not establish a physical motor torque or the cause of
every envelope exceedance.
The peak coincided with a marker-dropout frame, although that does not prove
causation. The diagnostic tracker differentiated the fitted positions twice
without filtering. Applying the repository's 12 Hz zero-phase reference filter
to pre-impact data reduced the fixed-state command norm from 2,720 to 643 N m,
but maximum marker RMS rose from 36.67 to 47.51 mm, maximum closure translation
from 2.27 to 8.52 mm, and six named ROM flags appeared. The pre-impact
face-centre speed proxy fell from 49.73 to 45.09 m/s. Filtering used later
pre-impact samples and edge padding; it was not a new rollout or an acceptable
standalone correction.
Applying the existing trail-arm weld projection after that filter reduced
maximum positional closure from 8.519 to 0.000860 mm, with little change in
marker RMS. It did not enforce ROM: one named wrist excess grew from 0.063 to
1.719 degrees. The projection therefore does not establish a hard-feasible
reference, even though closure improved.
A seven-frame bounded pose-solver diagnostic used in-bounds original seeds,
the filtered poses as a prior and the actual available observations. It met
the 29 supplied named bounds, with closure translation 0.290--1.650 mm and
marker RMS 23.56--35.89 mm. All seven were classified at bounds; closure was
a soft residual and 18 model coordinates lack named ranges. This is not a
full anatomical or trajectory acceptance.
Extending that independent-frame diagnostic to all 477 pre-impact frames
preserved the supplied bounds, but every solve was classified at bounds and
not converged. Maximum closure translation fell from 8.519 to 2.226 mm and
maximum marker RMS from 47.515 to 36.577 mm. Peak twice-differentiated
reference acceleration rose from 679 to 6,692 rad/s²; at the dropout hold,
the fixed-state command norm rose from 643 to 2,719 N m. These are reference
and fixed-state sensitivities, not rollout torques. The result does not
establish a feasible trajectory or complete anatomical coverage.
A 13-frame dropout-window diagnostic instead used each previous bounded fit
as the next seed and prior; all seeds were checked in bounds. All 13 solves
remained at bounds and not converged. Peak consecutive angular finite-
difference speed was 41.70 rad/s, versus 43.84 for independent refits and
11.44 for the filtered reference. The filtered reference exceeded four
named ranges in that window; the two bounded refits met the supplied ranges.
The sequential fit does not establish smooth or anatomically complete motion.
A soft face-axis target inferred from observed head-marker orientation was
available in 476 of 477 pre-impact frames, but not at the dropout frame; no
target was imputed there. At continuity weight 0.1, adding face weight 3.0
reduced the target's median angular residual from 8.30 to 0.80 degrees, while
maximum marker RMS rose from 36.49 to 55.14 mm and maximum closure translation
from 1.09 to 7.48 mm. The target depends on calibrated offsets and assumed
club roll; fitting it is not independent face or contact validation.
An intermediate face-weight sweep found no unqualified optimum. At continuity
weight 0.1, face weight 0.1 gave 19.20 degrees face-residual p95, 36.67 mm
maximum marker RMS and 2.27 mm maximum closure translation; face weight 1.0
gave 3.12 degrees, 49.07 mm and 5.59 mm, respectively. All frames were
classified at bounds. These are fitted-data sensitivity results, not physical
acceptance or ballflight predictions.
A bounded, wrist-coordinate-only diagnostic on six saved poses did not match
both parent-relative translation and rotation in all cases; several solutions
hit a wrist bound. Translation and rotation were combined with arbitrary
numerical weights, each pose was solved independently, and the compared body
origins include a remote club origin. This neither establishes hand-surface
separation nor proves that a valid full-model fit is impossible.
An exploratory calibrated v2 derivative can be built using the source
receipt's hip-center landmarks and the existing pelvis relocation. Its
migrated initial state closes to numerical precision, with 5.44 mm RMS
world-position difference over the transferred markers. It is a separate
model, not the committed v2 spec: total mass changed by 0.300 kg and zero-pose
center of mass shifted by 9.05 mm. Those differences need explanation and
validation before it can be admitted or compared as physically equivalent.
The v2 derivation explicitly removes a 0.300 kg dummy body; this explains the
mass difference but not the entire COM or inertia difference, which also
reflects relocation and pose changes. A bounded first-frame derivative fit
gave 40.08 mm RMS on 33 valid markers and 0.560 mm closure translation, but
stopped at a bound without convergence. The solver classifies any active or
near bound as nonconverged; this is not proof of physical infeasibility. It is
not an accepted model. Reassigning the removed mass is possible mechanically,
but its segment attachment has not been validated over articulated motion.

A fresh headless run of the Brick model's existing computed-torque tracker,
without ball force or the replay-time roll adjustment, produced 1,815 finite
states. Against its repaired raw target, sampled joint position RMS was
0.07069 rad and maximum absolute error 1.19346 rad; velocity RMS was
1.2101 rad/s using numerically differentiated target positions. Peak sampled
controller effort was 1,662.81 N m. A zero-tolerance check using the existing
partial ROM table flagged nine coordinates. These are diagnostic samples,
not a complete anatomical or continuous-time feasibility assessment, and the
run is neither the pinned-v2 baseline nor the required controller.

A previously committed forward rollout
is a different artifact: its saved marker errors give 99.5301 mm RMS, and its
receipt reports 0.04080 rad joint tracking RMS and 8.879 mm root tracking RMS.
Those numbers must not be attributed to the current best match. The historical
rollout has four reported range-of-motion flags with the existing 0.5 degree
tolerance; that helper does not cover all anatomical joints.

The existing controller uses computed torque. It is not the fixed-gain
feedforward-plus-PD controller required by this epic. The recorded torque is
not sufficient to establish the total torque at every integration stage, and
no complete anatomical speed or torque-capacity audit has been established.
Do not admit a baseline or candidate as physically feasible on these numbers.

The existing `impact_optimization.PhysiologicalEnvelope` supplies asymmetric
torque capacities, ROM and velocity bounds with provenance flags. Its current
rollout controls a subset of joints open loop; its kinematic binding check
estimates rates from sampled positions. Neither establishes this epic's
full-body feedback-torque and anatomical-speed gates. A separate historical
same-input bundle exceeded 12 ROM bound sides, 8 of 44 speed limits and 36 of
44 torque capacities at recorded samples. Its limits include 19 assumed
torque slots, 44 unverified velocity limits, soft ROM and unbounded root
rotations. These assumption-limited checks are not current-best results or
clinical judgments.

The experimental sampled-evidence audit checks supplied ROM, direct rates,
and total post-saturation joint torque against the envelope. It reports assumed
limits and rejects missing channels, duplicate names, unsupported mappings and
nonfinite records. Stage identifiers distinguish the two RK4 midpoint states.
The inventory, mapping, provenance and expected schedule remain caller
assertions: a sampled pass is neither verified full-rollout coverage nor an
anatomical or continuous-time feasibility certificate. It is not wired to the
controller or an optimization acceptance gate.

The optional zero-order-hold recorder retains numerical RK4 stage states and
the held source effort after root zeroing, with accepted segment plans and
pre/post-projection states. Contact-search trial advances are excluded, and
failed steps discard their pending trace. A finish-time integrity check rejects
missing, reordered or malformed stages and inconsistent held efforts. This is
not evidence that the held
effort is the requested controller's post-saturation total torque. Plan and
stage records are produced by the same integrator, and neither they nor the
sampled audit prove completeness against an independent actuator inventory or
continuous-time bounds.

The current v2 envelope has 44 non-translation slots; three are pelvis root
rotations that the integrator zeroes, not 44 independent actuators. All 44
velocity bounds carry assumption flags (35 unverified, nine sourced with an
assumed margin). Of 41 non-root torque slots, 16 are assumed. The envelope
also contains soft trunk limits and unbounded root rotations. Group-level
studies such as [Liu et al.](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0331051)
do not directly provide subject-specific limits for every model coordinate.
The [Anderson et al. torque study](https://stacks.cdc.gov/view/cdc/188754/cdc_188754_DS1.pdf)
illustrates posture and velocity dependence that a fixed torque ceiling does
not capture. The existing anatomical-angle modules require calibrated segment
frames and do not establish complete v2 mappings or direct anatomical rates.

To reproduce the historical sampled-limit counts from the repository root:

```powershell
@'
import json
from pathlib import Path
import numpy as np
from src.shared.python.impact_optimization.physiological_envelope import PhysiologicalEnvelope, load_limits
from src.shared.python.loop_nullspace.capacity import CapacitySubject
p = Path('docs/development/full_body_models/evidence/trunk_visuals/driver_v2_bundle.npz')
with np.load(p, allow_pickle=False) as z:
    manifest = json.loads(str(z['manifest'].item()))
    spec = json.loads(bytes(z['spec']).decode('utf-8'))
    env = PhysiologicalEnvelope.from_spec(spec, CapacitySubject.from_spec_subject(spec['subject']), load_limits())
    idx = [manifest['coordinate_order'].index(n) for n in env.names]
    q = np.degrees(z['reference_q'][:, idx])
    v = np.degrees(z['reference_v'][:, idx])
    tau = z['efforts'][:, idx]
    rom = sum(int(np.isfinite(lo) and q[:, i].min() < lo - 1e-9) + int(np.isfinite(hi) and q[:, i].max() > hi + 1e-9) for i, (lo, hi) in enumerate(env.rom_deg))
    speed = np.sum(np.max(np.abs(v), axis=0) > env.vel_max_deg_s + 1e-9)
    cap = np.where(tau >= 0, env.tau_max_positive_nm, env.tau_max_negative_nm)
    torque = np.sum(np.max(np.abs(tau) / cap, axis=0) > 1 + 1e-9)
    print(rom, speed, torque, len(env.names))
'@ | python3 -
```

The output is `12 8 36 44`. This checks the bundle's recorded states and
held efforts only; it does not prove between-sample or anatomical coverage.

## Prescribed-Launch Check

At main `d0eda5b47c9b11193005391e55bf002c4d7c3b89`, the existing
`tests/unit/physics/test_reference_shots_15443.py` calls
`BallFlightSimulator` directly, independently of the impact pipeline. With
the revision-2 Rust kernel built from this checkout and Python 3.12.10, the
suite produced **15 passed, 1 expected failure**. The expected failure is the
pitching-wedge carry; several fixture values are approximate, and the air
conditions of the published reference rows are not normalized. Passing this
suite is regression evidence, not validation of swing-to-impact carry.

The system Python 3.14 wheel reported physics revision 0 and failed 14 flight
cases before simulation. The isolated revision-2 wheel avoided modifying that
installation. A first build attempt using Python 3.14 failed because this
checkout's PyO3 version supports through Python 3.13.

Reproduction from the repository root, using a Python 3.12 environment with
the pinned Tools submodule, test dependencies and `maturin` installed:

```powershell
python -m maturin build --manifest-path rust_core/upstream-physics/Cargo.toml --release --interpreter <python312> --out <wheel-dir>
<python312> -m zipfile -e <built-wheel> <isolated-kernel-dir>
$env:PYTHONPATH='<isolated-kernel-dir>'
$env:QT_QPA_PLATFORM='offscreen'; $env:MPLBACKEND='Agg'; $env:PYTEST_QT_API='pyqt6'
$env:MPLCONFIGDIR='<writable-matplotlib-config-dir>'
<python312> -m pytest -q -o addopts='' tests/unit/physics/test_reference_shots_15443.py
```

Replace placeholders with local paths. Check the imported kernel's
`PHYSICS_REVISION` and origin when reproducing the result; do not install over
an existing shared environment merely to run this check.

## Contact Verification

The opt-in surface trigger is experimental. A synthetic one-step MuJoCo
probe using the rendered face patch and production RK4 stepper found a
bracketed event with a sphere-to-face residual of approximately 1.4e-11 m.
The public synthetic MuJoCo regression compares 1 ms and 0.5 ms steps and
checks event state and residual, using the shared RK4 stepper. Neither check is
an impact verification for the pinned swing. The event finder searches nine
states for a sign-change bracket and can miss tangencies or multiple crossings
within one grid cell. Its finite nine-sample miss penalty is a smooth
surrogate, not a proven global closest distance. Independent validation on the
actual matched trajectory remains necessary.

The search reuses rigid-face boundary topology within a search. On the modeled
576-triangle driver face with synthetic motion, a 30-run median was 10.6 ms
for a miss and 34.0 ms for a hit, versus 46.5 ms and 202.9 ms when rebuilding
the boundary for every sample. These are local computational timings, not
swing-speed gains. The cache requires stable triangle order and rigid motion;
deformation or insufficient coordinate resolution is rejected.

## Open Admission Gates

- Resolve the verified pinned bundle's model mismatch and run forward tracking
  on the specified v2 model against the actual reference.
  Record runtime provenance and distinguish filtered and unfiltered targets.
- Validate anatomical ROM, speed and total applied torque for every relevant
  joint, including controller feedback and between-sample behavior. Report
  assumed or unavailable limits and fail closed where coverage is missing.
- Establish an approaching bracketed contact root on the modeled finite face
  using the simulated state, with convergence and missed-contact checks.
- Verify the flight fixture limitations before interpreting carry changes.

No optimized candidate or farther-and-straight result is established here.
