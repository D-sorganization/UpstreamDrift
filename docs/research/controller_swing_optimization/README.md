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
