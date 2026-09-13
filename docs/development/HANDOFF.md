# Native Multi-Engine Matching Handoff

## Current Status

Work RESUMED 2026-09-12 (claude, leases on #9967/#10043). The objective is
incomplete. All numerical jobs through77 are terminal. Audit77 qualifies every
run76 derivative block against measured floors; the next lane is the bounded
direct-node two-window SLSQP trial (once-only shared boundary for parity). Resume instructions:
[Agent Resume Prompt](simscape_tour_matching/AGENT_RESUME_PROMPT.md).

The full matching goal is OPEN. Robust trajectory equivalence of the alternate
representation is not yet qualified. No accepted full-swing open-loop sixth-order
candidate exists. MATLAB R2025b is the required reference release. Keep the
original capture, physical model, initial state, actuator mapping and acceptance
criteria; quaternion conversion does not change the required native torque family.

Branch: feat/9967-native-simscape-pinocchio. Checkpointf96766c8e is pushed with
all normal hooks passing. SELF archives terminal audit77, its reclassification and
the once-only shared-boundary shooting option. PR not created. Issue #9967 owns native matching; #10043 owns representation work,
under #9921. Check/renew the lease before new issue work. Workspace:
C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native.

Latest cleanly returned exploratory fit: run73,0.85 s only, whole RMS28.105 mm
and terminal65.398 mm. It is rejected, not a full-swing match. See
[Candidate](simscape_tour_matching/native_evidence/regularized_fit_9967_73/returned-candidate.json)
and [Measured Comparison](simscape_tour_matching/native_evidence/regularized_fit_9967_73/marker-comparison.png).

## Representation Qualification

Run58 is terminal (handle2605 exited0). Local-chart DOP853 on native spherical
Pinocchio PASSES all existing parity gates over the run19 0.85-second fixture:
marker maximum2.62982e-8 m; native q maximum1.28599e-7; native rate
maximum1.15849e-5; closure pose3.74372e-11 and rate9.40426e-11.
Elapsed37.6895 s/9972 RHS calls versus7.2147 s for the tighter scalar reference.
This is a same-input representation result, NOT a C3D fit or full R2025b parity.

Run59 is TERMINAL, original handle97788 exited0: same setup with smaller maximum
step. It FAILS native velocity parity (0.000587270); marker maximum1.15187e-7 m,
native q6.06676e-7, closure pose6.60249e-11/rate1.61521e-10. Elapsed61.2725 s.
Therefore58's isolated pass is not robust convergence or representation acceptance.
Those representation runs are terminal. The pointwise acceleration/history audit
found no sampled history dependence or substantial actuator-route mismatch:
maximum routed native acceleration difference2.18034e-6 rad/s² at a reference
539073.49 rad/s²; effort roundtrip1.25056e-12. Sampled unscaled inertia condition
is about6.7e7 in both representations. These results support investigating
conditioning/trajectory sensitivity, not declaring a physical-model mismatch.
They do not establish global convergence. Detailed receipt:
simscape_tour_matching/native_evidence/manifold_acceleration_10043_19.
Remote runtime /home/dieterolson/native-manifold-10043-18; driver
/mnt/c/Users/diete/compare_native_manifold_replay_9967_59.py; output
/mnt/c/Users/diete/native-manifold-replay-9967-59. Horizon0.85, methoddop853,
rtol1e-12, atol1e-14, max_step1/1440, max_evaluations100000. Scalar reference
rtol1e-12, atol1e-14, max_step0.000125. Poll the existing handle/process before
launching another run. All root runs through59 are terminal; none should restart.

## Transition Sensitivity and Regularized Fitting

Run60 completes six original-state replays. Repeated baseline results are exactly
identical. A late LSInputX B6 perturbation of1e-6 Nm changes native rates by
0.00750466 rad/s near0.786111 s but markers by only0.5386 micrometres. The
1e-4 Nm trials produce about0.765 rad/s and50.8 micrometres. Central derivative
estimates retain amplitude dependence; these are measured sensitivity evidence,
not a waiver of parity gates. Exact inputs, executed source and raw trajectories
are archived in native_evidence/perturbation_9967_60.

Shared prefix and multiple-shooting fitters now accept checked analytic penalty
Jacobians. NativeEffortPenalty reuses the native actuator/frame provider and
seven-point quadrature to evaluate exact mean-square total degree-six primitive
efforts. Twelve new tests passed after RED; combined fitter/effort/control regression77 tests
passes, as do Ruff and four-module mypy. Immutable ControlTower runtime61 passes
96 focused/real Pinocchio tests; source archives/hashes are preserved in
native_evidence/regularized_fit_runtime_9967_61. Existing runner behavior remains
the default with zero penalty. Trial61 is terminal0 in75.40 s: whole RMS29.813 mm, early10.707 mm,
terminal75.846 mm, club23.930 mm, versus baseline30.791/10.667/99.989/56.557 mm.
All three primal sensitivity gates pass. Max_nfev3 was reached; no numerical
acceptance or convergence is claimed. Native effort cost barely changes
(1.55060 to1.55026), so improvement cannot be attributed to the penalty alone.
Exact evidence and convenient candidate: native_evidence/regularized_fit_9967_61.

Run62 is terminal0 after468.965 s: whole28.6353 mm, early10.8248 mm,
terminal66.7134 mm, club31.0615 mm. It reaches max_nfev20 with four active
correction bounds; accepted/converged both false. Returned canonical candidate
7467c5d82817251858255bdf0e560a712f0d20a366ee0094c364a6c14481e1d5.
Native evidence folder regularized_fit_9967_62 contains the terminal receipt,
measured replay arrays and marker-comparison.png (root visually inspected).
Run63 is terminal1 after77.672 s and three forward evaluations. The B4/B5/B6
trial fails the unchanged sensitivity-primal marker agreement gate on its third
candidate, hash e057a7e977cfe2a163fe4c5e7e5767c17d5025b4ef7f33166672afbf658c358d.
There is no returned accepted candidate. First two Jacobians passed. Diagnostic64
PASSES at that exact candidate before its600 s wall budget: augmented solve
409.086 s/1,222,535 evaluations, primal marker discrepancy8.9147861e-9 m against
the unchanged1e-7 gate. Total launch466.239 s; diagnostic call/save419.190 s.
The augmented settings were rtol1e-12/atol1e-14/max_step0.000125. The provider's
independent primal remained rtol1e-11/atol1e-13; max_step applies to both. This
supports an accuracy-dependent63 failure, not a derivative correctness or C3D
acceptance claim. Cost is far above the default sensitivity calls. All jobs through69 are terminal; diagnostic70 is the next authorized experiment.
Runtime65 adds the checked call-budget API and passes36 tests including nine
real Pinocchio tests. Diagnostic65 changes only max_step to0.000125 relative
to63, retaining augmented tolerances1e-10/1e-12: marker difference3.88459e-8 m,
82931 sensitivity calls,27.0993 s (versus64's409.086 s). Gates are unchanged.
Audit66 checks saved64 LSInputX B6 column41 against six original-state replays:
relative errors9.11e-5/4.94e-4/9.53e-6 for1e-5/1e-4/1e-3 Nm, all below the
existing1e-3 threshold. This verifies one direction, not the entire Jacobian.
Raw sources/inputs/replays are preserved in sensitivity_9967_65 and derivative_9967_66.

Runner now accepts --max-step for both forward and sensitivity paths and
--max-sensitivity-evaluations; defaults preserve prior behavior. Five new parser
checks went RED/GREEN;13 runner tests pass. Runtime68 qualifies32 tests including nine real Pinocchio tests, plus runner
--help. Fit68 is TERMINAL1 after425.444 s, former PID2842183 / handle54657;
output /mnt/c/Users/diete/native-regularized-fit-9967-68. Authorized fit68 retains original
baseline19, restarts returned62, uses B4/B5/B6,max_nfev10,max_step0.000125,
sensitivity budget100000, and unchanged weight0.01/scales/amplitude/bounds.
Eight Jacobian checks pass, then the ninth candidate fails unchanged marker
agreement (not its evaluation budget). Candidate canonical hash:
c8885ace0b1021dea6e2fbd824286fd37eeff8cb91498147d2d585cb64bed039.
The last unqualified checkpoint reports28.138 mm whole/65.565 mm terminal;
it is not a returned solution. Exact evidence: native_evidence/regularized_fit_9967_68.
All previous numerical runtimes remain immutable.

Manufactured diagnostic67 confirms inactive sensitivity-column padding changes
physical integration error under the existing combined error norm. This suggests
an architecture improvement, not a proof of native model error. Keep it as a
regression case for opt-in grouped DOP853 error control, now implemented through
shared integrate_forward/integrate_sensitivities and the native provider/runner.
The maximum of SciPy's existing physical and sensitivity block norms controls
steps; default behavior is unchanged. The protected SciPy hook requires runtime
qualification. Initial-step selection remains unchanged.57 focused tests pass
after RED/GREEN, plus Ruff; no global accuracy guarantee is inferred.
Diagnostic70 is TERMINAL1 on the exact failed68 candidate, max_step0.000125,
rtol1e-10/atol1e-12,100000 sensitivity calls, separate_error_control=True.
Runtime70 passes49 focused tests including nine real Pinocchio cases. Agreement
fails narrowly at1.02308673e-7 m, t0.85, WaistRBack axis1; limit remains1e-7.
Call35.969 s, launch40.103 s. Diagnostic71 is TERMINAL1: tighter augmented
rtol3e-11/atol3e-13 yields1.02308684e-7 m, effectively unchanged, in34.025 s.
Diagnostic72 is TERMINAL0: max_step0.0000625 with original70 tolerances yields
2.506897942e-8 m agreement,164339 calls/53.916 s sensitivity (71.554 s call/save).
Marker samples and state/marker sensitivity Jacobians are archived; primal q/qd
trajectories were not saved. This qualifies one candidate, not all derivatives.
Fit73 is TERMINAL0 after989.414 s, accepted/converged false, max_nfev10.
All ten Jacobian checks pass; final whole28.10547 mm, terminal65.39791 mm,
early10.86040 mm and club30.04069 mm. Settings: original19 baseline, B456,
grouped control, max_step0.0000625,200000 sensitivity calls.
Runtime73 qualifies62 tests plus runner help. Former PID2891462/handle55242
are terminal. Output /mnt/c/Users/diete/native-regularized-fit-9967-73. A separate
post-return replay saves actual q/qd (not qdd), markers and a rejected overlay.
Final candidate canonical SHA256:
786522cd5380a9f62602920b2cff6e6b404f7ceb99bfc6783f1f359b98a6457a. Four changed source modules pass mypy.
Marker agreement failures now report discrepancy, time, marker and axis.

The shared native motion sequence API is implemented and publicly exported.
It converts complete batches with immutable time/model/frame/branch metadata
and original per-sample references, reusing NativeJointStateAdapter. Root passes
101 related tests including12 new strict atomic file-I/O tests. Public
save_native_motion/load_native_motion preserve the validated sequence plus an
optional distinct raw-model hash. UI and complete dynamic frame transport remain;
Five new consumer checks pass, including actual MuJoCo all16 frames at three
samples; Drake/Pinocchio consumer tests skip locally for missing runtimes.
These are conversion checks, not alternate MuJoCo/Drake dynamics qualification. See dedicated
REPRESENTATION_HANDOFF for runnable usage and the remaining ordered plan.

## MuJoCo Spherical Export and Fitting Conditioning

The opt-in native_spherical_mjcf exporter now reuses canonical MJCF geometry and
shared group inventory to replace three XYZ triples with ball joints. Real local
MuJoCo3.3.4 compiles nq30/nv27; five tests verify unchanged inertias/closure/scalars
and all body/site poses across three manufactured states. This is kinematics
qualification, not a dynamics adapter or stock mj_step acceptance. Next gates:
tangent conventions, dual effort/convective acceleration, reused rigid-closure
solve, then same-input trajectories. Dedicated turnover:
../development/mujoco_native_matching/MUJOCO_MANIFOLD_HANDOFF.md.

Offline audit74 uses saved72 Jacobians, existing valid masks, amplitude10 and
terminal weight10. Marker-only23100-by-81 condition number is7.452e7; column
normalization leaves3.767e7. Penalty rows/bounds are excluded. This supports
checking correlated sensitivities, not claiming a proven cause of stagnation.
Fit73 ended with only modest matching improvement. Inspect
actual/predicted reduction and a mixed-control directional derivative before
more compute. See native_evidence/sensitivity_condition_9967_74.

## Fixed-Attachment Feasibility

Audit69 uses existing rigid-body relaxation on all654 frames. Three head markers
are attached to Hub; no independent native head joint exists. Fixed-offset lower
bounds are17.393 mm whole swing/24.279 mm terminal, even with independent body
poses and no dynamics/connectivity. Separating head as a hypothetical independent
six-DOF body lowers this to4.471/9.177 mm, but changes the model. No marker/model
changes were made. Current torque errors exceed the rigidity floor, so both
optimization and physical approximation matter. Preserve original25/35 mm gates;
never silently omit head markers. See native_evidence/rigidity_9967_69.

## Reproducible Compute Limits

`integrate_forward` and `integrate_sensitivities` now accept optional positive
integer max_evaluations, checked before excess derivative/linearization calls.
`replay_marker_sensitivities` exposes max_sensitivity_evaluations for the augmented
solve only; its independent primal replay remains separate. DefaultsNone retain
existing behavior. Seven new tests and native forwarding checks went RED/GREEN;
27 related tests, Ruff and pinned mypy pass. This source update is NOT in immutable
runtime61; qualify a new runtime before using it remotely. Budgets are not a
numerical accuracy or wall-time guarantee. Exhaustion raises without partial data.

## Why Earlier Attempts Stalled

- Static pose fitting and feedback tracking did not produce an open-loop sextic
  motion. Run45 feedback whole RMS35.2 mm becomes346.5 mm in time-only replay,
  and655.5 mm after global-sixth-order compression. Do not promote the initializer.
- An actual native gimbal singularity stopped run41. An independent inverse-angle
  branch bug caused run49 to apply a different native effort map. Branch metadata
  is now explicit and tested; true singular native actuator inverses remain rejected.
- Native rate errors are amplified near the shoulder chart. Run52 physical
  angular error0.000208 rad/s became native rate error0.00465 rad/s. Tight scalar
  reference54 and manifold tolerance refinement led to passed58. Gates were not
  relaxed. Run54/55 second-level budgets were insufficient even for their selected
  maximum step; those failures are not proof of solver nonconvergence.
- The Python manifold implementation remains slower than scalar Pinocchio. Keep
  scalar native dynamics available as the fitting baseline while qualifying
  alternate representations. Do not block fitting on all-engine alternate builders.

## Two-Window Preflight and Direct Chart Derivatives

Stage1 diagnostic75 is TERMINAL0. Window0 starts at original q0/qd0 and window1
at the saved uninterrupted73 sample216 at0.6 s. Against uninterrupted73, maximum
marker distance is5.369e-12 m, native q4.422e-11 and qd8.804e-9. First endpoint
minus saved node is9.77e-14. Full pose/rate closure and the42-dimensional q/v
node chart pass. Zero retraction shifts physical state by4.75e-12. Exact evidence:
native_evidence/two_window_preflight_9967_75. This is fixture qualification, not
an optimized segmented swing or a derivative pass.

MultipleShootingOptions now accepts window_jacobian_state_coordinates="node"
(default "physical"). Node mode takes theta followed by direct chart columns,
retaining physical endpoint rows and the negative next-node transform derivative
in continuity. No pseudoinverse extension is needed. Eleven new nonlinear tests
went RED/GREEN; root passes44 combined shooting/node tests and source mypy/Ruff.

Diagnostic76 is TERMINAL (former PID2923511/handle25121), on frozen runtime73. It checks
81 control columns in window0 and81+42 in window1, then at most16 signed
perturbation trials for control, node-position, node-velocity and mixed directions.
Marker, q, qd and scaled endpoint/continuity checks retain1e-3 derivative gates
with explicit weak-block reporting. Process exit0 is not scientific acceptance:
status is failed_derivative_gates. LS B6 h1e-4 and mixed h1e-6 pass all blocks;
smaller steps fail some resolved checks. Both node directions pass resolved blocks
but near-zero cross-continuity derivatives fail relative checks (analytic norms
about5e-14/2e-17 versus central estimates about1e-10/1e-11). Establish physical
absolute-error floors/structural-zero treatment before declaring full qualification;
do not widen C3D or replay gates. Call225.112 s, launch250.611 s, all16 trials
complete. No optimizer or other numerical job remains active. Evidence:
native_evidence/two_window_derivatives_9967_76.

## Derivative Resolution Floors (Audit 77, Terminal)

Local analysis of the archived run76 arrays already resolves the two failure
classes. (a) Node-direction cross-continuity blocks: the zero-retraction
Jacobian equals diag(scales)\*basis to1.6e-10 with scaled orthonormality defect
3.1e-15, so the velocity response of the pure-position singular direction and
the position response of the Dq null direction are structurally zero up to
that defect; the archived retracted nodes match the linear prediction to
1.2e-14 (h1e-5) and1.3e-12 (h1e-4), so the central estimates (1.7e-10/1.5e-11)
are retraction roundoff divided by2h, not derivative error. (b) LSInputX h1e-5
and mixed h1e-7 absolute errors are consistent with replay noise divided by
step (endpoint q about4e-10 to2e-9 per replay); the archived perturbed
coefficients reproduce the intended Bernstein increments to4.8e-10 (control)
and4.3e-7 (mixed h1e-7) relative, and effort evaluation error is at most
2.3e-6 relative to h, far below the observed4e-3 failures.

New shared providers: `motion_matching/derivative_resolution.py`
(central_difference_floor, classify_derivative_block with verdicts passed /
structural_zero / unresolved_at_step / failed, qualify_direction and the
orthonormality cross_block_norm_bound) and
`pinocchio/python/native_node_chart.py` (closure/Jacobian/basis/retraction
wrapper reused by drivers). Twenty new unit tests pass with Ruff and mypy.
A pass now additionally requires the measured floor to be below the gate times
the analytic norm; agreement inside noise is reported unresolved, never passed.

Audit77 is TERMINAL0 (202.6 s launch,26 of32 budgeted window replays) on
runtime77 (frozen73 file set refreshed from f96766c8e plus five new files;122
qualification tests). Measured per-block replay error by tolerance variation at
unchanged max_step: markers8.2e-10 m, endpoint q9.3e-10, qd1.6e-8. Loose and
working replays coincide while both differ from tight by that amount: the
integration is max-step limited and the non-reproducible part is step-sequence
roundoff/constraint-solver noise, so tolerance alone does not shrink it. Base
replays reproduce run76's primal arrays exactly. Every factor-one failure sits
at the smallest step with absolute error1.02x–1.14x the single-pair floor;
`reclassify.py` applies the tested `floor_safety_factor=2` to the same archived
arrays and every direction/block then has a resolved pass (LSInputX h1e-4;
mixed h1e-6 and h1e-5) or an orthonormality-verified structural zero, with no
unexplained failure. Gates are unchanged. Evidence and turnover:
native_evidence/two_window_floor_9967_77 (raw ZIP SHA2563d5768fc…).

MultipleShootingOptions now also accepts shared_boundary_policy="once"
(default "both"), which observes the capture sample shared by adjacent windows
only in the earlier window so marker rows, Jacobian rows, segmented RMS and
equality offsets match an uninterrupted single-window objective at zero
defect. Four RED/GREEN tests;46 combined shooting tests, mypy and Ruff pass.

The archived run73 sampled rotation-chart audit gives peak condition2.680/17.886/
2.889 for hip/left/right shoulder. These samples do not bound between-sample
extrema or explain the full7.45e7 trajectory-control Jacobian condition by
coordinate maps alone. See native_evidence/returned73_chart_audit.

## Ordered Next Work

Use [Next Agent Execution Plan](simscape_tour_matching/NEXT_AGENT_CONVERGENCE_EXECUTION.md)
for the existing-provider, two-window derivative preflight now that fit73 is terminal.
It specifies integrated nodes, retraction/continuity chain-rule checks and bounded
SLSQP acceptance without recreating the solver or accepting state-reset motion.

1. Run the bounded two-window direct-node SLSQP trial (Stage3): exact run73
   coefficients with zero increments, B4/B5/B6 subset, bounds derived from run73's
   saved parameters, integrated0.6 node with zero chart coordinates, node mode
   Jacobians, once-only shared boundary, effort penalty parity, small iteration
   and evaluation budgets, then original-state replay. Report projected rank,
   active bounds and predicted versus actual reduction before any horizon change.

2. Qualify the variant over further representative trajectories and against
   MATLAB R2025b before claiming full equivalence. Qualify tangent derivatives
   before using the manifold variant in an optimizer. Preserve existing scalar API.
3. Resume trajectory/control co-optimization using native constrained forward
   dynamics. Reuse native profile, sensitivity, marker and shooting providers.
   Start from integrated states, permit trajectory changes, and optimize a global
   degree-six native effort basis. Do not repeat tight static-node boxes or simply
   compress an arbitrary feedback torque trace. Existing seven-DOF bioptim tracking
   is a different physical model and cannot silently replace this native model.
4. Require uninterrupted original-state replay through1.813888889 s, all654 capture
   frames and valid observations, closure, effort audit and independent R2025b
   validation before accepting the final swing. Preserve per-marker/time residuals,
   overlay animation, coefficient/time-basis metadata and exact input hashes.
5. Implement/qualify remaining engine variants sequentially using canonical
   pose_interchange providers, without inferring dynamics parity from pose roundtrips.

## Engine and Representation Handoffs

- [Pinocchio Manifold](simscape_tour_matching/PINOCCHIO_MANIFOLD_HANDOFF.md): native
  spherical variant nq30/nv27, exact fixed transforms/inertia/weld reused. DOP853
  integrates local tangent coordinates and carries physical endpoints unchanged;
  no quaternion projection or target-state reset. Runtime18 passes96 tests with
  one optional MuJoCo skip; root independently verified generic and real Pin tests.
  Remote receipts qualify the numerical runtime bundle, not full GUI/application
  deployment; retain the documented package boundaries when reproducing.
- [Canonical Representations](simscape_tour_matching/REPRESENTATION_HANDOFF.md):
  angle/quaternion/rate/convective-acceleration/effort conversions and fixed-frame
  transport. Preserve units, named frames, branch/winding and model identity.
  Moving-frame acceleration transport and #8867 convention consolidation remain open.
- [MuJoCo](mujoco_native_matching/HANDOFF.md) and
  [Drake](drake_native_matching/HANDOFF.md): native adapters and narrower receipts
  exist; alternate quaternion builders/full trajectory equivalence remain open.
  Drake's projected accelerations are not derivative-consistent trajectories.
- OpenSim epic #10003: plan/handoff branch docs/10003-opensim-matching-epic,
  documented plan commit1a68091b6. Implementation/runtime qualification is not
  accepted. Inspect its branch and requested planning check-in before proceeding.

The documented development-log validator path is absent in this checkout; no
validator pass is claimed. Normal configured commit/push hooks still apply.

## Evidence and Reproduction

Evidence root: simscape_tour_matching/native_evidence. Preserve raw ZIP archives;
formatted JSON raw hashes can differ without changing parsed physics. Original
native model SHA256 b817fea76407f71b29e42aeb188890df53d2d86874c7e5875d14d353f4a1a248.
Run19 fixture canonical candidate SHA256
b5b1c3823c86a21df323dc4e430366dc093495b069b3d25c361b0d007b8ff24f.
It is a rejected0.85 s C3D fit (whole30.79 mm, terminal99.99 mm), not the best full swing.

ControlTower: ssh alias controltower; WSL ControlTower-Runner; Python
/home/dieterolson/simscape-pinocchio-9967/.venv/bin/python; runtime18 PYTHONPATH
/home/dieterolson/native-manifold-10043-18; use one BLAS/OpenMP thread for comparable
runs. Driver compare_native_manifold_replay.py records independent scalar settings.
Raw run receipts identify exact archived source and inputs. Never overwrite runs.

[Convergence Review](simscape_tour_matching/CONVERGENCE_REVIEW_20260912.md) gives
strategy and delegation gates. [Historical Handoff](HANDOFF_HISTORY_20260912.md)
preserves previous matching and unrelated incoming project context. Its ACTIVE
statements are historical, not current process evidence. Update this concise
handoff and DEVELOPMENT_LOG with each implementation commit; append history only
when needed. Do not reintroduce competing current-status sections.
