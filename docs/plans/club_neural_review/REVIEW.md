# Club-Only Matching and Neural Acceleration Review

## Recommendation

Proceed with both programs, using the existing physical models as the final authority.
Club-only fitting should return plausible, explicitly assumption-dependent candidate
swings, not claim to reconstruct an unobserved body. Neural acceleration is worth a
bounded pilot for repeated queries: first learn useful starting trajectories/controls
and refine them with the actual constrained solver. Predicting a coefficient vector
alone is not evidence that it produces the requested swing.

The desired end state includes a versioned trained model artifact for every supported
physical model identity, not merely a single Simscape network. Shared trainers, data
schemas and inference services remain common; checkpoints and qualification are
model-specific. A backend is not necessarily a different topology. Reuse weights across
equivalent engines only after native parity qualification. Missing runtimes or failed
training/benefit gates remain tracked per-model blockers, never silent scope reductions.

## Repository and Workbook Evidence

Reviewed local source at `c3111a9177885af945018d730ec40de308cd9971` on 2026-09-20.
Unrelated local edits were preserved. This is planning/review evidence, not a fresh
training or physical-match qualification. The earlier baseline epic is
[#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584); its model
identity work [#10585](https://github.com/D-sorganization/UpstreamDrift/issues/10585)
and contracts [#10587](https://github.com/D-sorganization/UpstreamDrift/issues/10587)
are dependencies. The main matched-swing program
[#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363), full-body
qualification [#10378](https://github.com/D-sorganization/UpstreamDrift/issues/10378),
portable jobs [#10379](https://github.com/D-sorganization/UpstreamDrift/issues/10379),
fast physical matching
[#10430](https://github.com/D-sorganization/UpstreamDrift/issues/10430) and strategy
integration [#10440](https://github.com/D-sorganization/UpstreamDrift/issues/10440)
retain ownership.

Read-only workbook inspection used openpyxl through the bundled Python runtime, reading
cached values and source labels without changing either workbook. The JSON audit
accompanies this document; CO-00 owns the production audit tooling. Physical figures
below explicitly apply the existing repository's cm-to-m interpretation; they are
independent sanity checks, not a new calibration authority.

| Workbook                                                                                                                             | SHA-256                                                            | Findings                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `data/Club_Data.xlsx`                                                                                                                | `5d9183e1d01ea7c6f9c162375dd6855c076ee26a96b76c90e59e9cf2679dde25` | Four distinct trial sheets, a duplicate Filtering Experiments sheet, and empty Sheet1 |
| `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Motion Capture Plotter/Wiffle_ProV1_club_3D_data.xlsx` | `88d3eb31541d886031f6c5ad7c82493b0e3ee4e3f73a8652372277cc14d02234` | Same numeric values in the four shared trial sheets                                   |

| Trial     | Numeric Samples | Native Time Range (s) | A / T / I / F Sample IDs | Median Grip-to-Face Distance (m, cm Interpretation) |
| --------- | --------------: | --------------------- | ------------------------ | --------------------------------------------------: |
| TW_wiffle |             882 | -2.158333 to 1.512500 | 240 / 412 / 519 / 832    |                                            1.069339 |
| TW_ProV1  |             775 | -2.183333 to 1.041667 | 240 / 418 / 525 / 725    |                                            1.069339 |
| GW_wiffle |             774 | -2.150000 to 1.070833 | 240 / 448 / 517 / 724    |                                            1.081854 |
| GW_ProV11 |             771 | -2.166667 to 1.041667 | 240 / 452 / 521 / 721    |                                            1.081854 |

Data occupy columns A:Z starting at row 4; row 1 holds events, row 2 identifies
Mid-hands and Center of club face, row 3 labels channels. All four tabs have worksheet
max_row 885, but padded rows are not samples. Native time increments are 1/240 s and
impact is zero on the supplied clock. TW_ProV1 and Filtering Experiments have identical
inspected numeric content; copies across workbooks must share a trial lineage and split
group.

The Definitions sheet states global X points away from the target, Y toward the ball and
Z up; local z runs from head toward grip. This conflicts with some prose in
`docs/motion_training/README.md`, which describes X toward target and local X along
shaft. Resolve a proper rigid transform explicitly, not by mixing descriptions.
Definitions says inches, while current MATLAB/Python loaders intentionally correct to
cm. The cm interpretation gives about 1.07–1.08 m grip-to-face distance and
finite-difference peak speeds about 50.9–52.9 m/s; inches would scale those by 2.54.
Save both the original declaration and the reviewed override with source evidence.

Only two orientation axes are populated. Columns L:N and X:Z are missing for every
numeric row. Their third axes may be derived from a reviewed orthonormalization/cross
product, with derivation metadata and degeneracy checks. Do not call them independently
measured. The sheet GW_wiffle has ProV1 in A1; retain the conflict, do not infer a
definitive ball label. CHS is a separate recorded impact-speed field, not automatically
equal to a numerical derivative peak. T is documented as a pelvis-rotation event
although no body trajectory is present; retain the event meaning and do not relabel it
as club top.

A direct call to the current `read_excel_event_markers` reproduces another defect:
TW_wiffle's `A=`, `T=`, `I=`, `F=` return NaN, while bare-letter events in the other
three sheets parse. The current canonical Excel loader uses peak-speed impact inference
and resampling, so the original event/clock decisions need to become explicit. The
Pinocchio parser uses cm and derives z from x/y, but expressions such as `value or 1.0`
can replace a valid zero component. These are concrete regression cases, not authority
to rewrite another active agent's work.

## Existing Software to Reuse

| Concern                  | Inspected Source                                                                                                        | Interpretation and Gap                                                                                                                                                                                            |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Club targets             | `src/shared/python/motion_matching/loaders/excel.py`, `club_target.py`, `target.py`                                     | Existing canonical loader/target; needs explicit masks, native events/frame provenance and both observed frames where required                                                                                    |
| Club-only body motion    | `src/engines/physics_engines/pinocchio/python/motion_training/`                                                         | Existing parser, dual-hand Pink/DLS IK, visualization and trajectory export; the word training here does not establish neural training or torque replay                                                           |
| Current physical fitting | `provider.py`, `kinematic_smoother.py`, `windowed_ik_refinement.py`, `contact_force_allocator.py` under motion_matching | Extend current abstractions and qualified contact/closure work; club-only assumptions need distinct acceptance                                                                                                    |
| Simscape ML work         | `src/engines/Simscape_Multibody_Models/3D_Golf_Model/MachineLearning/`                                                  | Existing club calibration, body-to-club and torque-to-club extraction, two-stage optimizer, smoothing, polynomial export and replay tools; current code/tests must qualify documented expectations                |
| Dataset generation       | `src/shared/python/data_io/dataset_generator/`                                                                          | Records q/v/controls and optional dynamics; zero-initialized optional arrays plus suppressed exceptions can preserve unavailable labels as zero; post-step finite-difference acceleration has an interval meaning |
| Dataset schemas          | `motion_matching/dataset/`, `src/shared/python/dataset_tools/load_compact.py`, `canonical.py`                           | Sweep and compact Parquet loaders; compact-1.0 is specifically 27 coordinates and 189 coefficients, including translations; not a generic all-model schema                                                        |
| Forward surrogate        | `motion_matching/surrogate/`, `surrogate/compact/`, `surrogate/perstep/`                                                | FiLM/MLP trajectories and instantaneous models already exist; some paths resample by array index, and perstep train splits random rows rather than trial groups                                                   |
| Inverse models           | `motion_matching/inverse/`, `inverse_timestep/`                                                                         | CVAE, deterministic regressor and q/v/a-to-tau MLP exist. Source notes report both trajectory-to-189 models plateauing near a mean-prediction baseline; this is not a successful model-family training campaign   |
| Hybrid verification      | `motion_matching/hybrid.py`, `surrogate/validate.py`                                                                    | Existing injected native polish/replay boundary; expand model support and physical validation, preserve real clocks                                                                                               |
| Training infrastructure  | `src/shared/python/training/`, `runtime/adapters/pytorch_cvae.py`, `src/tools/training_controller/`                     | Dataset registry, scheduler, process execution, metrics, persistence and GUI exist; CVAE-only runner selection must not dictate the new architecture                                                              |

Historical
[#4075](https://github.com/D-sorganization/UpstreamDrift/issues/4075)/#4076/#3999/#4000/#6014
and the closed PINN epic
[#5419](https://github.com/D-sorganization/UpstreamDrift/issues/5419) supply prior
designs, not new current speed/accuracy evidence. Documented old timings and
model-release assumptions are not acceptance gates. R2025b is required for new Simscape
acceptance, regardless of older runbooks. The documented
`C:/Users/diete/Repositories/data/TenThousandFiles.parquet` does not exist at that path
on this host. Tracked discovery found `data/sweep_synthetic/{trials,timesteps}.parquet`;
those are test fixtures. Other machines may hold real datasets/checkpoints, but none is
qualified by this review. First worker inventories exact hashes, retrieval locations,
generation provenance and current physical compatibility before reuse.

## Club-Only Method

Use an observation operator y(t)=H_model(q(t), geometry) that selects exactly the
measured position/orientation components. Minimize robust, uncertainty-scaled
measurement residuals subject to the real kinematics/dynamics, then regularize the
unobserved directions with explicit golf-pose, smoothness, contact, joint-range and
effort priors. Keep measured-fit quality and prior plausibility separate. Full body is
underdetermined even if both club frames are supplied; two points alone do not identify
shaft twist.

Candidate generation should combine a retrieval warm start from qualified reference
swings, model-appropriate constrained IK, and a small set of plausible posture/contact
branches. Use Jacobian null-space exploration only as a local proposal mechanism;
reproject and independently validate each full candidate. Score a Pareto set of club
fit, physical feasibility, body plausibility and runtime rather than hide all tradeoffs
in one scalar. A fixed-pivot pendulum cannot reproduce arbitrary spatial club
orientation; publish its attainable residual and compare only supported observables.

The standard fast path is retrieval -> calibrated initial state/geometry -> coarse
constrained fit -> sparse temporal refinement -> native continuous torque replay. A
longer best-fit path can spend a declared extra budget. Report distinct kinematic
preview and validated dynamic result. If prescribed hub motion is used, record it as an
external input with work; never conceal it as inferred whole-body actuation. Ball type
and CHS alone do not supply impact forces. Before/after-impact discontinuities need an
explicitly measured or modeled collision boundary; no invented impulse to make a
continuous swing match.

Novel project experiments are (1) feasibility-screened posture branches with
prior-sensitivity envelopes, (2) continuation from the actual reduced-model fit into a
higher-complexity model, and (3) mask-conditioned learned proposal diversity followed by
physical projection. These are hypotheses to benchmark, not claims of research novelty
or accuracy.

## Neural Formulation and Data Efficiency

Separate three supervised tasks. Forward dynamics: (q,v,u,geometry,contact state,dt
where applicable) -> acceleration or next state. Inverse dynamics:
(q,v,a,geometry,contact/actuation convention) -> feasible controls/reactions under a
specified allocation objective. Amortized matching: (masked target history,
timestamps/duration, q0/v0 or posture prior, geometry, model/profile) -> candidate
state/control trajectories or continuous-control coefficients. Club-only input does not
uniquely determine torque; a point regressor can average incompatible solutions. Train a
declared selected solution or a conditional proposal distribution, then check it
physically.

For floating/contact models enforce M(q)a+h(q,v)=B(q)u+J(q)^T lambda plus closure and
contact laws. Record passive terms, applied actuator inputs and generalized forces
separately; q dimension can differ from v and control dimension. Muscle
excitations/activation history cannot be replaced by net joint torque. Legacy
translational-force channels and polynomial A..G order/time domains must be mapped
explicitly; bounded coefficients do not imply bounded torque over a horizon.

Generate native rollouts near qualified swings and structured, bounded perturbations
before broad random sampling. Add diverse feasible teacher-optimized trajectories to
avoid learning only one posture/control allocation. Include failures in a separate
feasibility dataset with reasons. Acquire new samples where native residual, uncertainty
or coverage warrants them; an ensemble's disagreement is a heuristic, not proof of
calibration. Reuse immutable raw episodes; derive task-specific views with train-only
normalization and content-addressed caches. Split by source trial/seed family, geometry
and contact regime before slicing windows or augmentation. Workbook aliases and nearby
augmented trajectories must not straddle train/test.

Start with 100 native episodes for contract qualification, then nested 500 and 2,000
episode learning-curve stages for the first reduced model; use three training seeds and
measured compute/storage caps recorded before dispatch. These are experiment sizes, not
claims of sufficient accuracy. Scale full-body data only after the generation/benefit
review, using one checkpoint/model card per roster model. Train on both observable-rich
and club-only masks; four workbook trials and two C3Ds are external targets, not enough
independent empirical examples to support broad generalization. Synthetic body labels
are simulation truth under a chosen model, not measurements of the golfer.

Compare classical cold start, nearest-neighbor retrieval + physical solve, existing
neural baseline + solve, forward-surrogate inversion + native polish, and learned
trajectory/control proposals + polish. Small MLP/temporal models first. Revisit
CVAE/mixture proposals only with mode-collapse diagnostics; diffusion and
physics-structured dynamics are controlled follow-on ablations, not default
infrastructure rewrites.

## Evidence-Based Decision Gates

Freeze physical/observation thresholds with the current model-specific acceptance
authority before testing; do not reuse full-body gates as if body markers were observed.
Numerical synthetic invariants have exact/tolerance-based tests; workbook fit thresholds
require calibration uncertainty and attainable geometry evidence. Keep all existing
full-body G3 requirements unchanged.

For the pilot, pre-register at least 30 held-out synthetic query episodes per supported
test stratum, three seeds, all four workbook targets and both C3D-derived club-only
controls. The tiny empirical set limits conclusions; do not report population confidence
from it. Measure startup, preprocessing, candidate generation, neural inference,
physical refinement, rejected attempts and independent replay; report cold/warm CPU and
available GPU median/p95, native-call counts, memory, failure rate and quality-matched
success with uncertainty intervals.

Proposed performance promotion target: at least 2x median end-to-end speedup, no worse
p95, and no degradation in accepted physical-quality rate at frozen thresholds versus
the strongest classical/retrieval baseline. These are proposed experiment gates to
freeze in NM-01, not measured outcomes. If they fail, retain a trained research
checkpoint and the exact blocker; do not market acceleration. Report the actual
time/accuracy tradeoff instead of tuning the benchmark or hiding failures.

Compute break-even as offline generation + training + teacher-solve cost divided by
per-query savings (same units). If savings are nonpositive there is no break-even.
Report wall time and monetary/energy estimates separately, include retraining after
model changes, and compare projected usage without inventing a demand estimate. Do not
fund larger full-body training solely because inference is fast.

## Primary Research and Applicability

- [TOAST: Constraint-Informed Learning for Warm Starting Trajectory Optimization](https://arxiv.org/abs/2312.14336): learns optimizer starts using constraint-informed losses. Supports testing learned proposals plus physical refinement; its rover/spacecraft results do not establish golf speedups.
- [Learning to Warm-Start Fixed-Point Optimization Algorithms](https://www.jmlr.org/beta/papers/v25/23-1174.html): trains starts for downstream iterative solves. Supports measuring time to accepted output; its theoretical conditions are not automatically satisfied by nonlinear contact golf dynamics.
- [MaskedMimic](https://arxiv.org/abs/2409.14393): physics-based control from partial motion descriptions. Motivates explicit observation masks and priors, but golf-specific contact/club calibration and data are still needed.
- [Diffusion Policy](https://arxiv.org/abs/2303.04137): represents multimodal action sequences. Motivates an optional diversity ablation; sampling latency and data requirements may defeat this project's speed goal.
- [Deep Lagrangian Networks](https://arxiv.org/abs/1907.04490): uses mechanics structure in learned dynamics. A possible sample-efficiency comparison for smooth rigid models, not a reason to replace fast known analytical equations or assume contact/muscle validity.

The recommendation is an engineering inference from these primary sources and the repository audit, not a published demonstration of the proposed golf system.
