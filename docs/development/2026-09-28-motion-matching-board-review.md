# Motion Matching Product Review and Board Issue Proposals

## Decision Brief

**Recommendation:** fund a measured qualification program around the existing shared pipeline, with MuJoCo/MJX as the fast experimental lane, Pinocchio/Crocoddyl as the analytical/optimal-control lane, Drake as an independent native dynamics check, and MATLAB R2025b GS3DX as the improved Simscape product. Preserve all four; do not equate a successful import, attractive animation, inverse-kinematics fit, or solver exit with a qualified dynamic swing.

The recent MATLAB work is substantial. The open GS3DX branch now includes quaternion shoulders/hip, capture-derived proportions and grip, leg control and ground contact, improved inertia, ellipsoid anatomy, a motion-driven neck, and articulated feet. Its latest Human variant reports **942 compiled blocks**, leaving **58 to the Home limit of 1,000**, or **33 after the existing 25-block validation reserve**. Earlier 751/967/973/975 figures describe different variants or counting states, not this latest model.

There is **no demonstrated, fully qualified dual-club, full-swing, cross-engine product** in the evidence reviewed. There are good partial fits, valuable controls and infrastructure, and several misleading historical summaries. Historical-video reconstruction remains a research/integration program: manual evidence review exists, but the public Shadow Tracker service deliberately refuses automated fitting and its model segmentation provider has no real inference implementation.

“Perfectly matched” should mean the best physically admissible reconstruction within measured uncertainty, with transparent residuals. Zero skin-marker error is not an appropriate release requirement: rigid bodies, soft-tissue motion, missing observations, joint-centre proxies, and averaging can make it unattainable. Neither extra optimizer iterations nor better-looking meshes resolve identifiability.

## Review Scope and Evidence Rules

- Main snapshot: `94ade65293` (full revision recorded in Git); Tools gitlink and initialized provider: `3678409fc51024150ab28970b72e3b468935f345`.
- Recent model work: [PR #10963](https://github.com/D-sorganization/UpstreamDrift/pull/10963), reviewed at **`752a94fdd9444f98b5e6f9a39b6e1638ffdb269e`**, still open at inspection. Branch findings below are not claims about merged main. Do not overwrite or claim the active #10979 work.
- Sources inspected: native JSON receipts, replay producers, acceptance evaluator, model branch documentation/builders' interfaces, Shadow Tracker service/segmentation/rendering, current issue/PR titles, model coverage and qualification reports. Existing images were inspected; no new MATLAB qualification, native fit campaign or physical experiment was run for this review. The focused test run did exercise existing MuJoCo calibration checks.
- Main-relative paths in this document are relative to the repository root unless they are clickable relative links. Branch references use immutable GitHub commit links.
- Evidence strength: **receipt** = committed numerical record; **branch report** = author's reported native experiment, independently rerun still required; **source** = implemented behavior; **proposal** = work or acceptance to approve. These are deliberately separate.
- This is a board packet of draft issue bodies, not newly filed implementation issues, an execution dispatch, or approval to close existing issues. It complements [the broader product review](2026-09-28-product-review-board-proposals.md), particularly R05/R06/R12, rather than replacing it.

## What Is Matched Today

### Comparable Numbers and Their Limits

All distances below are millimetres unless stated. “RMS median” is the median of frame RMS values; it is **not** the pooled marker RMSE. Marker sets, topology, time horizon, calibration, and drive mode differ across rows; this is not a global leaderboard.

<!-- prettier-ignore -->
| Model / Lane | Best Useful Evidence Found | Remaining Gap / Interpretation |
| --- | --- | --- |
| Simscape legacy run-102, driver | R2025b independent cold replay: 0.85 s, 307 samples; whole **20.267**, early **9.995**, terminal **40.301**, club cluster **8.389**; maximum marker-distance discrepancy against analytical replay **0.0605** | Terminal exceeds 35 by **5.301** (15.1% above ceiling; needs 13.2% reduction). Not through-impact/full-swing; cluster error is not independently calibrated clubface/impact accuracy. Run-103 gate is explicitly blocked. |
| New GS3DX fitted-grip IK, driver | Branch `FIT.md`: 654 frames, **7.0 RMS median / 16.1 p95 / 19.1 max per frame**, 43 min; held-out downswing first-pass wrist medians **18/15**, clubhead-proxy median **29** | Strong kinematic advance. Whole-trial offsets are calibrated in sample; held-out results use another fit protocol. Not pooled full-marker dynamic error. Smoothed IK has different statistics (6.3 median, 25.2 p95, 26.3 max). |
| GS3DX Shape through impact | Branch `SHAPE.md`, saved joint-centre-trunk reference: at 1.319 s, pelvis **22 RMS / 44 end**, horizontal COM **14.4 RMS / 19.7 max**, foot slip **38/16**, support **0.36–1.83 BW**, late peak **1.77 BW** | Progress in balance, not full-marker qualification. No force plates in this C3D; inferred GRF is not independent validation. Whole swing/iron and foot/contact criteria remain. |
| MuJoCo calibrated baseline, driver / iron | Canonical receipts: address **7.9 / 6.6**, full-capture IK **34.1 / 31.6**, dynamics **84.5 / 88.6** | Zero IK RoM flags in current calibrated reference; dynamics still imperfect. Older 27.3/28.6 IK claims used widened leg bounds and must not be current acceptance evidence. |
| MuJoCo shared simulator with MJX L-BFGS proposals | Latest 60-iteration benchmark: dynamic replay **40.8 driver / 42.6 iron**; worst segment **53.7 club / 50.1 left arm**; downswing minimum support **0.31 / 0.29** | Best lower-error full-body optimization lane found on main, but both stop at `max_iterations`. Runtime **6,217 / 5,097 s**, versus baseline **1,113 / 890 s**. Default remains `none`; optional better accuracy is not convergence or release. |
| Pinocchio/Crocoddyl 44-coordinate dynamic lane | Same-integrator RK45 0.85 s attempt: **46.8 whole / 25.7 early / 64.3 terminal / 52.6 club**, yaw **9.3°**; short 0.30 s verified window **19.7 whole** | 0.85 s rejected on tracking, yaw, 17.2 penetration and zero minimum support. Whole G1 gap is **21.8**, terminal **29.3**. Short-window success cannot stand for G1. |
| Pinocchio fast inverse-dynamics lane | Turnover records driver **133.5 whole / 50.2 club**, iron **336.9 whole**, about **8 s/swing** | Useful fast kinetic analysis/seeding, not forward matching. Iron reused driver attachments; native control replay can diverge. Refit by capture. |
| Drake returned81 | Receipt: **26.365 whole / 11.426 early / 46.305 terminal / 15.953 club**, yaw error **13.923%** | Producer evaluates FK on supplied states across 307 frames; not a new independent torque-driven full-body fit. Receipt itself has `kinematic_parity_passed=false`. Whole gap **1.365**, terminal **11.305** do not describe native dynamics readiness. |
| OpenSim | Native model/IK `.mot`, viewer and adapter work exist; bootstrap nightly evidence exists | No comparable accepted dual-club full-swing muscle-driven receipt established. Viewer receipt explicitly says kinematic playback. Finish Moco/excitation replay and muscular/force qualification. |
| MyoSuite | Model/interface and nightly lane scaffolding | Reviewed nightly receipt has unavailable inventory with a probe SyntaxError and zero collected tests. This does not prove the environment cannot work, and certainly does not prove a native match. Refresh on the real host. |
| Driven double pendulum, driver / iron | Current TB-04 receipts: pooled whole-marker RMSE **567.1 / 425.1** | `scientific_qualification=disqualified`, `product_promotion=exploratory`, `solver_convergence=max_iterations`. Historical “<15 mm qualified” prose is contradicted by current files. |
| Driven triple pendulum, driver / iron | Current TB-05 receipts: pooled whole-marker RMSE **489.6 / 414.8** | Same disqualified/exploratory status. 62 valid observations over two landmarks; not full-body capture coverage. |
| Constrained planar upper-body golfer | TB-06 normal-error lower bound **110.4 driver / 112.7 iron**, versus 55 ceiling | Planarity preflight rejects before a fit. More tuning cannot remove the spatial model deficiency. |
| Reference human / simple humanoid / native URDF and reconstruction variants | Coverage/identity registrations and comparison paths | Inventory entries are not unique qualified fits; require per-variant receipts. No accepted result should be inferred from aliasing or display availability. |
| Club-only Excel roster | Four unique trials × 20 models; reproduction guide reports **0 scored / 80 unresolved** | Software contracts exist. Inferred body motion is not measured body motion; club-only acceptance cannot satisfy full-body G3. |

### Main Evidence Index

1. [Simscape Cold Replay JSON](simscape_tour_matching/native_evidence/two_window_fit_9967_102/qualified_candidate_replay.json), fields `metrics`, `gates`, `duration_s`, `cross_engine_parity`; [Blocked Run-103](simscape_tour_matching/native_evidence/two_window_fit_9967_103/native_gate.json).
2. [Canonical MuJoCo Runs](full_body_models/evidence/ground_support/CANONICAL_RUN.md); numerical authority is `address.calibrated.marker_rms_m`, `ik.marker_rms_m`, `dynamics.marker_rms_m` in the linked `anthro_driver_seeds` and `anthro_iron_seeds_zmp` receipts.
3. [Latest MJX Benchmark](full_body_models/evidence/mjx_benchmark_lbfgs60/REPORT.md), with per-run receipts and explicit incumbent reuse provenance.
4. [Pinocchio Turnover](matched_swing_program/MS31_PINOCCHIO_CROCODDYL_TURNOVER.md); [Crocoddyl Candidate Evidence](../../evidence/matched/driver_g1_crocoddyl_rk45_b100/); [Drake Receipt](full_body_models/evidence/replays/drake_receipt.json). `generate_drake_replay.py` in that directory documents the FK-only evaluation.
5. [TB-04 Driver](../plans/tour_baselines/evidence/tb04_driver_qualification_receipt.json), [TB-04 Iron](../plans/tour_baselines/evidence/tb04_iron_qualification_receipt.json), [TB-05 Driver](../plans/tour_baselines/evidence/tb05_driver_qualification_receipt.json), [TB-05 Iron](../plans/tour_baselines/evidence/tb05_iron_qualification_receipt.json), fields `metrics.whole_marker_rmse_m`, `statuses`. [Planarity Driver](../plans/tour_baselines/evidence/tb06_driver_planarity_receipt.json) and [Iron](../plans/tour_baselines/evidence/tb06_iron_planarity_receipt.json).
6. [OpenSim/Viewer Evidence](full_body_models/evidence/viewer/receipt.json), [MyoSuite Nightly](matched_swing_program/evidence/nightly/myosuite_receipt.json), [Club-Only Reproduction](../plans/club_only_matching/REPRODUCTION_GUIDE.md), [Model Coverage](../plans/tour_baselines/coverage_matrix.md).

### What the Best Swing Looks Like

There is no single scientifically accepted “best swing” to show. The closest verified **short dynamic** match is the Simscape run-102 evidence above. For the most developed **anatomical appearance**, inspect the new Human variant; for current **full-body replay optimization**, inspect MJX L-BFGS with its failure badge visible.

The inspected Human address image has separate ellipsoid chest/pelvis/limbs, shaped head and hands, a recognizable driver head, face-normal pointer, and articulated feet. Shape's down-the-line impact view shows a recognizable bent golf posture but still has an angular trunk and simplified head/club. These are model poses, not marker-overlay proof. The Human documentation reports its head centre differs from the head-marker centroid by **83.9 mm 3D RMS / 125.4 max**; those are different anatomical points, so fix attachments and definitions before calling this a physical head-fit error. The neck drops axial yaw: **44.1° orientation RMS** remains against full captured head rotation. Better meshes would make that omission visible.

- [Human Address Image, Pinned Branch](https://github.com/D-sorganization/UpstreamDrift/blob/752a94fdd9444f98b5e6f9a39b6e1638ffdb269e/src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/docs/screenshots/GS3DX_Human_fo_addr.png)
- [Shape Impact Image, Pinned Branch](https://github.com/D-sorganization/UpstreamDrift/blob/752a94fdd9444f98b5e6f9a39b6e1638ffdb269e/src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/docs/screenshots/GS3DX_Shape_dtl_imp.png)
- [Historical Returned81 Playback](full_body_models/evidence/visual_layer/returned81_playback.gif): useful orientation aid, not the latest Human or accepted dynamics.

The board demonstration should show synchronized measured marker dots, fitted rigid surfaces, residual vectors, address/top/impact/follow-through, front/side/top cameras, and a residual timeline. A smooth movie without the observations is insufficient.

## MATLAB Options and Product Architecture

### Preserve and Qualify the New Work

Pinned branch authorities: [GS3DX Documentation at the Reviewed SHA](https://github.com/D-sorganization/UpstreamDrift/tree/752a94fdd9444f98b5e6f9a39b6e1638ffdb269e/src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/docs). Read `FIT.md`, `SHAPE.md`, `NECK.md`, `HUMAN.md` and `BLOCK_BUDGET_FINDINGS.md` for the reported experiments.

The branch addresses singularities, mass distribution, grip, legs/ground and neck motion. Qualify and integrate these improvements rather than recreating them:

- The 80 kg subject mass is an assumption; the roughly 80.393 kg system includes the club. Capture kinematics do not identify absolute mass or validate muscle forces.
- The original saved drive was ill-conditioned; equivalent variants were checked on a stable impact drive. That equivalence is useful but does not qualify the original drive or full capture.
- Motion-driven neck angles and reference-following leg/COM controls must be disclosed. A model using measured motion to actuate a coordinate is not an entirely torque-driven prediction of that coordinate.
- Removing the Inertia Sensor subsystem saves 75 compiled blocks in Human. Before product promotion, prove no required diagnostic/export consumer loses observability. Preserve an instrumented qualification variant and measured count.

### Better Anatomical Shapes Within the Block Budget

**Yes, likely**, without multiplying articulated bodies. The existing Human ellipsoids are the lowest-risk base. Replace a segment's graphical geometry with a single low-polygon anatomical mesh, keep its attachment frames and independently specified mass/COM/inertia, and use simple collision proxies. MathWorks' [R2025b File Solid](https://www.mathworks.com/help/releases/R2025b/sm/ref/filesolid.html) supports imported rigid geometry; detailed geometry need not mean a joint or block per triangle. Confirm the actual compiled delta for every proposed substitution.

<!-- prettier-ignore -->
| Option | Benefit | Cost / Decision |
| --- | --- | --- |
| Refine existing ellipsoids and frame-bearing solids | Fast, stable, easy sizing; already implemented substantially | First product tier; inspect shoulders, trunk continuity, feet and grip at all key poses. |
| One anatomical File Solid per segment | Recognizable anatomy and silhouette without separate bones/muscles in the solver | Prototype head, trunk, hand and shoe first. Preserve custom inertia and units; audit mesh asset rights and attachment frames. Unknown compiled delta until measured. |
| External skin/mesh rendering driven by saved body transforms | Rich visual quality with no extra Simulink dynamics blocks | Reuse existing viewer/mesh tooling; version transforms and test pose parity. Surface deformation is a visual assumption, not contact geometry. |
| Separate bone/muscle bodies or detailed deformable tissue | Potential biomechanics detail | Poor first choice under the cap; requires new parameters, observability and validation. Keep a future higher-fidelity profile or OpenSim/MyoSuite lane. |

The [MathWorks Home answer](https://www.mathworks.com/matlabcentral/answers/492781-what-will-the-home-license-of-matlab-with-simulink-add-on-allow) states 1,000 nonvirtual blocks including referenced models. More importantly for this project, the GS3DX branch reports a native 1,000-pass/1,001-fail boundary probe and compiled counts. Subsystems/model references are not a loophole. A visual-only solid was measured at **seven compiled blocks** in Shape: do not assume one diagram icon costs one block. Mesh vertices affect rendering/memory, even if they do not add dynamics blocks. Version-specific local compilation remains the acceptance authority.

### Efficient and Flexible Fitting

Use one capture/marker/frame identity and one model-parameter document, with engine adapters. Freeze subject geometry and marker attachments after a calibration split, then fit initial state and controls; evaluate held-out phases without refitting. Keep model/marker uncertainty separate from optimizer convergence. Persist a portable run package: capture and asset hashes, source/provider revisions, coordinate map, attachment set, drive mode, q0/v0, control coefficients, solver/tolerances, controller gains, losses, residuals, failures and timings.

For MATLAB, optimize tunable control parameters in a reusable session, benchmark Fast Restart against cold simulation, and invalidate on geometry/topology/initial-state changes. MathWorks documents [scripted Fast Restart](https://www.mathworks.com/help/simulink/ug/write-matlab-scripts-to-run-fast-restart-simulations.html) and [Simscape limitations](https://www.mathworks.com/help/simscape/ug/limitations.html), including operating-point restrictions. Do not assume every parameter is tunable or an accelerator is licensed. Always qualify winners with a fresh R2025b cold replay. Use Pinocchio/MJX as proposal generators only after measured native parity, not as replacement evidence for Simscape.

The latest MJX lane halves baseline error but costs roughly 1.4–1.7 hours per capture. Profile compile, IK, gradients, dynamics, contact and rendering separately; choose a Pareto frontier of accuracy, convergence and cost. More iterations alone are not an efficiency strategy. Preserve warm starts and checkpoints, cache by all physical inputs, bound workers and cancellation, and keep deterministic CPU fallback.

## Shadow Matching and Historical Footage

Do not confuse three different systems: **Shadow Tracker** reconstructs motion from visual observations; the **physics-informed ShadowModel** observes inverse-dynamics residuals/torques; **markerless multi-camera capture** has its own reconstruction/calibration program. The recent PINN shadow observer does not turn historical video into motion capture.

Source truth on reviewed main:

- `src/shared/python/shadow_tracker/ingestion.py`, `source_records.py`, `mask_records.py`, `service.py` and `artifacts.py` provide evidence identity, frame/mask review, revisions and persistence. There is a launcher GUI at `src/tools/shadow_tracker/gui.py`.
- `ModelSegmentationProvider.segment` in `segmentation.py` explicitly raises until real inference is integrated, even when a checkpoint file exists. Manual masks are the usable path.
- `DefaultShadowTrackerService.fit` raises `UnavailableBackendError`; it is not wired to a qualified automatic fit.
- A real articulated reference renderer now exists; its source declares **37 fields**, despite older turnover prose mentioning 27. It uses shared FK and capsule/ellipse-style rendering, not a qualified subject-specific mesh silhouette. Canonical/native state mapping needs integration tests, not tuple-length inference.
- `fitting.py` has bounded optimization, physical terms and replay-audit contracts. Existence and synthetic tests do not establish historical-video accuracy or public-service integration.
- [Qualification Findings](../plans/shadow_tracker/QUALIFICATION_FINDINGS.md) reports a 25 ms probe with grip translation **13.7 mm** and rotation **1.657 rad**, versus **5 mm / 0.05 rad** limits. The old coordinate-order bug was fixed; this particular model/probe remains unqualified. Do not generalize its 138.1 mm IK error to the newer MuJoCo/GS3DX models.

**Distance to product:** the manual import/review/persist slice is the nearest useful deliverable. Automatic historical-video-to-3D requires several dependent milestones: real body/club masks, camera/time/scale estimation, compatible articulated initialization, physically admissible continuous fitting, held-out validation and uncertainty/abstention, then packaged UI/export integration. A credible schedule requires a bounded pilot; a percent-complete claim would be misleading.

Single-view archives hide depth, scale and occluded joints; camera motion, cuts, variable playback speed, interlacing, motion blur and a thin fast club worsen ambiguity. Produce multiple plausible reconstructions when necessary and label measured 2D observations versus inferred 3D and model-dependent torques. Never present inferred forces as force-plate measurements. Modern synchronized video+C3D can test accuracy; uncalibrated archives alone cannot.

For automated masks, benchmark a pinned provider rather than promising a model name will solve capture. [SAM 2](https://ai.meta.com/research/sam2/) and [SAM 3](https://ai.meta.com/research/sam3/) are plausible segmentation candidates, not 3D biomechanics solvers. Assess body and club separately, including blurred/occluded frames, asset terms and real hardware. Reuse camera/markerless boundaries under #9619 rather than building a second reconstruction convention. Its deferred physical validation remains pending Board authorization and resources.

## Draft GitHub Issue Bodies

MMR IDs are local proposals. Suggested labels: `tier:strong` and existing domain labels. Owners are roles, not agent assignments. Effort: M = integration; L = native research/qualification. Before execution, reconcile existing work, check/post an issue lease and name an acceptance owner. Threshold changes require separately reviewed evidence.

### MMR-01 — Generate One Truthful Matching Status and Comparison Ledger

**Priority / Owner / Size:** P0; matching platform; M. **Disposition:** extend #10363/#10380 and reconcile product-review R06/R12; no duplicate epic. **Depends on:** none.

**Problem and Evidence:** TB-04/05 receipts are disqualified at 415–567 mm while handoff/final acceptance prose says qualified below 15 mm. Matched-swing documentation also mixes 30 mm IK milestones with 25/12/35 mm dynamic G1. Closed work packages are being mistaken for scientific completion.

**Scope:** generate dashboards, baseline cards and report tables from versioned receipt/evaluator output. Include topology, capture, marker set, horizon, kinematic/forward/controller mode, convergence, physical acceptance, runtime and source freshness. Retain historical records and corrections.

**Acceptance / Verification:** [ ] the current four pendulum receipts render exploratory/disqualified everywhere; [ ] FK-only Drake and short-window Pinocchio cannot acquire full-dynamics badges; [ ] missing/changed hashes and stale variants fail closed; [ ] main and PR candidates display separately; [ ] tests inject contradictory status strings and prove numerical/physical verdict wins; [ ] generated docs pass freshness checks.

**Deliverable:** canonical schema migration, generated comparison and negative fixtures. **Out of Scope:** changing numerical thresholds to turn rows green.

### MMR-02 — Freeze Dual-Club Observation, Calibration and Accuracy Contracts

**Priority / Owner / Size:** P0; biomechanics + matching; L. **Disposition:** child/follow-up of #10375/#10382, coordinate with #11007 phase authority. **Depends on:** MMR-01.

**Problem and Evidence:** C3D skin markers, synthetic joint centres, club clusters and head-centre proxies are used in different objectives. GS3DX has missing shoulder observations and in-sample offset calibration; no recorded events/force plates in the audited driver.

**Scope:** version driver/iron maps, valid residual masks, units, handedness, capture/world transforms, event provenance, static/backswing calibration versus held-out downswing/follow-through, and clubface calibration. Store excluded markers and reasons. Define pooled RMSE, frame distribution, p95/max and per-phase/segment values distinctly.

**Acceptance / Verification:** [ ] reproduce 654 driver frames at 360 Hz and the source hash; [ ] independently audit iron rather than copying driver geometry; [ ] no gap-filled sample becomes measured evidence; [ ] held-out observations never update offsets; [ ] common-target metrics permit comparison while coverage differences remain visible; [ ] measurement uncertainty and justified model floor are documented before optimizer tuning.

**Deliverable:** frozen benchmark manifests, capture audit and independent metric tests. **Risk:** tour averaging may be physically inconsistent; quantify rather than suppress it.

### MMR-03 — Promote GS3DX Variants With Reproducible R2025b Evidence

**Priority / Owner / Size:** P0; MATLAB model owner; M. **Disposition:** review/extend active #10950/#10979 and PR #10963, not duplicate their implementation. **Depends on:** MMR-01/02 and owner-approved branch integration.

**Problem and Evidence:** the best anatomy/control work is on an open branch; docs contain variant-specific counts and experiments not yet a portable consumer release.

**Scope:** inventory Baseline/Slim/Quat/FullBody/Contact/Golfer/Fit/Shape/Neck/Human, exact builders/SLX/assets, capture/calibration, drive modes and outputs. Save machine-readable per-variant receipts and cold-replay commands. Keep hand-built originals protected.

**Acceptance / Verification:** [ ] clean-host explicit R2025b build/save/reopen succeeds; [ ] originals' hashes unchanged; [ ] reported MATLAB suite rerun at promoted SHA with skipped tests disclosed; [ ] stable-drive equivalence is distinguished from C3D fit; [ ] motion-prescribed neck and servo/feedforward contributions are exposed; [ ] main ledger consumes only the reviewed variant; [ ] image and metric packages carry matching candidate hashes.

**Deliverable:** promotable native bundle and regression report. **Out of Scope:** declaring the branch full-swing qualified because it compiles or stands from rest.

### MMR-04 — Enforce Compiled Home Budgets and Preserve Diagnostics

**Priority / Owner / Size:** P0; MATLAB infrastructure; M. **Disposition:** extend #10953 and #10979. **Depends on:** MMR-03 inventory.

**Problem and Evidence:** diagram counts understated compiled use; Human saves 75 blocks by dropping unused inertia sensors and reports 942 compiled. Shape measured an added visual solid at seven compiled blocks.

**Scope:** count compiled nonvirtual blocks including references for production and instrumented variants; reserve 25 blocks as an explicit project policy, independently of the 1,000 license ceiling. Audit sensor consumers and replacement offline diagnostics.

**Acceptance / Verification:** [ ] actual R2025b production count ≤975 with instrumentation reserve documented; [ ] exact required audit variant compiles ≤1,000; [ ] tests fail on a deliberately over-budget clone; [ ] no consumer loses mass/COM/energy/contact/closure evidence; [ ] report before/after count by subsystem, not an inferred icon count; [ ] tool/license errors have actionable diagnostics.

**Deliverable:** compiled budget gate and variant manifest. **Risk:** future topology additions consume reserve; remeasure rather than assuming 942 remains valid.

### MMR-05 — Add Anatomical Meshes Without Changing Qualified Physics

**Priority / Owner / Size:** P1; model geometry + rendering; M. **Disposition:** new bounded child of #10950; reuse club mesh epic #10120. **Depends on:** MMR-03/04.

**Problem and Evidence:** ellipsoids improve Human, but head/trunk/hands/shoes remain approximate. More separate solids could exhaust the Home budget.

**Scope:** prototype one-segment File Solid substitutions and external render skins; select by visual quality, compiled count, load/render time and asset rights. Keep mass/COM/inertia, joint frames and collision proxies separate. Provide low/high detail assets and graceful fallback.

**Acceptance / Verification:** [ ] pelvis/trunk/head/hand/shoe prototypes render at address/top/impact/follow-through with marker overlay; [ ] same-state FK and inertia audit unchanged for visual-only changes; [ ] pre/post native replay matches existing regression tolerances; [ ] compiled count remains within MMR-04; [ ] units, orientation, mesh provenance and redistribution terms retained; [ ] real UI visual review covers clipping, self-intersection and left/right handedness.

**Deliverable:** one approved asset strategy and reusable segment adapter. **Out of Scope:** claiming a more attractive silhouette proves subject-specific anatomy.

### MMR-06 — Resolve Head, Trunk and Grip Observability Before Adding Controls

**Priority / Owner / Size:** P1; biomechanics + MATLAB; L. **Disposition:** extend #10979/#10382. **Depends on:** MMR-02/03/04.

**Problem and Evidence:** neck drops axial yaw; head orientation remains about 44° RMS. Head centre versus marker centroid and C7-based trunk COM are inconsistent observables. Improved grip fitting demonstrates why geometry matters.

**Scope:** rigid head-marker attachments and full orientation; compare two-axis versus three-axis neck in a budgeted spike; joint-centre trunk frame/COM; subject-calibrated hand/grip and clubface transforms. Separate motion-driven reference quality from torque-driven reconstruction.

**Acceptance / Verification:** [ ] fit all three head markers with fixed offsets on held-out phases; [ ] report full orientation error and any unobservable rotation; [ ] no head marker exclusion to pass terminal G1; [ ] positive-definite inertia and mass conservation survive changes; [ ] held-out wrists and actual face orientation improve without hiding offset growth; [ ] compiled native regression and full marker budget pass or remain explicitly rejected.

**Deliverable:** model-selection evidence and qualified attachment map. **Risk:** a surface-centroid error is not a head-centre error; do not optimize the wrong target.

### MMR-07 — Complete Simscape Continuous Dual-Club Dynamic Qualification

**Priority / Owner / Size:** P1; MATLAB dynamics; L. **Disposition:** continuation of #10979 and matched-swing #10363; reference historical #10440. **Depends on:** MMR-02/03/06.

**Problem and Evidence:** run-102 misses terminal G1; new Shape improves pelvis/COM but still slips and has no complete marker/force acceptance package.

**Scope:** freeze calibrated geometry, fit feasible q0/v0 and bounded controls, continue G1 → impact → full swing for both captures. Declare controllers, gains, root actuation, prescribed coordinates, torque limits, contact law and regularization. Compare pure torque replay and supported tracking as distinct product modes.

**Acceptance / Verification:** [ ] cold uninterrupted R2025b replay from saved initial conditions, no measured-state resets; [ ] full duration and all required marker groups; [ ] unchanged acceptance evaluator thresholds for applicable profile; [ ] finite real solver clocks and joint logs; [ ] contact, penetration, slip, closure, RoM, effort and energy audits; [ ] failure diagnostics and best rejected checkpoints retained; [ ] tighter-tolerance sensitivity and independent rerun supplied for both clubs.

**Deliverable:** native per-club run packages and explicit G1/G2/G3 verdicts. **External Dependency:** measured kinetics require a separate force-plate dataset; inferred C3D GRF cannot close that validation.

### MMR-08 — Qualify MJX Accuracy, Convergence and Cost Before Promotion

**Priority / Owner / Size:** P1; MuJoCo/MJX optimization; L. **Disposition:** extend open #11006, reuse #11058/#11071 benchmark. **Depends on:** MMR-01/02.

**Problem and Evidence:** 40.8/42.6 mm replay is better than 84.5/88.6 but takes 5,097–6,217 s and stops on iteration budget.

**Scope:** profile gradients/compile/contact, inspect scaling and termination, optimize bounded residuals and continuation, compare equal accuracy/equal wall-clock budgets with existing shooting and `none`. Preserve per-capture provenance for reused baselines.

**Acceptance / Verification:** [ ] convergence reason is numerical, not relabeled budget exhaustion; [ ] native shared-simulator rescoring and holdouts; [ ] both captures satisfy promotion contract and physical gates; [ ] at least three seeded repeats report median/p95 cost and failures on named hardware; [ ] cancellation/checkpoint/resume does not alter candidate identity; [ ] no default change until all promotion gates pass.

**Deliverable:** reproducible Pareto benchmark and explicit promotion decision. **Out of Scope:** a promised GPU speedup without measurement.

### MMR-09 — Finish Pinocchio Native Fitting and Cross-Engine Torque Transfer

**Priority / Owner / Size:** P1; Pinocchio/Crocoddyl; L. **Disposition:** extend #10430/#10363, reconcile MS-107/MS-111 history. **Depends on:** MMR-02.

**Problem and Evidence:** same-integrator 0.85 s candidate is rejected; older mixed-tolerance replay diverged, and fast iron analysis reused driver attachments.

**Scope:** capture-specific calibration, manifold q/v maps, armature/contact/interpolation identity, warm-start continuation and bounded effort. Preserve fast inverse dynamics as its own product mode. Carry residual/root assistance and controller settings into exports.

**Acceptance / Verification:** [ ] independent saved-state RK45 or declared fixed-step replay; [ ] G1 passed before G2/G3 promotion; [ ] force/closure/RoM checks are not just solver convergence; [ ] iron does not use driver calibration; [ ] controls imported into MuJoCo/Simscape reproduce declared tolerances or reject with diagnostic; [ ] energy/virtual-work and velocity mapping tests cover quaternion/scalar joints.

**Deliverable:** dual-club native candidate receipts plus transfer matrix. **Risk:** open-loop instability needs an explicit tracking profile, not hidden assistance.

### MMR-10 — Establish Genuine Drake, OpenSim and MyoSuite Native Baselines

**Priority / Owner / Size:** P1; engine adapter owners; L, split into three bounded implementation issues. **Disposition:** children of #10363, reuse #10346 for MyoSuite. **Depends on:** MMR-01/02 and model inventory.

**Problem and Evidence:** Drake FK parity, OpenSim motion playback and zero-test nightly probes are not native dynamic fits.

**Scope:** for each engine/club/variant, declare supported topology/actuation, run a genuine native rollout, and compare against the same observations. For musculoskeletal lanes, preserve excitation/activation dynamics and physiological limits. Repair environment probes before interpreting availability.

**Acceptance / Verification:** [ ] nonzero native test count on pinned host; [ ] saved controls drive a fresh simulation, not a loaded state trajectory; [ ] derivative/energy/contact/constraint checks use independent references; [ ] aligned common-marker metrics plus all-engine-specific limitations; [ ] unsupported features produce unavailable/rejected states; [ ] portable reproduction and failure receipts for both clubs.

**Deliverable:** three native acceptance packages; no requirement to pretend every lane has identical fidelity or speed.

### MMR-11 — Repair Reduced-Model and Club-Only Product Claims

**Priority / Owner / Size:** P1; tour-baseline/club-only owners; M. **Disposition:** extend #10602 and product-review R12; historical TB epic need not be blindly reopened. **Depends on:** MMR-01/02.

**Problem and Evidence:** four pendulum receipts are disqualified; planar upper body has an irreducible normal residual; club-only has 80 unresolved cells.

**Scope:** regenerate honest baselines; independently validate landmark mapping and physical 3D metric calculations; distinguish teaching/planar approximation from spatial fitting. Select required native club-only cells with the Board, preserve the complete roster, and explain every excluded cell.

**Acceptance / Verification:** [ ] raw-to-package reproduction generates the reported status and residual; [ ] no fabricated out-of-plane values or missing geometry/control hashes in a promoted package; [ ] planar floor prevents futile fitting; [ ] verified club-only fit requires fresh continuous replay; [ ] inferred body posture is always labeled; [ ] matrix cannot show all-complete while required cells are unresolved.

**Deliverable:** trustworthy model chooser and baseline packages. **Out of Scope:** requiring every simple model to match all full-body markers.

### MMR-12 — Ship the Historical-Video Evidence Review Workflow

**Priority / Owner / Size:** P1; Shadow Tracker UI/service; M. **Disposition:** bounded follow-up of ST-11/#10134, coordinate #10380. **Depends on:** none for manual slice; MMR-01 for qualification badges.

**Problem and Evidence:** source/mask/session services exist, but a complete installed import → review → correction → reopen journey needs end-user proof.

**Scope:** file import, shot selection, actual decoder timestamps, rotation/crop/interlacing/playback-speed provenance, manual body/club masks, revision lineage, worst-frame navigation, safe save/reopen/export and cancellation. Display unavailable auto-fit honestly.

**Acceptance / Verification:** [ ] real constant/variable-rate clips plus cuts and slow motion preserve timing; [ ] same frame IDs do not collide across shots; [ ] corrections invalidate downstream results; [ ] bounded memory and prompt cancellation during decode, not only afterward; [ ] installed PyQt journey tested with keyboard/visual review; [ ] corrupt media and missing codecs leave recoverable sessions.

**Deliverable:** useful manual archive tool while reconstruction remains experimental. **Out of Scope:** fabricated 3D or forces from manual masks alone.

### MMR-13 — Integrate and Benchmark Real Body and Club Segmentation

**Priority / Owner / Size:** P1; vision provider; M. **Disposition:** follow-up of ST-04/#10127 and #10231; verify closed issue scope before filing. **Depends on:** MMR-12.

**Problem and Evidence:** current ModelSegmentationProvider explicitly refuses inference; a checkpoint path is not a working model.

**Scope:** select a pinned SAM-family or other provider through a bounded comparison; separate person/club masks, manual correction, occlusion states and per-frame provenance. Keep provider optional and lazy.

**Acceptance / Verification:** [ ] actual frames run actual pinned weights and generate mask artifacts; [ ] arbitrary/corrupt checkpoints cannot pass; [ ] held-out modern/archive clips include blur, thin shafts and occlusion; [ ] report body IoU, club boundary/recall, correction effort, latency and memory; [ ] no hidden downloads; [ ] model/code/weight terms and hardware requirements recorded; [ ] manual path works when provider is absent.

**Deliverable:** real inference adapter and benchmark/model card. **Risk:** good person masks may still miss the club at impact.

### MMR-14 — Unify Camera, Morphology and Native State for Shadow Fitting

**Priority / Owner / Size:** P0 for automated fitting; reconstruction + engine owners; L. **Disposition:** follow-up of ST-01/ST-05/ST-06; reuse #9619 contracts. **Depends on:** MMR-02/06 and MMR-12; masks may be manual.

**Problem and Evidence:** source renderer has 37 fields, legacy native rollout 41, other models 27/44; the old short probe violates grip closure and archived camera/scale is ambiguous.

**Scope:** version named q/v mapping including quaternion derivatives; bind real subject geometry to renderer and native rollout; estimate camera/time/scale with declared priors; independent limb/club projection oracles; multi-hypothesis initialization with no-evidence abstention.

**Acceptance / Verification:** [ ] FK and velocity parity across initialization/render/rollout; [ ] offscreen clipping, unequal focal lengths, distortion/crop and occlusion verified; [ ] grip translation/rotation gates satisfied separately; [ ] empty/invalid observations cannot select a winner; [ ] yaw/depth/scale ambiguities produce alternatives; [ ] no fictitious TourCapture is constructed to satisfy a video-only API.

**Deliverable:** physically qualified renderer/rollout boundary. **Out of Scope:** freezing an accuracy claim from the legacy 25 ms probe.

### MMR-15 — Wire Continuous Shadow Optimization With Uncertainty and Abstention

**Priority / Owner / Size:** P1; Shadow Tracker dynamics; L. **Disposition:** follow-up ST-07–ST-10, coordinated under #10375/#10380. **Depends on:** MMR-14; MMR-13 optional for manual-mask pilot.

**Problem and Evidence:** ControlFitter exists, while public service.fit refuses the unqualified backend; historical monocular footage cannot uniquely determine full motion/forces.

**Scope:** wire one real backend only after gates pass; fit camera/morphology/initial state/control in staged bounded steps; fresh replay; ensemble alternatives and uncertainty calibration; recoverable cancel/resume. Keep observational PINN shadow separate.

**Acceptance / Verification:** [ ] real video → observations → native continuous 3D candidate → fresh replay through public service; [ ] no per-frame state resets or hidden pelvis assistance; [ ] modern synchronized held-out video+C3D validates joint/marker/club errors by phase; [ ] archive stress set tests cuts, blur and unknown camera; [ ] confidence coverage and abstention assessed on holdouts; [ ] torques/forces labeled model-dependent and unavailable when unidentifiable.

**Deliverable:** bounded pilot result and evidence-based go/no-go for automated product. **External Dependency:** consented/usable ground truth and physical capture belong in the existing deferred-validation authority, not synthetic substitution.

### MMR-16 — Publish a Best-Candidate Viewer With Honest Residuals

**Priority / Owner / Size:** P1; UX + matching service; M. **Disposition:** extend #10380/#10382, reuse Tour Matching Viewer/Matched Swing Browser. **Depends on:** MMR-01/02; consumes current rejected candidates immediately.

**Problem and Evidence:** attractive poses, historical returned81 GIFs and newer numerical fits are easy to confuse; incompatible metrics cannot identify one global best.

**Scope:** comparable-candidate filters; observed dots versus fitted mesh; source/candidate hashes; temporal alignment; synchronized camera views, residual vectors/timeline and failure badges. Export board-ready video/stills with legible units and evidence link.

**Acceptance / Verification:** [ ] raw observations cannot be overwritten by fitted positions; [ ] all captions match selected candidate and drive mode; [ ] worst marker/phase can be selected from the residual plot; [ ] camera/appearance changes do not change physics score; [ ] numerical receipt and rendered frame indices agree; [ ] same journey works in installed PyQt and supported web/API surfaces with accessibility and missing-engine recovery.

**Deliverable:** inspectable best-per-profile demonstration. **Out of Scope:** using aesthetics as an acceptance gate for numerical validity.

### MMR-17 — Establish Clean-Host End-to-End and Native Release Gates

**Priority / Owner / Size:** P0 for release; QA + packaging; L. **Disposition:** extend #10380/#9417 and existing nightly lanes. **Depends on:** MMR-01/02/03 and the engine work required by the declared release profile.

**Problem and Evidence:** many unit/GUI journeys validate software contracts but not native matching; default test runs skip live simulation, optional engines and physical evidence.

**Scope:** clean installation → load C3D → calibrate → fit → cancel/resume → independent replay → compare → export/import → reopen → uninstall/upgrade, on both shells and actual licensed R2025b/native engine hosts. Use the pinned Tools provider and fail closed on missing native jobs.

**Acceptance / Verification:** [ ] test matrix reports passed/failed/skipped/unavailable separately; [ ] one successful real-data dual-club path per advertised engine/profile; [ ] tampered package, missing engine, unsupported model and corrupt capture fail actionably; [ ] UI and CLI metrics agree; [ ] release artifacts reproduce stored receipts; [ ] no physical/scientific gate is closed by mocks; [ ] performance/memory/cancellation budgets are ratified from measured pilot and regression enforced.

**Deliverable:** signed release matrix and reproducible evidence archive. **External Dependency:** independent biomechanical validation remains separately approved; product testing is not scientific certification.

### MMR-18 — Benchmark MATLAB Iteration Throughput and Portable Model Exchange

**Priority / Owner / Size:** P1; MATLAB runtime + platform; M. **Disposition:** child of #10950/#10430, coordinate #11006. **Depends on:** MMR-03/04 and frozen MMR-02 inputs.

**Problem and Evidence:** reported 43-minute IK and approximately 18-minute Shape runs make naive optimization expensive; native/analytical differences and controller assumptions can invalidate transferred candidates.

**Scope:** stage timing, supported Fast Restart/session reuse, finite-difference versus analytical proposal cost, warm starts, bounded jobs and content-addressed cache. Export named coordinates, frames, mass/inertia, contacts and actuation; preserve engine-specific features instead of forcing a lossy lowest-common-denominator model.

**Acceptance / Verification:** [ ] cold/warm timing and peak memory on the same captures/hardware; [ ] cache invalidation covers model, geometry, marker map, initial state, solver, controls and provider revision; [ ] topology/operating-point changes trigger rebuild; [ ] identical input yields same metrics within declared tolerances; [ ] optimization winner gets uncached native cold replay; [ ] unsupported neck/contact/muscle mappings reject rather than silently drop dynamics.

**Deliverable:** measured throughput improvements and interchange conformance matrix. **Out of Scope:** purchasing acceleration products or changing required MATLAB release without an explicit decision.

## Board Sequencing and Completion Definition

<!-- prettier-ignore -->
| Wave | Proposed Work | Exit Evidence |
| --- | --- | --- |
| A: Trust and Inventory | MMR-01/02/03/04; manual-video MMR-12 can proceed independently | One honest ledger, frozen metrics, promoted variant inventory, compiled budgets and manual archive journey. |
| B: Anatomy and Native Fit | MMR-05/06/07/08/09/10/11/18 | Subject/model calibration, independent dual-club replays, preserved physics, measured optimizer cost and truthful reduced-model status. |
| C: Video Reconstruction | MMR-13/14/15 | Grounded masks and camera/state binding, actual continuous pilot, calibrated uncertainty/abstention. |
| D: Product Release | MMR-16/17, iterated throughout | Installed end-to-end journeys, compare/export/reopen, native host matrix and reviewed scientific limitations. |

Board decisions needed: release profiles and mandatory engines; pure torque versus assisted tracking product labels; subject/force validation resources; mesh asset strategy; performance budgets after pilot; acceptance owner for each wave. Existing #10979 work is already active and should continue under its owner. Resolve overlaps with #11006, #10375/#10380/#10382, #10602 and product-review R05/R06/R12 before filing new children. The scoped live issue and open-PR snapshots were reviewed; neither is proof of absence of closed historical work. Do not reopen/close by title alone.

Completion requires both clubs, complete requested time coverage, repeatable native results, valid marker attachments, declared assist/control mode, all applicable numerical/physical gates, clean-host packaging, error recovery and reviewable artifacts. Historical footage additionally needs uncertainty/abstention and independent modern ground truth. The current request completes with this review and its PR; it does not implement or scientifically accept the proposed program.

### Runner Dashboard Board Prompt

> Review this packet at its PR commit together with open PR #10963 at the stated SHA and existing product-review R05/R06/R12. Challenge the evidence and approve, amend, merge into an existing issue, or reject each MMR proposal. Preserve separation of kinematic fit, dynamic replay, scientific qualification and product acceptance. Select one accountable owner and a bounded first slice for approved work, with dependencies, resource requirements, unchanged numerical gates and raw evidence deliverables. Do not dispatch another agent onto #10979 without coordination. Return a disposition table keyed by MMR ID; only then create implementation issues under the existing epics. Keep external validation in the existing planning authority. No model is “matched” merely because it renders or an issue is closed.

## Review Validation

Numbers are committed evidence or attributed branch reports, not newly measured results. Initial test collection failed before the isolated worktree Tools pin was initialized; the rerun below passed. This setup failure is not a matching defect.

- Focused pytest: `python3 -m pytest tests/unit/shadow_tracker tests/unit/motion_matching/test_acceptance.py tests/unit/motion_matching/test_acceptance_gate_honesty_10960.py tests/unit/motion_matching/pipeline/test_receipt_provenance_chain.py tests/acceptance/test_tour_baselines_journey.py -q -n 0 --no-cov --timeout=60` — 375 passed, no failures or skips; existing deprecation warnings. Includes decoder and calibration checks, synthetic fitting, gates/provenance and the baseline widget journey. This is not full scientific/native acceptance.
- `python3 -m ruff check .` — passed; `python3 -m ruff format --check .` — 8,231 files already formatted. No Python source changed.
- `python3 scripts/check_document_title_case.py docs/development/2026-09-28-motion-matching-board-review.md` — zero violations. Local Markdown evidence links checked for existence. File-size budget passed.
- Full repository pytest, licensed MATLAB reruns, fresh Drake/Pinocchio/OpenSim/MyoSuite fitting, installed-product acceptance and independent physical validation were not run. These remain explicit work packages.
