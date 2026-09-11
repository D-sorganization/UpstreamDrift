# Simscape Tour-Average Fit Continuation

## Active 6Th-Order Polynomial (Sextic) Continuation — 1.2s Milestone & 1.3s Horizon

`prefix-1200ms-sextic-01` has completed with native fit and independent cold replay exit codes zero on DeskComputer R2025b (runtime `0715f95f3`). Over 433 samples and 25 markers across 0–1.20 s:

- **Marker Accuracy**: Overall RMS is **311.617528 mm**, p95 **643.184442 mm**, max **2149.570454 mm** (580 evaluations computed).
- **Sub-window Fidelity**: Across the shared [0, 0.8] s window, marker RMS is **87.427014 mm**; across the shared [0, 1.0] s window, marker RMS is **192.206387 mm**; across the shared [0, 1.1] s window, marker RMS is **245.850057 mm**. The extension [1.1, 1.2] s deepens the downswing power phase (708.109788 mm).
- **Cold Validation & Audit**: Independent cold replay reproduces marker coordinates with zero error; initial states pass with q error 7.246e-13 and qd error 3.986e-14. All 27 actuator channels (24 torques, 3 forces) audit to within floating-point precision ($3.411\times 10^{-13}$ N force error, $2.842\times 10^{-14}$ Nm torque error).
- **Archive & Evidence**: The 138-file bundle archive is verified locally and remotely with identical SHA256 `b6cfe04e478eef2fbbeb66f1cf061c9362443cc964cbda18c629108141550c98`. Evidence is committed in `native_evidence/prefix_1200ms_*` and receipt `prefix_1200ms_sextic_bundle_receipt.json`.
- **3D Video Overlay**: Generated and verified at `c:\Users\diete\Repositories\simscape-tour-checkpoints\prefix-1200ms-sextic-01\prefix_1200ms_overlay.mp4` (433 frames, 30 fps) and GIF (433 frames, 20 fps) displaying the synchronized Simscape model skeleton and C3D target markers over the first 1.20 s of motion.

Continuation has advanced to **`prefix-1300ms-sextic-01`** (1.30 s horizon, 469 samples, 25 markers, degree-6 Bernstein basis with 189 parameters). Transferred from `prefix-1200ms-sextic-01/first_prefix_fit.json` with identical global t0 initial state and fixed geometry. Actively running in background on DeskComputer R2025b.

## Worktree Cleanup Complete; Bounded Resumption Ready

All 20 user-approved completed worktree directories are removed, with original branches preserved. C: free space is about 30.4 GiB. Verified submodule-history backups are retained locally; borrowed-object dependencies discovered during removal were restored and relocated to stable `.git/retained-submodule-objects`. Do not delete that active object storage. Final checks show no missing alternate targets; the unrelated pre-existing unreadable `_wt_claude_model` was preserved. See RESUME_AND_STORAGE.md and its cleanup receipts for exact recovery paths and verification.

Read [Resumption Plan](docs/development/simscape_tour_matching/RESUMPTION_PLAN.md). Next proposed bounded assignment is 0.7 s cubic continuation from the verified 0.6 s / 9.062024 mm candidate, followed by cold replay, all-27 audit, interval-specific comparison and archive. A lower-cost coding agent can run the established scripts; a strong reviewer should decide subsequent horizons, model/geometry changes and acceptance. No new fitting process or agent has been launched. R2025b and the unfinished full-swing objective are unchanged.

## Saved Checkpoint and Storage Review — 2026-09-10

- **Archive root**: `C:/Users/diete/Repositories/simscape-tour-checkpoints`
- **Completed checkpoints**: `prefix-50ms-01`, `prefix-100ms-linear-01`, `prefix-200ms-linear-01`, `prefix-300ms-quadratic-01`, `prefix-300ms-quadratic-02-refined`, `prefix-400ms-quadratic-01`, `prefix-500ms-cubic-01`, `prefix-600ms-cubic-01-refined`, `prefix-700ms-cubic-01`, `prefix-800ms-quartic-01`, `prefix-900ms-sextic-01`, `prefix-1000ms-sextic-01`, `prefix-1100ms-sextic-01`, `prefix-1200ms-sextic-01`
- **Experimental**: `calibrated-100ms-linear-01`
- **Active candidate**: `prefix-1300ms-sextic-01` (running in background on DeskComputer)

# Common-Reference Calibration Continuation

This section supersedes every historical running/pending statement below. Both latest fits and their independent cold/all-27 audits are terminal, with explicit native fit and cold exit codes zero. No new fitting run was launched during the user's storage review. MATLAB R2025b remains the required release. The full 1.813889 s matching epic #9921 is unfinished.

Best verified longest forward prefix: DeskComputer `C:/Users/diete/SimscapeTour9921/prefix-600ms-cubic-refine-02`, pinned runtime `0715f95f3`. Cubic continuous torques, one global t0, unchanged original qualified state/geometry/attachments. 217 samples and 25 markers; RMS 9.062024 mm, p95 19.457409 mm, maximum 42.595899 mm. 877 native evaluations including baseline/final; optimizer convergence and numerical acceptance remain false (budget ten exhausted). Cold marker difference is zero; all 27 effort channels pass (force 2.274e-13 N, torque 2.842e-14 Nm).

Distinct calibrated 0.1 s experiment: `calibrated-prefix-100ms-01`, runtime `f28343726`, RMS 12.191880 mm, p95 21.632996 mm, maximum 25.221325 mm. Native initial-state verification passed with projection error 7.149e-10 m and initial target RMS 12.207266 mm. Cold marker difference and all-27 effort differences are zero. Convergence/acceptance remain false. Never transfer torques between these different state/attachment identities.

Both complete raw run ZIPs are now saved on DeskComputer and locally under `C:/Users/diete/Repositories/simscape-tour-checkpoints`, with identical SHA256, all ZIP CRCs checked and every archived file hashed. Refined archive SHA256 `6da7884767a01bfd4bf45fd1e26a7defecfc49a361c6c07168d5fb327fdeafe5`; calibrated archive SHA256 `bfe11699f3a3b2b4d805c5c3659473ef358c74de77be9cbccb0f70624a9b84a2`. Corresponding `prefix_600ms_refined_*` and `calibrated_prefix_100ms_*` Git evidence contains reports, audits and receipts. The refined plot was generated with the existing reproduction script and visually reviewed; it is a diagnostic chart, not a full-swing animation.

Read [Resume and Storage Review](docs/development/simscape_tour_matching/RESUME_AND_STORAGE.md) for exact result locations, reproduction commands and cleanup inventory. Preserve the Simscape source worktree, both remote pinned runtimes, the isolated native audit checkout, all raw archives, stashes and model backups. Other tasks' dirty changes have not been committed or removed. Repository status initially showed two stat-only entries with no content diff.

Next matching work after storage recovery: extend the refined original-state candidate to a modest longer prefix (for example 0.7 s) with the existing identity-checked transfer, continuous polynomial controls and all integration from t0; cold replay and audit all 27 efforts again. Compare the calibrated sequence separately before investing in its longer horizon. Full-swing fit, convergence, geometry/attachment adequacy and physiological qualification remain outstanding. The goal service currently reports blocked from an earlier run; that is not evidence the completed native jobs failed, and this tool cannot change goal status to active.

## Fresh Calibrated Forward Experiment Launched

New clean DeskComputer checkout C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-calibrated-runtime is detached at f28343726. It contains the modified native-state/projection checks and qualified calibrated seed. Fitting SSH 32952 is running C:/Users/diete/SimscapeTour9921/calibrated-prefix-100ms-01. Args: duration 0.1, constant basis, finite-difference-step 0.0001, max-nfev 5, initial_velocity_seed_calibrated_r2025b.json. No old-identity candidate/checkpoint transfer; initial controls start at zero effort. Launcher PID 40104. Source files and seed are pinned; preserve this checkout while fitting or validation is live.

Independent cold replay and all-27 audit are queued as SSH 88410. The saved cold-replay-launch.ps1 waits for the identity-checked calibrated-fit-launch.ps1 process and requires explicit nonblank fit exit zero, completed report and exact f28343726 before replay. Both scripts are copied into the run directory. The most recent report check found no first baseline report yet; do not claim the updated runner's native integration has passed until initial_state_verified and projection metrics are recorded. Observe the existing handles/processes before any restart.

Original-state refinement remains separate: SSH 37571, prefix-600ms-cubic-refine-02 at 0715f95f3, with cold validation SSH 5157. Both original and calibrated sequences must retain distinct identity and evidence. Published code/state qualification: f28343726; all commit/push checks passed. Next collect initial integration and complete both fit/replay pipelines, compare residuals with their respective kinematic diagnostics, then extend the promising sequence.

## Calibrated Initial State Qualified; Updated Runner Needs Native Integration

Native calibrated initialization SSH 64328 is terminal exit 0. Status initial-state-qualified and initial_state_verified=true. Cold 50 ms replay errors: q 1.748969e-9, qd 5.046641e-11, native marker projection 7.149074e-10 m; all below the unchanged 1e-8 checks. Target capture RMS remains 12.207266 mm and is reported separately. The tangent velocity was freshly fitted using the new pose and fixed calibrated offsets. This qualifies initialization only, not torque fitting or physiological validity.

Verified calibrated-initial-state-01-bundle.zip is stored on both machines, SHA256 b4b626e4123f4dc0836b79e26e90b28e59697a0e86ebded4e0634fcf4a11a59a. It preserves red TooManyInputs/exit 1, corrected source, input candidate/pose, native MAT/JSON and green exit 0. initial_velocity_seed_calibrated_r2025b.json is the qualified new state; calibrated_initial_state_01_receipt.json records provenance and hashes. Do not confuse it with the unqualified attachment_calibration_candidate_02.json.

Next execute the updated first_prefix_fit runner in a NEW clean calibrated runtime, using this state and a fresh short-prefix experiment without an old-identity torque transfer. The helper's 57 selected Python tests pass, but the modified runner must still be exercised natively. It records seed-specific attachment qualification and actual tracked differences from HEAD, instead of the historical hardcoded patch list. Keep original-state refinement SSH 37571 and cold validation SSH 5157 pinned to 0715f95f3. No other native pose/qualification process is live.

## Verified 0.6 s Milestone; Smaller-Step Refinement and New State Qualification Running

Original 0.6 s cubic forward fit and cold/all-27 validation are terminal (SSH 4727/24801, explicit exits 0). 217 samples, 25 markers, 335 native evaluations: RMS 11.914882 mm, p95 28.840186 mm, max 57.995739 mm. Budget eight exhausted; optimizer convergence/numerical acceptance false. Cold replay reproduces every marker exactly; initial q/qd errors 7.246e-13/3.986e-14; all 27 sensed efforts pass (2.274e-13 N, 2.842e-14 Nm). Verified 100-file prefix-600ms-cubic-01-bundle.zip on both machines has SHA256 b2b924817717756bda046e2b38e3d25bc3fea95989191b8c0c618e771d318c5e. Plot visually reviewed; prefix*600ms*\* and actuator_audit_600ms_r2025b.json contain evidence.

Current original-state refinement: SSH 37571, DeskComputer prefix-600ms-cubic-refine-02, runtime 0715f95f3. Same duration 0.6/cubic/global t0/geometry/attachments; candidate transferred from completed 0.6 s. Finite-difference-step reduced tenfold to 0.00001, max-nfev 10. Launcher PID 7980. Cold/all-27 validation queued as SSH 5157 with explicit receipt/source checks. Both launch scripts saved in the run directory. Preserve runtime while either handle is live.

Calibrated-offset native poses are now complete at all 55 sampled times after alternate-seed recovery: 26.824744 mm overall RMS over 1363 valid observations, versus old-offset 30.136761 mm; initial capture RMS is 12.207266 mm. These are local kinematic fits only. Calibrated prefix SSH 44400 rejected frame 445 after saving 37 poses, archive connected-marker-calibrated-01-bundle.zip SHA256 2f46aed774b7e1dbbc8c71638df6817c5df4b0b0058656af3977b34107d78a34. Suffix SSH 88102 is terminal exit 0; connected-marker-calibrated-suffix-01-bundle.zip SHA256 c531b739fafb7c8f9f832b7a5672fb5f1a85d66a6fbbbd137e900f26a54af99b. Both archives are verified locally/remotely with receipts.

SELF adds verify_native_initial_state, separating strict q/qd/native-projection consistency from capture fit error. Five new tests first failed, then all 57 selected state/transfer/CLI tests passed; scoped Ruff/mypy pass. first_prefix_fit now constructs the declared q0 native projection and verifies baseline against it, reporting initial_target_rms_m and initial_target_max_error_m independently. It no longer assumes perfect target coincidence. The live fitting runtime remains OLD 0715f95f3 and has not received this change; new-runner native integration remains to be exercised after candidate qualification.

initial_velocity_seed(repo,run_dir,candidate_path,pose_path) now optionally retains calibrated offsets and uses a valid frame-1 native q instead of recalculating single-frame offsets. It refits tangent qd and requires the cold initial marker projection to match native kinematics, while reporting target residual separately. Its four-argument native probe first failed with TooManyInputs (red exit 1). Updated native qualification is RUNNING as SSH 64328, DeskComputer C:/Users/diete/SimscapeTour9921/calibrated-initial-state-01, launcher calibrated-initial-green.ps1; source inputs attachment-calibration-02/candidate_seed.json and connected-marker-calibrated-01/frame-00001.json. Preserve isolated checkout and partial JSON/MAT; do not claim native qualification until status, actual exit and errors pass. Capture payload and red log/receipt are saved in the run. If qualified, start a distinct forward experiment with new identity; never transfer old torque checkpoints across changed offsets/q/qd.

## Geometry Comparison Archived; Fixed-Offset Calibration Candidate Under Test

Geometry sweep SSH 59719 completed with all four cases computed, all 52 native poses satisfying targets/constraints. Across the same 13 times (0–1.2 s), baseline (14.5,12) inch lengths gives 25.762390 mm RMS; (13.5,12) gives 27.299574, (15.5,12) 26.106391, (14.5,11) 27.522377, and (14.5,13) 26.254850 mm. Retain current geometry for the existing forward sequence. This is a local fixed-offset comparison, not anatomical/global optimization. Verified multiframe-geometry-01-bundle.zip SHA256 78f6dfbf4a26aba7c4af9a93dd831d0230f1db34fcb36072a204919a3ccbec9a is on both machines; multiframe_geometry_01_receipt.json records results and hashes.

Native export_pose_body_frames reconstructed all 55 sampled poses, asserted all q targets/constraints and reproduced every saved projected marker exactly (max difference 0). Export SSH 36866 is terminal exit 0. Verified calibration-body-frames-01-bundle.zip SHA256 012b126fc68f56077ced7a3556f1fed2e7b8ce72c852064103c88a31f3cafee1 is on both machines. It supplies body origins and world-from-body matrices for attachment calibration; it is not a dynamics replay.

SELF adds calibrate_pose_attachments.py, reusing the public pose_estimation.estimate_keypoint_offset estimator. Four existing estimator tests and two new CLI tests pass; new tests first failed before implementation and cover rotated-frame offset recovery, invalid-observation exclusion, valid origins and capture-identity rejection. Scoped Ruff/mypy pass. The CLI must include repo and repo/src import roots because the public package imports bunkershot3d. It verifies native provenance/marker parity and masks invalid observations before estimation. Every candidate explicitly sets initial_state_verified=false and status attachment-calibration-candidate.

With geometry and native poses fixed, calibration reduces RMS from 30.136761 to 27.228651 mm; maximum offset change 25.987986 mm. This conditional least-squares improvement is not anatomical identification or an improved forward replay. Candidate evidence: attachment_calibration_candidate_02.json and attachment_calibration_02_report.json. Verified attachment-calibration-02-bundle.zip SHA256 8f85c5c3dd094ef528e75aaad8959a13656295482bfc57d14fad80d3d654d31a exists on both machines, with source/tests and estimator hash. Earlier local attachment-calibration-01 is preserved; 02 adds explicit estimator provenance.

Now testing new offsets through native poses: SSH 44400, DeskComputer C:/Users/diete/SimscapeTour9921/connected-marker-calibrated-01, launcher connected-calibrated-launch.ps1. It uses attachment-calibration-02/candidate_seed.json and 55 times (1:12:649)' in the isolated checkout with current exporter/calibration patches. Same fixed geometry, offsets constant within this new experiment. Initial q is a guess; no forward state is qualified. Preserve partial outputs on native rejection; use alternate pose seeds if needed and record them. If the pose fit improves, requalify q/qd and explicitly handle nonzero initial marker residual before starting a distinct forward experiment. Never transfer an old torque checkpoint across changed attachment identity.

The original-state 0.6 s forward fit SSH 4727 and queued cold/all-27 validation SSH 24801 remain pinned to 0715f95f3 and are unaffected. Full 1.814 s forward matching, calibration qualification and final physiological/solver acceptance remain open.

## Sampled Pose Diagnostic Complete; Multiframe Geometry Comparison Running

Pose continuation SSH 83166 is terminal exit 0. Its 18 suffix poses plus the 37 saved masked prefix poses cover indices 1:12:649 (55 samples, 0 to 1.8 s). All final native target/constraint checks pass. Across 1363 valid marker observations, sampled RMS is 30.136761 mm. Largest per-marker RMS: HeadSide 50.986 mm, HeadFront 50.943 mm, WaistLeft 45.503 mm, BackLeft 39.529 mm, WaistRBack 38.487 mm, LWristTop 38.342 mm. This is a collection of local connected-kinematic solutions with alternate-seed recovery, NOT a globally optimal, continuous or torque-driven trajectory. It also does not cover the final 1.8–1.814 s interval. Use it to investigate model/attachment restrictions, not as goal acceptance.

Verified connected-marker-continuation-01-bundle.zip is on both checkpoint roots; SHA256 0b3bd8babe9ab519b3c8d3efc7e6b4281031e4887a2e786cf123679294648deb. connected_marker_continuation_01_receipt.json identifies the paired prefix archive and all 55 sample metrics, valid observation count and file hashes. Original failed/unmasked experiments remain separately preserved.

Multiframe geometry comparison now running: SSH 59719, C:/Users/diete/SimscapeTour9921/multiframe-geometry-01. Launcher multiframe-geometry-launch.ps1 and exact experiment.m record four fixed (UpperArmLength,LowerArmLength) inch pairs: (13.5,12), (15.5,12), (14.5,11), (14.5,13), compared with baseline (14.5,12). Each case fits 13 times from 0 to 1.2 s, indices 1:36:433, with existing fixed attachments and the same original q starting guess. Geometry stays constant within each case. Derived seed JSON explicitly marks initial_state_verified=false; no candidate is qualified for forward replay. Individual known native pose rejections are saved in case receipts and do not stop unrelated cases; outer exit zero means the sweep completed, NOT that every case succeeded. Inspect per-case status and every pose before comparing residuals. The isolated checkout is unchanged and must remain pinned while this process is live.

Forward work remains the 0.6 s cubic fit SSH 4727 with queued cold/all-27 audit SSH 24801, exact runtime 0715f95f3. Latest observed evaluation 85. Next collect geometry comparison and decide whether to refine geometry/attachments and requalify initialization; preserve the current fixed-state forward sequence as its own experiment.

## Masked Alternate Poses Qualified; Suffix Continuation Active

Masked reseed-02 is terminal (SSH 81482 exit 0). All three cases satisfy native targets/constraints: frame 445, 1.233333 s, 22 valid markers, RMS 38.358600 mm; frame 457, 1.266667 s, 25 markers, RMS 28.903796 mm; frame 541, 1.5 s, 22 markers, RMS 44.390240 mm. Thus the failed frames have feasible local solutions with alternate starting poses; do not infer an impossible model pose from the rejected previous attempts. This still does not prove global optimality or forward-dynamics matching. Verified archive connected-marker-reseed-02-bundle.zip SHA256 56f474503e41d847f1fcb366ae1e9c61032f2cefc6dd342bfecb3ef0abda4c9c exists on both machines; see connected_marker_reseed_02_receipt.json. Masked failed run 03 is also archived on both machines: connected-marker-poses-03-bundle.zip SHA256 c2919c1bd7cf14f465dd0380f087a8ee39624aaa827d85ec376ed7dbc08d87aa, with its own receipt.

Pose suffix continuation now running: SSH 83166, C:/Users/diete/SimscapeTour9921/connected-marker-continuation-01/case-01. It starts from the recovered frame-445 q, then requests indices (445:12:649)' through 1.8 s. Launcher connected-continuation-launch.ps1 and experiment.m are saved in remote scratch/run. Same isolated checkout and validity-masked helper; derived starting guess explicitly unqualified as a forward state. Preserve this run and collect every valid frame plus exit receipt. Together with the masked run-03 prefix it may provide a full sampled kinematic diagnostic; it never replaces the single-t0 torque-driven replay.

The 0.6 s cubic native fitter SSH 4727 and queued independent validation SSH 24801 remain on exact 0715f95f3. Preserve the fitting runtime. Next collect the pose suffix, then compare fixed arm-length candidates across several valid target times before requalifying any geometry change for forward fitting.

## Verified 0.5 s Forward Fit; 0.6 s Cubic Fit Running

The 0.5 s quadratic fit and queued cold replay are terminal (SSH 77987/87243, both explicit exit 0). Runtime 0715f95f3. 181 samples, 25 markers, 335 native evaluations; RMS 5.875584 mm, p95 13.214460 mm, maximum 34.032793 mm. Optimizer convergence and numerical acceptance remain false (budget eight). Cold marker difference is exactly zero, initial q/qd errors 7.246e-13/3.986e-14. All 27 sensed efforts qualify: force max error 2.274e-13 N, torque 2.842e-14 Nm, no unlogged channels. Plot visually reviewed. Verified 100-file prefix-500ms-quadratic-01-bundle.zip exists on both machines, SHA256 4171130239bac03814099b474b9714e35821650ff6bc164b6e86c6d2aea4d91a. See prefix*500ms*\* evidence and actuator_audit_500ms_r2025b.json. This remains an exploratory short prefix, not full-swing acceptance.

Next forward experiment: SSH 4727, DeskComputer C:/Users/diete/SimscapeTour9921/prefix-600ms-cubic-01, runtime still 0715f95f3. Launch args duration 0.6, basis cubic, finite-difference-step 0.0001, max-nfev 8, transfer-report from completed 0.5 s. Same qualified initial state, fixed geometry/offsets, single t0, continuous polynomial controls. Launcher PID 39832. Dependent independent cold replay/all-27 audit is queued as SSH 24801; it waits on that identity-checked launcher and requires explicit nonblank exit zero/completed fit/exact source. Both launch scripts are saved in the run directory. Preserve runtime until both handles are terminal.

Corrected masked pose run 03 is terminal: SSH 50910 exit 1, fit_golf_pose_seed:constraints at frame 445 (1.233333 s). It saved 37 valid poses through frame 433 (1.2 s), last RMS 35.644182 mm with 25 observations. The masking fix is tested, but this constraint failure persists with the reduced observation set; it needs alternate seeding/solver investigation. Run directory and adjacent log/exit remain on DeskComputer and still need their final archive. Do not treat this as a successful full pose trajectory. Next try the corrected selector with nearby successful coarse pose seeds, preserving failed evidence and comparing valid observations only. No live pose process remains.

## Capture Validity Defect Corrected; Masked Pose Rerun Active

CORRECTION: the large diagnostic errors at frame 445 (1.233333 s) and 541 (1.5 s) were contaminated by missing clubhead markers stored as zero coordinates with capture.valid=false. Original attached_pose_probe ignored that mask. Earlier interpretation as only local-solver failure is superseded. Pose runs 01, 02 and reseed-01 are preserved, with affected samples explicitly flagged in receipts; their contaminated errors must NOT be used for geometry/feasibility conclusions. Reseed SSH 44899 is terminal exit 0; archive connected-marker-reseed-01-bundle.zip SHA256 049cee95c8f212e1d3eac34d0d5bfb8e6cf6e79b872dafc483a10009e4df8dcf verified on both machines. A failed final native constraint check in run 02 remains an actual rejected result, but followed a contaminated previous pose.

SELF adds tested select_capture_marker_frame(capture,index,labels), preserving requested label order and selecting only validity-flagged finite observations. A valid point at the origin is retained; invalid zero placeholders are excluded. Empty selections, unknown labels, frame bounds, shape and nonfinite valid points fail explicitly. Two R2025b tests first failed, then passed (red exit 1, green exit 0). Verified capture-validity-01-bundle.zip SHA256 93423762b2e2180fdc16b15961192e83c41af8bb7ca5d2d4942961fff5d5f058; receipt records source/test/log hashes. The pose runner now selects matching bodies/offsets and records per-frame labels, selected indices and valid_marker_count. No torque-fitter source changed.

Forward first_prefix_fit.py already sets invalid observed points to NaN, and the rigidity-floor diagnostic already uses capture.valid. All 25 selected markers are valid through 0.5 s; first invalid selected sample is 1.233333 s. The completed short-prefix forward evidence is unaffected by this pose-only defect.

Corrected integration now running: SSH 50910, C:/Users/diete/SimscapeTour9921/connected-marker-poses-03, launcher connected-pose-03-launch.ps1. It repeats all 55 requested poses at 1/30 s spacing in the isolated actuator-audit checkout, with current validity helper/runner patches. Preserve this checkout while active. Check frames 445 and 541 contain 22 valid markers, and collect the complete interval before assessing geometry. The independent 0.5 s forward fit and queued replay remain SSH 77987/87243 on 0715f95f3.

## Finer Pose Failure Preserved; Alternate Seeds Running

The finer connected-marker-poses-02 experiment is terminal: SSH 25711 exit 1. It saved 38 of 55 requested poses through frame 445 (1.233333 s), then fit_golf_pose_seed:constraints rejected frame 457 (1.266667 s). Do not resume by assuming the failed pose is valid. The 1.233333 s local RMS was 255.942 mm. Common-time RMS values through 1.1 s differ from the coarse run by at most 0.009251 mm, so the early residual trend is reproducible, but neither run proves a global optimum. Archive connected-marker-poses-02-bundle.zip is verified on both machines, SHA256 876be065dc3d32b05cffa6ccc4bc44d57031a0dcc1497b28ceeeac89e4a7d962; connected_marker_poses_02_receipt.json records partial samples, failure and file hashes.

Alternate-seed experiment is running as SSH 44899 at C:/Users/diete/SimscapeTour9921/connected-marker-reseed-01. Launcher connected-reseed-launch.ps1 is saved in the remote scratch root; experiment.m records exact MATLAB statements. Three explicit source/target frame pairs: coarse pose 469 (1.3 s) seeds targets 445 and 457; coarse pose 577 (1.6 s) seeds target 541 (1.5 s). Geometry/offsets unchanged. Derived seed files explicitly set initial_state_verified=false and mark rates unused; these are kinematic guesses, never forward-initialization evidence. The existing tested attached_pose_probe performs each single-frame fit and saves independent case directories. Preserve all attempts. The isolated checkout remains unchanged while this process is active.

The forward fit and queued native replay remain SSH 77987/87243 on pinned 0715f95f3. Latest observed fit evaluation 200; this is distinct from pose-only experiments. Next inspect alternate-seed results to resolve local-solution failures before using poses for geometry assessment, then collect and qualify the torque-fit milestone.

## Coarse Pose Diagnostic Saved; Finer Continuation Running

The connected-marker-poses-01 experiment is terminal (SSH 34677 exit 0). All 19 final poses satisfy native targets/constraints. Pointwise marker RMS is 7.278 mm at 0.4 s, 13.255 mm at 0.5 s, 40.795 mm at 1.0 s and 51.377 mm at 1.8 s. These are individual-frame errors, not prefix RMS. An isolated 426.396 mm error at 1.5 s indicates poor local pose selection; do not report this experiment as the model's best attainable fit. Largest errors include LWristTop at 0.4/0.5 s, HeadSide at 1.0 s and club markers at 1.5 s. The verified archive connected-marker-poses-01-bundle.zip exists on both machines, SHA256 f630aa9946c7aa0a25b27f567aaa50dfc0b001e02d2b92ea4f9c83c4282d6ba8. connected_marker_poses_01_receipt.json includes sample metrics, marker diagnostics and every file hash.

Finer diagnostic is now running: SSH 25711, C:/Users/diete/SimscapeTour9921/connected-marker-poses-02, launcher connected-pose-02-launch.ps1. It uses the same helper/geometry/offsets but indices (1:12:649)' (55 poses, 1/30 s spacing) to reduce warm-start jumps. Its checkout remains the isolated actuator-audit worktree with archived patches. No changes to the native forward-fitting runtime. Compare common times after completion, especially 1.5 s; retain both experiments. Subsequent geometry/attachment refinement must be qualified with native forward dynamics, not inferred from these local pose solutions.

## Connected Marker-Pose Implementation and Original Launch Record

SELF extends fit_golf_pose_seed with optional fixed body-local marker offsets, reusing project_body_markers and intrinsic_xyz_to_rotm. Repeated body names support multiple attachments. Omitting offsets preserves origin-proxy calls. Final native targets/constraints remain required. Three R2025b native tests pass: existing translated pose, rotated pose with 48 nonzero attachments (RMS below 0.1 mm), and malformed attachment-count rejection. The two new tests first failed on the missing API. Red exit 1 and green exit 0 are explicit. Verified attached-pose-01-bundle.zip is on both checkpoint roots, SHA256 4fa110a222601c04f13b01ff922b1e0db60d92507eac9db09b2c8a94b8e9e39e; see attached_pose_test_bundle_receipt.json.

Diagnostic now running separately: SSH 34677, DeskComputer C:/Users/diete/SimscapeTour9921/connected-marker-poses-01. Native checkout is UpstreamDrift-simscape-actuator-audit at base 59e944b44 plus archived sensor and current pose-helper patches. DO NOT change this checkout until its process is terminal. Launch file C:/Users/diete/SimscapeTour9921/connected-pose-launch.ps1; log and explicit exit receipt are adjacent to the run directory with .log and -exit.txt suffixes. attached_pose_probe(repo,run_dir,state_path,capture_path,(1:36:649)') samples 19 capture times from 0 to 1.8 s. It holds the qualified geometry/offsets fixed, warms each local pose from the previous result, and writes one frame JSON immediately. Preserve partial results on failure. This is a local connected-kinematics feasibility diagnostic, NOT a global lower bound or torque-driven trajectory. No result is claimed before inspecting its outputs.

The 0.5 s forward fitter (SSH 77987) and queued cold validation (SSH 87243) remain independent on runtime 0715f95f3; no fitting source was changed. Latest observed progress reached evaluation 60. Next collect both experiments, archive their exact sources/results, and use connected-pose residuals to guide geometry refinement without substituting pose fits for forward dynamics.

## Current Checkpoint: Verified 0.4 s, Fitting 0.5 s

The 0.4 s quadratic experiment is terminal: 578 native evaluations, 145 samples and 25 markers give 3.119775 mm RMS, 6.356488 mm p95 and 16.886998 mm maximum error. Zero-effort RMS is 369.924952 mm. Optimizer convergence and numerical acceptance remain false (budget eight); this is an exploratory prefix, not full-swing acceptance. Independent cold replay has zero marker difference; initial q/qd errors are 7.246e-13 / 3.986e-14. Its historical 22-channel model audits to 2.274e-13 N and 2.842e-14 Nm. Both actual fit and validation receipts explicitly contain zero. SSH 42924 and 71640 are terminal; all process IDs below for that run are historical.

Archive prefix-400ms-quadratic-01-bundle.zip is verified on DeskComputer and locally under the checkpoint roots below. SHA256: 938bb0f88f54a1d1f5c6ae8a211cee1ad73d735f8f285e2de17fc95a17fab11c. All 148 archive files passed integrity checking and have SHA256 receipts. See native_evidence/prefix_400ms_bundle_receipt.json, prefix_400ms_fit_r2025b.json, prefix_400ms_summary.json, prefix_400ms_cold_replay_r2025b.json, actuator_audit_400ms_r2025b.json and the visually reviewed prefix_400ms_fit.png. Actual fitting/replay source was 59e944b44; use that revision or explicit legacy 22-channel auditing when reproducing its saved raw logs.

Next experiment launched on DeskComputer: C:/Users/diete/SimscapeTour9921/prefix-500ms-quadratic-01, SSH handle 77987. Runtime is now pinned to 0715f95f3, including the qualified 27-channel model. Args: duration 0.5, basis quadratic, finite-difference-step 0.0001, max-nfev 8, transfer-report from completed 0.4 s. Same global t0, qualified q/qd, geometry and fixed attachments; no measured-state resets. Launcher: C:/Users/diete/SimscapeTour9921/launch-500ms.ps1. Inspect actual process/report/exit receipts before restarting; do not change the runtime while this experiment is active. New independent cold replay must require all 27 sensed efforts. Dependent validation is queued as SSH 87243, waiting for fitting launcher PID 39744 (identity checked against launch-500ms.ps1). Its saved cold-replay-launch.ps1 requires an explicit nonblank fit exit zero, completed report and exact runtime 0715f95f3 before invoking cold replay and the default 27-channel audit. Both launch scripts are copied into the run directory.

R2025b remains the sole required release and the active goal constraint. The goal tool cannot edit an existing objective; this entry and project/Repository Management guidance record the user's amendment without falsely completing or replacing the active goal. Full 1.814 s fitting, geometry refinement, feasible residual assessment and final qualification remain open. The fixed-attachment independent-body relaxation is 1.036723 mm over 0.4 s and 2.239689 mm over 0.5 s; these are conditional bounds, not connected-model IK or recalibrated-attachment bounds.

Packaging ownership update: the other capture task owns follow-up PR #9950, branch fix/9949-installed-capture, head 3b125b77a after PR #9946 merged. Do not duplicate its .gitignore/installed capture work.

## All 27 Actuator Channels Qualified in an Isolated Replay

SELF instruments Kinetically_Driven_Revolute_Joint.slx in R2025b: enable_revolute_actuator_log adds the native primitive effort output through an N\*m PS-Simulink converter to a thirteenth SignalBus element named ActuatorTorque. It preserves the existing twelve signals and is idempotent. This supplies TorsoInput, LEInput, LFInput, REInput and RFInput. The native helper test first failed on the missing function, then passed; its fixture also verifies migration from the legacy layout when the saved model is already instrumented. Four actuator-audit tests pass, including new channels, injected error and explicit missing legacy channels.

Independent replay of the refined 0.3 s candidate in the isolated instrumented checkout reproduces EVERY marker, time, q, qd and qdd sample exactly (all maximum differences zero). All 27 sensed channels match the requested polynomials: force error 2.274e-13 N, torque error zero, no unlogged coordinates. Final native tests, replay and both audit modes have explicit exit code zero. The first integration attempt omitted the copied capture payload; its failed log/receipt is preserved and the corrected replay passed. This verifies instrumentation on this candidate, not full-swing/physiological qualification.

The audit helper still reports absent legacy revolute sensors as unlogged. audit_saved_golf_candidate now defaults to REQUIRE 27 channels; explicitly pass 22 for older saved fits. Both paths passed native validation. Model SHA before 90b2d0bb06af0758300a498cc919857cd717c8d22aa99644feab60376ec1c9ab; after deb039b2eebf00001cac8097a79d7dbf0dee4a9d2ea0cc883caa80941e044512. The top GolfSwing3D_Kinetic model was not changed.

Archive actuator-instrumentation-01-bundle.zip is saved under remote C:/Users/diete/SimscapeTour9921 and local C:/Users/diete/Repositories/simscape-tour-checkpoints. SHA256 b5ae16b4d4901d4d12765c64196efe38d051a19ca1702838f7c323b37525b025; ZIP integrity and all 36 file digests verified. It includes before/after model files, helpers/tests, red/green logs, original and instrumented raw MAT data, explicit legacy/current audit samples and state parity. actuator_instrumentation_bundle_receipt.json and all_actuator_audit_r2025b.json preserve provenance/results.

Isolated native checkout: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-actuator-audit at 59e944b44 plus the owned instrumented model/helper/test patches. Its processes are terminal. The active 0.4 s fitting checkout is DIFFERENT and remains pinned at 59e944b44 with the OLD 22-channel model; do not change it while fitting SSH 42924 or queued validation SSH 71640 is live. Latest observed fit count 395, recent trials around 3.2 mm RMS. Future fits on the promoted model should require all 27 channels. PR #9948 remains draft. Another task owns packaging-only .gitignore exception f13701c5e under PR #9946; no packaging edits here.

## Historical 0.4-Second Launch Record (Now Terminal)

Current native runtime: exact 59e944b44, tracked clean before checkout; preserve untracked subsystem backups and stashes. Run: DeskComputer C:/Users/diete/SimscapeTour9921/prefix-400ms-quadratic-01. Args: --duration 0.4 --basis quadratic --finite-difference-step 0.0001 --max-nfev 8 --transfer-report <run>/transfer_source_report.json (copied refined 0.3 s result). The 81 transferred controls preserve the previous physical-time curve and satisfy bounds (normalized 0.730625–1.863038). Initial state/geometry/attachments and t0 are unchanged. This is the first native quadratic experiment; no result is claimed yet.

Confirmed live: fitting SSH 42924; remote PowerShell launcher 43780, Python launcher 42772 / worker 36996, explicit R2025b MATLAB 41308. Dependent cold replay/audit is queued: SSH 71640, PowerShell 39208. Both launch scripts and launch-arguments.json are saved in the run directory. New launchers use Start-Process -Wait -PassThru, reject null ExitCode and write integer receipts. The validation gate uses TryParse and rejects blank/noninteger/nonzero fit receipts. process-id.txt and cold-process-id.txt are written after completion; use the recorded live launcher/child identities while running. Do not duplicate either process or update the runtime. Poll existing handles or verify actual processes before deciding anything stopped.

Draft PR #9948 remains open. Next collect fit and already queued independent validation, plot the full polynomial effort range, and archive before extending the horizon. Refined 0.3 s evidence and corrected historical exit metadata are published in 59e944b44. Full-swing/attachment/all-actuator qualification remains open.

## Refined 0.3-Second Candidate Verified

SELF preserves prefix-300ms-linear-refine-02: 499 evaluations, finite-difference step 0.0001, max_nfev 10, final RMS 1.695470 mm (previous 2.056357), p95 3.834779 mm, maximum 7.922777 mm. Optimizer convergence/acceptance remain false. Exact runtime 31595aba9, unchanged qualified q/qd/geometry/attachments. Independent cold replay has zero marker difference, q error 7.247e-13 and qd error 3.986e-14. All 22 sensed efforts pass: maximum force error 2.274e-13 N, angular error zero. Five coordinates remain unsensed. The cold replay/audit process has explicit exit receipt zero. Both fit and validation processes are terminal; sessions 99257 and 47374 are closed. The plot was visually checked.

Important receipt correction: exit-code.txt is EMPTY for the 0.1 s, original 0.2 s, original 0.3 s and refined 0.3 s fitting processes. Their process exit codes are unknown; earlier zero attributions were unsupported. All corresponding independent cold receipts explicitly contain zero, and marker-parity evidence remains valid. The three earlier bundle receipt JSON files now use native_fit_exit_code=null with a correction note; original archives/hashes are unchanged. Never cast blank text to [int] and call it success. The queued validation's old cast allowed blank through; its completed report/source checks and actual independent replay still passed. Future launchers must use Start-Process -Wait -PassThru and reject a null exit code; future receipt gates must reject empty/noninteger text explicitly. Short nonzero-exit probes returned 7 under old, Wait and retained-handle forms, so the long-run empty-receipt cause is not yet isolated.

Verified 128-file archive: prefix-300ms-linear-refine-02-bundle.zip in remote C:/Users/diete/SimscapeTour9921 and local C:/Users/diete/Repositories/simscape-tour-checkpoints. SHA256 ddead3176156d826d16f20c248e0a98208bfb501c6bc909ce83bd9ec5ce30aa7. prefix_300ms_refined_bundle_receipt.json records provenance and per-file digests; fit/cold/audit JSON and PNG are stored beside it. Plot implementation c46b78c7d; actual native fitting/replay source 31595aba9.

Draft PR #9948 is open: https://github.com/D-sorganization/UpstreamDrift/pull/9948, labeled agent:codex. Epic #9921 remains open; full-swing qualification is incomplete. Next advance the completed remote runtime to published current code, then launch fresh prefix-400ms-quadratic-01 with --duration 0.4 --basis quadratic --finite-difference-step 0.0001 --max-nfev 8 and the refined report as --transfer-report. Use explicit R2025b engine environment, preserve prior runs/backups/stashes, and record a real integer process exit. No 0.4 s experiment has launched at this commit.

## Dependent Cold Replay Queued

The fit remains live: SSH 99257, Python launcher 32420 / worker 33436, R2025b MATLAB 15096. Last confirmed 430 evaluations. A dependent validation process is now confirmed live as remote PowerShell 31188, SSH session 47374. It runs C:/Users/diete/SimscapeTour9921/prefix-300ms-linear-refine-02/cold-replay-launch.ps1. This waits for the exact fit process, requires exit-code.txt=0 and a completed report, asserts runtime commit 31595aba9, then starts one fresh explicit R2025b process running qualified_candidate_replay followed by audit_saved_golf_candidate. It refuses to overwrite an existing cold report. cold-process-id.txt and cold-exit-code.txt will record native validation identity/outcome. No cold result exists at this observation. Poll these existing handles; do not start another replay or alter the runtime while either stage is live.

Published implementation is 2f3e0299e; normal push checks passed. Next collect both terminal results, plot and archive the refined run, then extend toward 0.4 s with the prepared polynomial transfer. Preserve prior evidence. A full-capture attachment recalibration will require replacing the runner's current exact t0 marker coincidence assumption with an explicit model-versus-target initialization residual while retaining exact q/qd qualification; do not hide that residual by recalibrating attachments each trial.

## Tested Rigidity Diagnostic

SELF extracts motion_matching.rigidity.rigid_attachment_residuals using the existing Kabsch rotation helper. Contracts reject malformed offsets/body assignments, infinities, partially missing points and wholly unobserved inputs; whole missing points stay NaN. Seven tests first failed on the absent module and now pass known independent rigid motions, a stretched pair with exactly 1 m residual, reflection rejection and missing-data behavior. All 89 relevant tests plus scoped Ruff/mypy pass. The generic reproduction/rigidity_floor.py accepts explicit --repo --state --capture --output and repeated --prefix-end arguments; it records source/payload/state/implementation hashes and refuses to overwrite output. It reproduces the earlier result in native_evidence/fixed_attachment_rigidity_floor.json: full capture 17.393050 mm RMS, current Hub attachments 34.423066 mm. This remains a relaxation conditional on fixed attachments, not native dynamics or an anatomical lower bound after recalibration.

The active refinement is still on exact 31595aba9 / SSH 99257 / MATLAB 15096. Last observed 330 evaluations, recent non-perturbed-scale trials around 1.70 mm RMS; no terminal result. Preserve this runtime until fit and independent replay complete. Next collect, cold-replay, audit, archive, then continue to a longer prefix with quadratic freedom if justified. The long-horizon attachment limitation must be addressed separately from effort optimization.

## Exploratory Full-Capture Rigidity Floor

A read-only local probe reused body_part_viz.fitters.\_kabsch.kabsch_rotation to fit each body's current fixed offsets independently at each capture frame, relaxing all joint connectivity and dynamics. It used the qualified seed and the exact driver payload, all 654 frames/25 selected markers, and the valid-point mask. The conditional full-capture lower bound is 17.393050 mm RMS; Hub (three back plus three head markers) contributes 34.423066 mm body RMS. Prefix bounds: 0.3 s 0.453254 mm, 0.6 s 4.065022 mm, 1.0 s 13.790908 mm. This is exploratory evidence for CURRENT fixed attachments, not an anatomical bound after recalibration and not forward-dynamics qualification. Single-marker bodies have zero relaxed residual and do not establish kinematic identifiability. Full-swing work must revisit multiframe attachment calibration and explicitly report the rigid head/torso limitation instead of assuming a universal 5 mm target.

Probe and result are saved locally in C:/Users/diete/Repositories/simscape-tour-checkpoints/fixed_attachment_rigidity_floor_probe.py and fixed_attachment_rigidity_floor.json. The probe is scratch analysis with this machine's paths, not a tested reusable product API. Next extract a tested reusable diagnostic before treating this as acceptance evidence. Native refinement is still live on 31595aba9 / SSH 99257 / MATLAB 15096; last observed 215 evaluations, recent trials near 1.74 mm RMS. No terminal result yet. Published effort-range implementation is c46b78c7d; do not change the live runtime.

## Interior Effort Peaks and Profile Plots

SELF adds prefix_fit.bernstein_effort_range: min/max from endpoints and real interior derivative roots on normalized time, reusing the existing native coefficient converter. Two new regressions failed before implementation and now pass (interior peak versus control bound, off-grid stationary point, constant endpoints); the expanded 82-test selection plus scoped Ruff/mypy pass. plot_prefix_100ms.py now supports all seven control counts, draws full-profile min/max and start/end markers, and reports force_controls_N, force_ranges_N, max_abs_torque_Nm and the separate max_abs_torque_control_bound_Nm. These are numerical polynomial extrema, not physiological qualification. The saved 0.3 s linear plot was regenerated and visually checked; earlier archived evidence was not modified. When copying the plotting script outside the repository, pass --repo explicitly.

The smaller-step native experiment remains live on unchanged 31595aba9, SSH 99257 / MATLAB 15096; 160 evaluations were confirmed, with recent trials near 1.76 mm RMS. No final result or new cold replay exists yet. Do not update the active runtime. Next collect its terminal result, cold replay and 22-channel actuator audit, archive it, then decide whether to extend the linear prefix or add quadratic freedom. All progress remains provisional until native replay evidence qualifies it.

## Higher-Degree Candidate Transfer Prepared

SELF extends transfer_prefix_candidate through native polynomial degree six. De Casteljau interval transformation preserves physical-time efforts when the basis horizon changes, and degree elevation adds controls without changing the initial curve. All prior identity, units, no-downgrade and bounds checks remain. Six selected new regressions failed before implementation; the expanded 28-case table checks every supported nondecreasing degree pair at 17 physical times with deterministic distinct joint controls. All 80 state/checkpoint/prefix/CLI tests pass; scoped Ruff and mypy pass. first_prefix_fit.py accepts constant, linear, quadratic, cubic, quartic, quintic or sextic through the shared identity validation; nonconstant bases use the prefix duration. This is Python-level qualification, not a native higher-degree fit result. The plot script still explicitly supports constant/linear only; extend its effort visualization before reporting a higher-degree run so interior extrema are not mistaken for endpoints.

The live smaller-step linear run remains on exact 31595aba9 and MUST NOT receive these changes while active. Last saved report contained 60 evaluations and status running; MATLAB process 15096 is responding and accumulating CPU. SSH session 99257 remains the authoritative wait handle. Next finish/refine/cold-replay that run; choose quadratic freedom only after inspecting its outcome. No higher-degree experiment has launched.

## Smaller-Step Refinement Running

SELF exposes --finite-difference-step on first_prefix_fit.py (default 0.001 preserved), validates finite positive input before loading capture or MATLAB, and records both step and max_nfev in optimizer_options. Five CLI tests failed first and now pass; the complete state/checkpoint/prefix/CLI selection passes 52 tests. Launched fresh prefix-300ms-linear-refine-02 on DeskComputer with --duration 0.3 --basis linear --finite-difference-step 0.0001 --max-nfev 10 and --transfer-report from the completed prefix-300ms-linear-01 report. Same-horizon transfer is identity checked and starts a new objective history; it uses the cold-verified nominal final candidate. Live execution: SSH session 99257, remote Python process 32420 (confirmed with Get-Process). Exact remote checkout 31595aba9; normal commit and push checks passed. Run directory C:/Users/diete/SimscapeTour9921/prefix-300ms-linear-refine-02 stores launch-arguments.json, process-id.txt, fit.log and fit.err; exit-code.txt is written only after terminal completion. Poll this handle or inspect the actual process before any restart. At this checkpoint the engine is starting; no result is claimed. Keep immutable snapshots every five evaluations, then independently cold-replay and audit the result. Earlier evidence remains intact.

## Root-Effort Frames and 0.3-Second Continuation

SELF expands the actuator audit to 22 sensed channels: 19 angular torques and three root forces. The six root channels already existed under HipLogs.TranslationForce[XYZ]Input / HipTorque[XYZ]Input. Source tracing proves they come from Bushing Joint sensing outputs, not upstream command taps; root_effort_sensor_trace.json includes connections and the unchanged model SHA. Five coordinates (TorsoInput, LEInput, LFInput, REInput, RFInput) still lack scalar actuator sensing.

Important frame contract: TranslationInput[XYZ] polynomial rows are WORLD force components. The model multiplies them by RotationMatrixGlobaltoHipJointBase before actuation. The sensed root forces and sliding joint q/qd are in HIP-BASE axes. With PlaneTilt=30 degrees, the required map is transpose(Rx(PlaneTilt)); angular hip actuator torques are already in primitive axes. Do not remove this native transform or treat world-force coefficients as directly conjugate to base-frame sliding velocities. The initial extended audit failed by 381.557 N because it compared different frames; preserve root_effort_audit_200ms_unqualified_r2025b.json as that comparison failure, not a dynamics defect.

The corrected audit requires a proper explicit force-frame rotation and all three force coefficient rows. Schema golf-actuator-audit/2 carries unit-aware entry values, max_force_error_N and max_torque_error_Nm rather than mixing units. Three R2025b TestCase methods pass after the recorded red runs. The saved 0.2 s fit qualifies all 22 sensed channels: maximum force error 2.274e-13 N and torque error 0 Nm. Evidence: root_effort_audit_200ms_qualified_r2025b.json and root_effort_audit_tests_r2025b.json. audit_saved_golf_candidate reads the model's PlaneTilt and applies the established base rotation; do not infer a rotation by fitting to measured forces.

The 0.3 s linear experiment, independent cold replay and frame-aware actuator audit have all completed with exit zero. Fit/replay used exact published runtime d344b726d with no patches; the completed runtime was then advanced to 78b0feef4 for the audit. No native fit or replay remains live. Directory: DeskComputer C:/Users/diete/SimscapeTour9921/prefix-300ms-linear-01. Final RMS is 2.056357 mm, p95 4.590569 mm and maximum 9.323188 mm across 109 samples/25 markers; 173 evaluations, optimizer not converged. Cold marker difference is exactly zero, initial q error 7.247e-13 and qd error 3.986e-14. All 22 sensed efforts qualify: force error 1.137e-13 N, torque error 0 Nm. Five coordinates remain unlogged. The plot was visually inspected.

Incremental recovery point: prefix-300ms-linear-01-bundle.zip exists in remote C:/Users/diete/SimscapeTour9921 and local C:/Users/diete/Repositories/simscape-tour-checkpoints. Matching SHA256 is 94c3062b34d4292ea44fcfba78696a606e1e2e315be4079a2b557f50d277cdd3; ZIP integrity and 61 file digests were verified. The archive contains raw fit/cold MAT files, immutable optimizer candidate checkpoints, capture payload, qualified initial state, actual runner and dependencies, transferred source report and native logs. prefix_300ms_linear_bundle_receipt.json records provenance and all digests. Git stores the fit, cold replay, audit JSON and viewed PNG beside that receipt.

Next refine this SAME 0.3 s horizon with a smaller finite_difference_step before deciding whether higher polynomial order is needed. The shared optimizer supports it, but first_prefix_fit.py still hardcodes 0.001; expose and record a validated CLI option with tests. Use a fresh run directory and the saved identity-checked --checkpoint (candidate only, not optimizer internal state). Keep the qualified initial state, geometry, marker attachments and global t0 fixed. Large finite-difference trials reached about 40 mm while the final residual is 2 mm; this motivates a numerical refinement experiment, not a claim of exhausted linear model capacity. Continue with longer prefixes only after reviewing the refinement.

DL-#9921 is refreshed. Branch feat/9921-simscape-tour-matching; epic #9921 / issue #9927; PR not created. R2025b remains the only required release. Earlier archives, runtime subsystem backups and stashes remain intact; the full-swing goal remains open.

## Explicit Actuator-Log Audit

SELF adds audit_golf_actuator_torques.m, a read-only comparison of native A–G polynomials with the 16 existing scalar ActuatorTorque channels at their own native timestamps. It rejects missing/nonfinite logs, checks coefficient order/shape, and leaves eleven coordinates explicitly unlogged; no reaction torque is substituted. Two R2025b TestCase methods first failed on the missing helper, then passed native-clock polynomial comparison, injected-error detection and missing-actuator rejection. The saved 0.1 s raw run has zero error on all sixteen logged channels. Evidence: actuator_audit_100ms_r2025b.json and actuator_audit_tests_r2025b.json. Reproduce with audit_saved_golf_candidate(repo,run_dir). This does not qualify the unlogged root/torso/elbow/forearm efforts.

The 0.2 s experiment and its independent cold replay/audit have now completed with exit zero. Both constant and varying profiles are verified on the sixteen logged channels. The read-only audit process never changed the fitting checkout. The plot script supports constant/linear endpoints and derives its title/horizon from the report. Native tests and the known 0.1 s plot regression remain valid; the final 0.2 s plot was also rendered and visually inspected. DL-#9921 is refreshed.

## Linear-Torque 0.2-Second Fit Verified

SELF preserves the completed 0–0.2 s native R2025b experiment: 73 samples and 25 fixed markers, with continuous linear torques represented in native A–G coefficients. Initial q/qd, geometry (14.5,12) inches and fixed body offsets are unchanged. The transferred 0.1 s candidate starts at 6.108022 mm RMS; after 389 evaluations the result is 0.859647 mm RMS, 1.873274 mm p95 and 3.575221 mm maximum. The first 0.1 s remains at 0.367566 mm RMS (previous fit 0.356491 mm); the new second interval is 1.166042 mm RMS. No intermediate measured states are injected. The optimizer reaches max_nfev=8, so optimizer_converged and accepted_numerically remain false. Full-swing/anatomical/physiological qualification is still open.

A fresh cold R2025b replay reproduces every final marker coordinate exactly (0 m maximum difference), q within 7.247e-13 and qd within 3.986e-14. Fast Restart is off. The sixteen available actuator-torque logs match their varying polynomials with 0 Nm maximum error; eleven coordinates remain unlogged. Maximum angular endpoint magnitude is 172.602175 Nm and maximum endpoint change is 10.160988 Nm. Source evidence: prefix_200ms_linear_fit_r2025b.json, prefix_200ms_linear_cold_replay_r2025b.json, actuator_audit_200ms_linear_r2025b.json and the viewed prefix_200ms_linear_fit_r2025b.png.

Both native processes exited zero. No fit/replay remains live. Completed directory: DeskComputer C:/Users/diete/SimscapeTour9921/prefix-200ms-linear-01. Runtime remains a58c866ca plus tour_fit_state.py / prefix_fit.py from 6875566d5; audit implementation is f91920f0f in remote scratch. The exact scripts, capture, state, transferred source report, raw fit/cold MAT files, logs and all checkpoints are in prefix-200ms-linear-01-bundle.zip under remote SimscapeTour9921 and local C:/Users/diete/Repositories/simscape-tour-checkpoints. Both archive hashes equal 32a491e2a14631c950ae4633edbad860fe917c90954cfd6839fa1e7f046bdfba; archive integrity and 104 individual file hashes were verified. prefix_200ms_linear_bundle_receipt.json records paths and provenance. Preserve earlier bundles, the three subsystem backups, and all stashes, including 'preserve 200ms linear source patches before a58c866ca checkout'. The corresponding file backup is remote linear-source-precheckout.

Next run a fresh 0.3 s linear-prefix experiment with --transfer-report pointing to this completed run's first_prefix_fit.json, --basis linear --duration 0.3, and the same qualified --initial-state. Transfer already passes the bounds check (normalized parameters 0.766201–1.863011). Use a fresh run directory, retain SSH through process completion, save every five evaluations, then cold-replay and audit. Expand polynomial freedom when the growing-prefix residual demonstrates the need. Exact --checkpoint remains for the same identity/horizon only. Do not change the global t0 or insert intermediate states. The runner/converter/transfer contracts have 47 passing Python tests; the actuator audit has two passing R2025b TestCase methods.

Governing epic #9921 / issue #9927; branch feat/9921-simscape-tour-matching; PR not created. DL-#9921 is refreshed. Working directory C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour. R2025b is the only required release. The empty design-manual calculation inventory remains an independent pre-existing release blocker.

## Qualified-State 0.1-Second Fit and Cold Replay

SELF preserves the completed qualified-state 0–0.1 s torque fit: 37 samples, 25 fixed markers, geometry (14.5,12) inches and the saved nonzero q/qd. Across 283 native evaluations, zero-effort RMS 22.540848 mm falls to 0.356491 mm; p95 is 0.827594 mm and maximum 1.591758 mm. The optimizer reached max_nfev=10, so optimizer_converged and accepted_numerically remain false. These constant degree-six Bernstein controls are an exploratory prefix result, not a full swing or anatomical identification. Maximum angular effort is 171.674 Nm; vertical root force is 761.513 N. Effort bounds remain exploratory, not physiological qualification.

A fresh R2025b cold replay reproduces every final marker coordinate exactly (maximum difference 0 m); initial q error is 7.247e-13 and qd error 3.986e-14. Both native processes exited zero, Fast Restart is off for the cold replay, and no fit/replay remains live. Saved results: native_evidence/prefix_100ms_fit_r2025b.json, prefix_100ms_cold_replay_r2025b.json and prefix_100ms_fit_r2025b.png. The plot was rendered and visually inspected. Reproduction: first_prefix_fit.py (implementation 7f4ab9d5f), qualified_candidate_replay.m and plot_prefix_100ms.py in native_evidence/reproduction. Test receipt qualified_state_contract_tests.json records the missing-module red run and 34 passing state/checkpoint/prefix tests; scoped Ruff and mypy pass.

Completed run directory: DeskComputer C:/Users/diete/SimscapeTour9921/prefix-100ms-01. Runtime remains detached at ba18f2e7b with tour_fit_state.py copied from 7f4ab9d5f. Do not discard the three existing subsystem backups or preserved stashes. The bundle includes actual scripts, state, capture, final raw MAT, independent cold raw MAT, logs and all checkpoints. Remote C:/Users/diete/SimscapeTour9921/prefix-100ms-01-bundle.zip and local C:/Users/diete/Repositories/simscape-tour-checkpoints/prefix-100ms-01-bundle.zip both hash to e673a2a9db8772c3718ae8cefaa049b224a5830f3599cae43ad7b5a910c1fa1a. Archive integrity and all 77 per-file hashes were checked; prefix_100ms_bundle_receipt.json records them. The original default-state 50 ms experiment remains intact.

Next implement explicit candidate transfer to a longer prefix (start 0.2 s), retaining the qualified t0 state and fixed attachments, and unlock continuous nonconstant polynomial efforts. --checkpoint currently requires exactly the same horizon and identity; do not bypass it or mislabel changed-horizon costs as resumed history. A new experiment may explicitly reuse torque values while starting a new report. The runner requires --initial-state, --duration and a fresh run directory; it saves every five evaluations and verifies native initialization before fitting. DL-#9921 is refreshed. Branch feat/9921-simscape-tour-matching, epic #9921, active issue #9927, PR not created. Working directory C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour. R2025b is the only required release. Full-swing fitting, anatomical calibration, all-actuator/physiological qualification and solver refinement remain open. Existing design-manual governance has an empty calculation inventory and continues to block release independently.

## Checkpoint Identity Guard

SELF adds exact comparison of the optional fit_identity manifest when loading a candidate. This prevents a declared initial state, geometry, attachment set, joint order, basis or horizon from being changed or omitted during resume. Legacy snapshots remain readable against legacy reports only. The new regression failed before implementation and all four checkpoint tests now pass; scoped Ruff passes. The runner has not yet been extended to populate this manifest or consume the qualified velocity seed, so the next action remains implementing that runner extension and executing the 0–0.1 s fit in R2025b. No new native fit was launched in this checkpoint.

Branch: feat/9921-simscape-tour-matching; PR: not created; epic: #9921; active issue: #9927. Working directory: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour. Previous published checkpoint is 77633eeb9. Qualified state and archive paths below remain valid. Development-log entry DL-#9921 is refreshed. Validation: python -m pytest tests/engines/simscape/test_tour_checkpoints.py -q --no-cov (4 passed); scoped ruff check and format pass. Design-manual governance passes its structural check but release remains blocked by the pre-existing empty calculation inventory. Preserve the user-owned main checkout and remote subsystem backups/stashes.

## Qualified Initial Velocity and Session Targets

SELF fixes the nonzero initial-velocity regression with configure_capture_velocity_targets.m. Load the model, keep its returned onCleanup guard alive for the entire fit, and clear it when the fit ends. It enables the schema's independent velocity targets, leaves dependent rates to the native constraints, releases Fast Restart at session boundaries, and restores original mask selectors and primitive flags on normal completion or error. It does not save the source model. The sole forward wrapper and default simulation options remain unchanged.

Two R2025b TestCase methods pass, covering cold/warm/warm/cold replays and exception cleanup. The saved 50 ms marker replay now has initial_state_verified=true and status=initial-state-qualified: q error 7.247e-13, qd error 3.986e-14 and initial marker error 6.686e-13 m. Geometry remains (14.5,12) inches, with single-initial-frame offsets; the seven-frame velocity residual is 4.225 mm/s. This qualifies initialization, not anatomical geometry or a complete swing.

The rejected per-call SimulationInput mask-override prototype achieved the requested rates but failed restoration and emitted non-tunable Fast Restart warnings. Its archive is C:/Users/diete/Repositories/simscape-tour-checkpoints/transient-velocity-policy.zip, copied to C:/Users/diete/SimscapeTour9921/transient-velocity-policy.zip. The prototype helper was removed from both current source trees. Do not recreate a per-call independent_velocity_targets option; use the session guard. The first_prefix_fit.py reproduction runner now creates the guard and clears it in finally, then closes its private model without saving.

Qualified evidence is native_evidence/initial_velocity_seed_qualified_r2025b.json and capture_initial_velocity_session_green_r2025b.json. Earlier unqualified reports remain failure evidence. Completed run: C:/Users/diete/SimscapeTour9921/velocity-seed-02, with raw MAT, capture payload, helper, probe, tests and log in velocity-seed-02-bundle.zip. Both local and remote archive hashes match 798e83a1f5916e9004d29ec9d0fef9e08c9266f2c8ccd7a5d0aa41f96af081ed; qualified_velocity_bundle_receipt.json records paths and per-file hashes. MATLAB exited zero; no native process remains live. Runtime is 81a8086d9 with the new helper copied into shared (ignored until tracked by this implementation); the default/forward/capture-option files were restored to published content. Preserve existing subsystem backups and stashes.

Next extend the fitting runner to consume the qualified q/qd, geometry and fixed offsets and to accept a longer horizon (start with 0–0.1 s). The existing runner still constructs default geometry and zero initial rates; --checkpoint alone only supplies torque parameters. Record the initial-state/attachment identity and joint order in resume checks before allowing changed-state runs. Keep the earlier verified torque checkpoint intact, fit each prefix from the same t0 state, and progressively allow continuous nonconstant polynomial efforts. Full-swing qualification remains open. R2025b is the only required release.

## Initial-Velocity Mapping and Failing Regression

SELF adds golf_marker_velocity_map.m. It resolves native velocity identities by block path/primitive, uses explicit m/s and rad/s units, and maps 21 independent rates into all 27 joint rates plus fixed-marker world velocities. Frame angular velocities use follower axes and frame linear velocities use world axes. The native TestCase first failed on the missing function, then passed a hip-angular column comparison against centered native pose differencing and exact independent-rate identity. The dependent-coordinate list is now shared in golf_kinematic_schema.json; fit_golf_pose_seed uses it and its R2025b regression also passes.

The best geometry seed's first seven capture frames (0–1/60 s) give target marker-velocity RMS 0.0206622 m/s. A damped least-squares tangent fit (damping 0.001; rank 21) reduces residual RMS to 0.00422497 m/s. Offsets remain calibrated only at t0 and geometry remains provisional. IMPORTANT: its forward initialization FAILED acceptance. q and initial markers match within about 7e-13, but maximum qd mismatch is 0.0819077 rad/s. TorsoInput requests 0.002033743 rad/s but starts at zero; LWInputY requests -0.081907693 rad/s but starts at zero; RWInputY shifts from -0.024958902 to 0.056948791 rad/s. Other requested rates match. Do not classify the report's status=replayed as accepted.

Inspection of the source SLX confirms Torso Revolute Joint RzVelocityTargetPriority=None and both wrist Universal Joint RyVelocityTargetPriority=None. Their initial velocity value bindings are correct; the targets are disabled. Next add a per-simulation initialization policy through simulate_with_coefficients/SimulationInput: explicitly enable the schema's independent velocity targets and leave dependent targets unconstrained. Use mask priority selectors (their initialization callbacks are already repaired and tested), verify any direct hip-joint Specify/priority settings, and test restoration of source defaults. The failing nonzero-qdot replay is the red regression; do not weaken its 1e-8 acceptance threshold or overwrite source model defaults to hide the issue.

Evidence: marker_velocity_red_r2025b.json, marker_velocity_green_r2025b.json, pose_dependency_green_r2025b.json, initial_velocity_seed_unqualified_r2025b.json and initial_velocity_mismatch_audit.json. Reproduction: native_evidence/reproduction/initial_velocity_seed.m, with a fresh output directory containing driver_marker_payload.json from the saved bundle. Failed raw replay is C:/Users/diete/SimscapeTour9921/velocity-seed-01/initial_velocity_seed.mat, copied locally to C:/Users/diete/Repositories/simscape-tour-checkpoints/initial_velocity_seed_unqualified.mat. SHA256 matches on both machines: 824355037eb045a631e7635121c8496f65f635b31b44eee2c0bcf8941ea4fee5. The failed MATLAB process exited 1; the two qualification tests exited 0. No native process remains live.

Runtime is 317cc5d8d with copied schema and fit_golf_pose_seed changes in its shared directory; golf_marker_velocity_map.m is in remote scratch. Preserve those owned patches before checkout. R2025b remains mandatory; the full torque-driven swing remains open. Once initial velocity is verified, refit and extend prefixes with continuous polynomial efforts from the same t0 state.

## Changed-Geometry Forward Replay Verified

SELF records successful R2025b forward replay of the (14.5,12)-inch geometry seed for 0–0.05 s with zero polynomial efforts and Fast Restart off. Initial q agrees with the fitted pose within 7.247e-13; initial native-snapshot positions agree within 6.378e-13 m. Fifteen logged frame positions agree with the changed-geometry snapshot within 2.625e-13 m at 0, 0.025 and 0.05 s. Hip has no qualified direct position source and is omitted from that logged-frame comparison; its zero entry in the error array is not an independent Hip check. Native logged arm lengths are 0.3683 m per upper arm and 0.1524 m per forearm half, proving the requested geometry reaches the forward model. MATLAB exited zero, original workspace objects were restored and no model file was saved.

Evidence: native_evidence/geometry_candidate_replay_r2025b.json; reproduction: native_evidence/reproduction/geometry_candidate_replay.m, called with repository and output directory. Raw replay is C:/Users/diete/SimscapeTour9921/geometry_candidate_replay.mat. Runtime is 317cc5d8d; no native process remains live. This validates geometry/state application only, not anatomical identification or a torque-fitted swing. The earlier default-geometry torque checkpoint remains intact. Next estimate native constraint-consistent initial velocities and compare fixed-attachment fits over longer prefixes, retaining the geometry-identification limitations. R2025b remains mandatory.

## Bounded Starting-Pose Geometry Sweep

SELF adds fit_golf_pose_seed.m, extracting the existing pose objective into a reusable native helper with finite frame/coordinate contracts and strict final target/constraint acceptance. The R2025b TestCase first failed on the missing function, then passed recovery of a known translated native pose; the final test discovers its versioned fixture from the repository, without environment configuration. Evidence: pose_fit_red_r2025b.json and pose_fit_green_r2025b.json.

Nine fixed arm-length pairs were fitted at the initial pose using the same nine surface proxies and weak pose prior. All nine returned native status 1 and optimizer exitflag 3. Default (12,14) inches gave Euclidean proxy RMS 54.471 mm; best tested (14.5,12) gave 49.343 mm. This is a coarse single-pose seed, not anatomical calibration: the upper-arm optimum is at the tested boundary, surface proxies differ from joint centers, and no changed-length forward replay has yet been qualified. Keep the verified default-geometry torque checkpoint intact. Next constrain attachments/anatomy using multiple frames, estimate consistent initial velocities, then compare candidate geometry in the authoritative forward dynamics and grow polynomial-torque prefixes.

Runtime is 5309fb140 with fit_golf_pose_seed.m loaded from C:/Users/diete/SimscapeTour9921. The report's source_revision=9a381f018 denotes the unchanged physics base; geometry_sweep_bundle_receipt.json records the actual runtime and source hashes. Completed run: C:/Users/diete/SimscapeTour9921/geometry-sweep-02; MATLAB exited zero and no run remains live. Each candidate saved a separate numbered checkpoint. The earlier root-directory sweep rejected malformed target-array input before fitting; preserve it as failure evidence, not as geometry results. The corrected sweep stops on programming errors and rejects only explicit native constraint failures. Original workspace objects were restored and the model files remain unchanged.

A verified continuation bundle is stored at C:/Users/diete/SimscapeTour9921/geometry-sweep-02-bundle.zip and C:/Users/diete/Repositories/simscape-tour-checkpoints/geometry-sweep-02-bundle.zip; SHA256 ca61cfbe43a0800f04bb53190a2b03da25584213ddf37975141f306172ece64a. It includes input pose/targets, helper, test, sweep script, logs and all nine checkpoints. Numerical result: native_evidence/native_geometry_seed_sweep_r2025b.json. Reproduction: native_evidence/reproduction/native_geometry_seed_sweep.m, called with repository and a fresh run directory containing the two input JSON files from the bundle. R2025b remains the only required release.

## Native Arm-Length Bindings Verified

SELF records a successful R2025b native geometry probe (MATLAB exit zero). Defaults are UpperArmLength=12 in and LowerArmLength=14 in. Changing LeftUpperArmLength and RightUpperArmLength from 12 to 13 leaves all six measured arm-segment distances unchanged. Changing the shared UpperArmLength to 13 and LowerArmLength to 15 increases each upper arm by 0.0254 m and each forearm half by 0.0127 m. All three KinematicsSolver cases return status 1 with all independent targets satisfied. These are geometry/pose checks, not new forward-dynamics fits. Original model-workspace objects are restored; no model is saved. Native runtime remains at 9a381f018, and no probe process remains live.

Evidence: native_evidence/native_geometry_probe_r2025b.json; executable probe: native_evidence/reproduction/native_geometry_probe.m. The first snapshot took 38.51 s and subsequent changed-geometry snapshots about 9.9 s, making a small outer geometry search practical. Use the active shared parameters in inches, rebuild KinematicsSolver on each changed geometry, and apply identical overrides to the authoritative forward replay. Do not optimize the inactive aliases. Workspace values may be Simulink.Parameter objects; read their Value for numerical work and preserve their originals when restoring.

The surface-distance audit (native_evidence/arm_surface_distance_audit.json) covers all 654 frames: left/right elbow-to-wrist median distances are 0.2781/0.2817 m; these are surface-marker distances, not joint-center lengths. Right shoulder-back to elbow differs substantially from left shoulder-top to elbow, so do not treat those labels as symmetric joint centers. Next perform bounded geometry/attachment calibration with explicit anatomical priors and estimate a constraint-consistent initial velocity, then refit longer prefixes. Full-swing qualification remains outstanding.

## Independent Candidate Replay

The saved 25-marker, 0�0.05 s candidate passed a fresh R2025b process on DeskComputer: maximum coordinate difference from the saved fitted prediction is exactly 0 m; initial q differs from its seed by at most 1.4094098665928811e-9. Fast Restart was off; MATLAB exited zero. This verifies repeatability of this candidate only, not full-swing fidelity or physiology. Native source is 9a381f018c2fe96b1b36c23a3c7d8c5aaeb3f74e. Runtime checkout now points to that commit; its prior prefix-fit patch is preserved in a named stash, and the three .slx.r2025a backups remain untouched. No native fit/replay process remains live.

Evidence: native_evidence/cold_candidate_replay_r2025b.json. Reproduction: native_evidence/reproduction/cold_candidate_replay.m and cold_candidate_input.json (fixed coefficients, state, attachments, clock and expected marker array). The original raw replay is C:/Users/diete/SimscapeTour9921/cold_candidate_replay.mat. The first probe stopped before simulation on the unloaded-model precondition; the saved probe explicitly loads the model before constructing KinematicsSolver. Next calibrate active lengths and consistent initial velocity, then extend prefixes from t0 with continuous polynomial efforts. R2025b remains the only required release.

## Latest Saved Result and Checkpoints

SELF records the first completed native torque fit: 25 fixed body-marker attachments over 0–0.05 s, 0.176671 mm RMS / 0.280906 mm p95 / 0.594141 mm max, after 283 native calls including baseline and final checks. The optimizer hit max_nfev=10 (281 calls counted in its stage), so accepted_numerically is false. Initial native q agrees with the waist-constrained seed within 1.410e-9. Default lengths and zero initial velocities remain provisional. No full swing or physiological effort qualification is claimed. The fit process exited zero; no fit/observer process remains live.

Read [Incremental Checkpoints and Replay](docs/development/simscape_tour_matching/CHECKPOINTS.md) from the repository root (or CHECKPOINTS.md beside REMOTE_EXECUTION.md). It gives exact paths, commands, limitations and ordered next actions. Source, inputs, exact original script, enhanced replay script and SHA256 manifest are preserved in first-prefix-50ms-20260910.zip on both machines: local C:/Users/diete/Repositories/simscape-tour-checkpoints/first-prefix-20260910 and remote C:/Users/diete/SimscapeTour9921/checkpoints/first-prefix-25. Receipt: native_evidence/first_prefix_bundle_receipt.json. The immutable evaluation-00283 snapshot in those directories contains the best warm-start parameters and complete history. These are candidate checkpoints, not serialized SciPy internals.

The tested tour_checkpoints.py helper rejects partial reports, preserves earlier snapshots and checks capture hash, marker order and effort scales for warm starts. The enhanced reproduction script automatically snapshots its existing five-evaluation reports and accepts --checkpoint. Three checkpoint tests, 23 prefix/capture tests, scoped Ruff/mypy and two R2025b native marker-adapter tests pass. The user explicitly requested incremental saving and breadcrumbs; retain this behavior for every subsequent run. Next validate a cold replay of the candidate, calibrate active UpperArmLength/LowerArmLength bindings (inches) and initial velocity, then grow the prefix with continuous polynomial inputs. Both native upper arms use UpperArmLength, not the left/right aliases. The full epic stays active.

## Current Native Fit Experiment

The two R2025b `test_simulate_golf_markers` tests pass after the missing-function red run. The adapter calls only `simulate_with_coefficients`, projects fixed offsets at native state timestamps, then interpolates Cartesian marker positions onto the requested clock. Invalid simulations, joint states, frame solves or coverage fail rather than becoming optimizer penalties. The shared Python prefix fitter now accepts a positive finite relative finite-difference step; twenty tests pass, including a quantized-oracle regression that stalls with the default step and converges with a resolved perturbation. Scoped Ruff and mypy pass.

DeskComputer's isolated `C:/Users/diete/SimscapeTour9921/python-r2025b` environment uses Python 3.12 and the engine installed from R2025b's local extern/engines/python directory. The original global R2024b engine remains untouched. The engine probe confirms MATLAB root C:/Program Files/MATLAB/R2025b. The waist-constrained pose seed in `native_pose_seed_waist.json` adds a Hip target from the four waist markers to the previous eight proxies: 54.478 mm RMS Euclidean proxy error, max 80.819 mm, hip proxy error 0.4 mm, native KinematicsSolver status 1 (all targets and physical constraints satisfied). This still uses provisional surface proxies; the initial forward replay now passes the q and fixed-offset marker assertions.

The first 22-marker experiment stopped on a forearm perturbation because canonical interpolated q at 1/120 s violated the closed chain (KinematicsSolver -1, only 9/27 targets satisfied). The corrected adapter extracts joint states on the original raw HipPositionX log clock, projects native poses, then interpolates marker positions. The stronger R2025b native test uses a 360 Hz output grid and 0.2 Nm forearm torque; both tests pass. The retry has 25 markers including all three head markers and has passed the previously failing perturbation. Baseline initial q matches the waist seed within 1.410e-9, and first-frame markers match their fixed-offset calibration within 1e-8 m. The completed experiment is `first_prefix_fit.py` in remote scratch, launched through the isolated Python environment with --repo pointing to the runtime checkout and --run-dir pointing to scratch. It fits 50 ms, 25 active markers, fixed offsets calibrated from the first frame, zero initial joint velocities and constant degree-six Bernstein controls. Parameters p in [0,2] map to effort=scale\*(p-1), with exploratory scales 1500 N for root translations and 200 Nm for angular joints; these are search bounds, not physiological qualification. Relative finite-difference step is 0.001, max_nfev is 10. Existing prefix_fit performs the optimization; MATLAB is the forward oracle. Report `first_prefix_fit.json` updates every five evaluations, logs are first_prefix_fit.log/.err. Inspect the existing process before restarting. Current files are copied patches over runtime commit 0d8b34004; do not checkout or modify its loaded source during the experiment.

The 22-marker selection is an early fitting stage, not full-capture coverage. RShoulderTop is missing initially; Marker_0 has unknown meaning; Uname\*36/37/38 are redundant waist derivatives; eight leg markers lack modeled legs. Contrary to an earlier working assumption, the native model DOES have a rigid Head/Neck body fixed to the Hub's rigid chain (SLX system_7475, Head216-Neck214-RigidTransform217-COMRod628-Hub7493). Include the three head markers as Hub-fixed attachments in the expanded objective; independent neck motion cannot be represented by the current 27-DOF topology. Do not classify head markers as absent geometry. Back markers are assigned to Hub's upper-torso body, shoulder skin markers to scapula bodies, upper-arm markers to shoulder bodies, wrist markers to forearms and club clusters to the rigid club. All attachment choices remain provisional pending multiframe geometry calibration.

Next: inspect the first native fit result, retain all source scripts and numerical evidence, then expand the marker set and time prefix. No full torque-driven tour swing has been qualified. R2025b remains mandatory.

## Current R2025b Marker-Frame Qualification

Working branch: `feat/9921-simscape-tour-matching`; implementation commit SELF (resolve with git rev-parse HEAD); PR not created. SELF extends the native frame schema to 16 frames and add intrinsic XYZ world-from-body rotations. Twelve orientations agree with native logged rotation matrices. `HipGlobalPosition` is the fixed hip-joint BASE position, not the moving hip follower: SLX system_7475 wires sensor 6266 through converter 6271 to that log. The Hip frame intentionally has no direct position source; its test uses measured sliding coordinates resolved through the model's PlaneTilt. Do not attach waist markers to HipGlobalPosition.

`simulate_with_coefficients` now explicitly disables Fast Restart when requested; its six workspace regressions pass in R2025b. A native zero/2/zero Nm probe reduced subsequent 20 ms runs from 55.88 s to 2.87/2.80 s and reproduced the first zero-torque joint trajectory exactly. Its successful result report does not imply clean process shutdown; the four-run cold-replay probe with explicit cleanup now succeeds with MATLAB exit zero, exact warm/cold joint trajectory equality and FastRestart off on the fourth run. Evidence: native_evidence/fast_restart_cold_probe.json.

Current validation: all seven R2025b tests pass: four marker-projection tests, two intrinsic XYZ tests and the expanded native frame replay (16 positions and 12 logged orientations at three times). Evidence: native_evidence/rotation_frames_green.json; MATLAB exits zero. The projector uses fixed body-local offsets in metres and rejects invalid body indices, attachment counts and improper rotation matrices. Red evidence was recorded before implementation. No capture marker calibration or torque optimization has completed.

Resume on DeskComputer using R2025b explicitly. DeskComputer runtime is now detached at validated commit `0d8b3400421c8c71807356eba4d9b7cdb273c33f`. The copied validation patches were preserved in a named stash before checkout; the three historical .slx.r2025a backups remain untracked and untouched. Local implementation is published and clean; this documentation checkpoint is SELF. Scratch is `C:/Users/diete/SimscapeTour9921`. The native frame and cold-replay probes are complete. Inspect JSON status and MATLAB exit separately when repeating them. Next calibrate fixed marker attachments and initial velocity before optimizing the first torque prefix through the sole simulation wrapper. Local design-manual governance passes but release remains blocked by the existing missing calculation inventory. The central development-log validator reports pre-existing duplicate/missing fields in other entries; DL-#9921 adds no reported errors. R2025b remains the sole required release. Repository Management policy PR #1633 is merged.

SELF adds build_golf_kinematics.m and golf_kinematic_schema.json: 27 coordinate identities resolved by block path/primitive and 15 qualified sensor frames. A fresh R2025b TestCase replay passes for all frames at three timestamps; independent saved-run parity is within 2.054e-13 m. Clubhead must use Transform Sensor10/F, not the adjacent sensor (127 mm offset). Kinematics solves take roughly 1.5-6 ms after construction. This enables pose calibration only; contacts, joint limits and torque dynamics require the authoritative forward replay. The provisional first-frame seed completed: eight surface proxies improve from 1.298 m to 69.635 mm RMS Euclidean distance (40.204 mm per-coordinate RMS), max proxy distance 106.547 mm. The optimizer ended with flag 2 and KinematicsSolver flag -1 (physical constraints satisfied, some requested joint targets missed); use the returned q, not optimizer x. A fresh R2025b 20 ms forward replay of returned q succeeds, reproducing joint components within 1.037e-12 and all frame positions within 9.722e-13 m. Evidence: native_pose_seed_r2025b.json, native_pose_replay_r2025b.json and initial_pose_seed.png. This is an initial-pose seed only; full marker attachments, geometry, initial velocities and continuous torque fitting remain required. Reference artifacts are at C:/Users/diete/Repositories/reference-fit-artifacts-9914; their surface-proxy fits are seeds, not native full-dynamics evidence.

SELF adds capture_fit_sim_options(duration_s), which builds on default_sim_options and keeps both killswitch values at one for the complete requested horizon. Three R2025b native tests pass after a recorded red run, including a 1.81 s wrapper replay across the old one-second cutoff and preservation of source workspace defaults. This is the required starting configuration for capture fitting; add fixed geometry and initial-state overrides to its input_overrides struct. The R2025b KinematicsSolver probe now succeeds on the exact model (73.15 s construction); include both src/model and genpath(src/functions), as in the normal runtime setup. Its joint-position/velocity table is recorded in native_evidence/kinematics_solver_probe_r2025b.json. Next use this solver to qualify body-frame outputs and initialize the capture pose; final acceptance remains a torque-driven forward replay. Full marker calibration and torque fitting remain outstanding.

R2025b actuation audit (SELF): the loaded GolfSwing3D_Kinetic model contains 13 physical joint blocks exposing 27 coordinates, all InputTorque/ComputedMotion. The stale readable main-model snapshot contains a kinematic RE definition absent from the loaded SLX; use the loaded model as authority. Sixteen scalar ActuatorTorque logs exist (scapula, shoulders, wrists and spine); the remaining 11 efforts need qualification. Default KillswitchStepTime is 1 s, shorter than the 1.814 s driver capture, so the fitting configuration must keep actuation enabled through the complete target. Evidence: native_evidence/actuation_audit_r2025b.json. The 2 Nm LSInputX probe succeeded in R2025b: all 21 actuator samples equal 2 Nm over 20 ms, with an explicit 2 s killswitch override (49.67 s wall time). Evidence: native_evidence/torque_probe_r2025b.json; raw replay remains in remote scratch torque_probe_sim_out.mat. This verifies one polynomial-to-actuator path, not all 27 efforts or a fitted swing. Next: qualify the remaining efforts, preserve actuation for the full horizon, and register native body markers and initial pose against the capture.

R2025b qualification: all 22 native tests pass on DeskComputer in MATLAB 25.2.0.3177638 (R2025b) Update 5. The actual 20 ms forward simulation succeeds with finite joint states and club/grip positions (95.89 s). Source revision: `4191ca3aca126216ea02e26e485ff72908f3ad2b`. Evidence: `native_evidence/native_r2025b_suite.json` and `native_evidence/baseline_r2025b.json`. Actuator-torque extraction and full-swing fitting remain unqualified.

## Required MATLAB Release

MATLAB R2025b is the required execution, model-save and validation release for the Simscape golf model and tour-average matching epic #9921. The user has the complete required licensed feature set in R2025b. R2026a is not a requirement; do not select it from PATH or use its successful probes as R2025b acceptance evidence. On DeskComputer and ControlTower launch `C:/Program Files/MATLAB/R2025b/bin/matlab.exe` explicitly. Preserve historical reports with their actual release; run acceptance checks in R2025b. The full forward-dynamics matching goal remains active with this constraint.

Epic #9921 and foundation #9924 are active in
`C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour`, branch
`feat/9921-simscape-tour-matching`, implementation commit
`62c7f845ee5fc4ef88e3b37a2948099841a3d3d4`; no PR yet. Native interface update: SELF.
Read `docs/development/simscape_tour_matching/README.md` and the driver/iron
capture audits. The tested prefix optimizer and native polynomial conversion
are numerical foundations only: no live Simscape fit has run. SSH over Tailscale
works to DeskComputer and ControlTower; both have R2026a/R2025b executables.
DeskComputer runs R2026a successfully and has completed a 20 ms native baseline.
Foundation and handoff commit `9dd2ff95f` are published. Native interface #9925
fixes model-workspace overrides, optional raw retention and timestamp-preserving
resampling. Thirteen MATLAB tests also pass in a fresh runtime checkout after
removing unnecessary MAT discovery for explicitly supplied joint order (SELF).
Use remote `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour-runtime`;
the preliminary `...-9921` worktree remains preserved. Native angular logs are degrees, unlike the canonical
rad contract. SELF adds tested direct joint-bus extraction: all 27 q/qd/qdd
channels are finite in the saved native baseline after conversion. Five new
tests pass; all 18 native tests pass together (`native_joint_suite.json`).
The actual wrapper baseline now returns finite q/qd; tau remains
unqualified. The consolidated RScap acceleration log is miswired and bypassed.
Critical next work #9927: the referenced gimbal hard-codes LS initial targets
and High priorities, ignoring per-instance mask values. Both shoulder angles
therefore initialized identically before repair. SELF repairs the six target
value bindings with `repair_gimbal_initial_targets.m`; two native tests pass and
an R2026a full-model baseline initializes the shoulders independently. Priority
callbacks are now repaired in all three referenced joint types and pass all
36 selector settings plus independent-instance checks. Native R2025b saves
and idempotent migration tests pass. All 22 native tests pass together in R2026a.
The full-model replay of both repairs succeeds (`baseline_priorities`, 20 ms,
39.18 s wall time) with finite q/qd and club/grip; actuator tau remains unqualified.
All three readable model snapshots and model hash/connection manifests are updated.
Backward Simscape export is unsupported and its exploratory outputs must not
become the delivered model. Lease and
presence include the model directory; preserve existing remote copied patches.
Read `REMOTE_EXECUTION.md` in the
same directory for exact paths, probe state and machine-transfer instructions.
Next: qualify actual actuator efforts and active motion/torque modes, attach
the measured markers to native body frames, then calibrate and fit growing
prefixes. Preserve reference-model agent #9914 and all other worktrees. Do not
call this epic complete based on toy-oracle tests or a kinematic overlay.

# Attributed Club Catalog Continuation

User-requested transfer: read `docs/development/capture_product_turnover.md`.
Active #9899 / PR #9959 (`feat/9899-calibration-revision-status`); prerequisites #9950 and #9954 are merged into main.
Tools #5169 merged. Preserve candidate PID 61800 and peer work.

Analysis #9945/#9947 merged; canonical handoff preserves provider and analysis scopes.

Concurrent eacd69858 integrates main 5ada5e6a6 and its player/equipment work.
Provider source and pin are unchanged. The inherited inventory omitted provider
files; the existing test failed, then full-authorship regeneration restored it.
All 271 integration controls pass again. Tools launcher follow-up is PR #5144;
final reviewed-pin qualification and physical/acoustic work remain open.

Main 90c3d0b77 is integrated, preserving merged C3D fitting and club/volume/handedness overlays (#9918/#9922) and attributed club catalog (#9919). Both task scopes remain in canonical HANDOFF.md. All 271 integration controls pass.
Main 8fce9f238 is integrated with capture PR #9917 LoD fixes and capture journey implementation preserved. Canonical HANDOFF.md retains the incoming capture handoff.

## Preserved Repository Context

The previous base handoff is preserved at [Full Base Handoff](https://github.com/D-sorganization/UpstreamDrift/blob/6e3610a9b/AGENT_HANDOFF.md). Canonical handoff retains incoming text below.

## Standing Constraints

- Read AGENTS.md and CLAUDE.md; maintain issue leases and session presence.
- Shared physics belongs in Tools. Do not copy or edit vendored source.
- Use topic branches and normal hooks; preserve protected CI and reviews.
- UP-D0 (#9066) and UP-D1 (#9067) remain a separate design-manual program.
  manuals/upstreamdrift QMD is the editable authority; generated LaTeX, PDF,
  DOCX and HTML remain non-editable artifacts. Run the configured
  scripts.check_design_manual_governance checks before changing calculations.
- Numerical convergence, measured calibration and perceptual evidence are
  distinct; source hashes alone do not qualify a physical model.
- Refresh this file, canonical handoff and DL-#9912 in implementation commits.

# Player Bag and Capture Equipment Continuation

Active #9905 in `feat/9905-player-club-bag`, worktree UpstreamDrift-player-bag. My Clubs UI, capture assignment/library display and immutable model context are implemented;652 broad regressions and69 focused/map/parity checks pass. Parent catalog #9919 merged01831aa4c; capture UX #9917 merged8fce9f238. Canonical state is `docs/development/HANDOFF.md`. Preserve live Capture Rig childPID61500 and standing manual governance. Wizard integration and broader goal remain open.

## Scoped Ubuntu CI Dependency Installation (#9894)

Isolated branch `fix/9894-ubuntu-ci-sources` replaces four standard CI APT
installers with signed Ubuntu sources and temporary package indexes. Source
configuration on shared runners is preserved. See `docs/development/HANDOFF.md`.

# Reference Alignment Controls

Current implementation: issue #9883, PR #9890, branch feat/9883-reference-alignment-controls. The canonical continuation state is docs/development/HANDOFF.md. Preserve the standing governance and earlier task context below.

## Comparison Rendering Qualification (#9882)

Active isolated branch `fix/9882-comparison-rendering` extends timing PR #9885.
Shared preview/export pixels, retained expert decoders, opacity/homography and
strict staged publication are implemented. Canonical evidence and remaining
#9883 controls are in `docs/development/HANDOFF.md`. Epic #9863 stays open.

## Reference Timing and Camera Evidence (#9881)

Active isolated branch `fix/9881-reference-timing` builds on #9884 and preserves
its state/lifetime fixes. Bounded event mapping, efficient sampling and actual
calibration/clock bindings are implemented; final qualification remains. See
`docs/development/HANDOFF.md`. #9882/#9883 remain open before epic closure.

## Comparison State Qualification (#9879)

Branch `fix/reference-comparison-qualification` preserves merged reference work
and fixes state loss, sidecar validation and export lifetime through the existing
SwingExportActions controller. Canonical evidence: `docs/development/HANDOFF.md`.
Epic #9863 is reopened; #9881, #9882 and #9883 track remaining release acceptance.
Earlier entries below describe historical implementations, not current completion.

## Bioptim Simultaneous State and Parameter Estimation (#9762)

Phase 4 of bioptim OCP migration complete. Branch `feat/9762-bioptim-parameter-ocp` implements
`src.shared.python.optimization.ocp.parameter_ocp` (335 LOC) for simultaneous trajectory tracking
and parameter estimation with quadratic priors, identifiability gating, and IPOPT Hessian tuning.
All unit contracts, isolation guards, and quality gates pass. PR #9878 merged.

## Capture Rig GUI: Layout Inversion, Responsive Adaptation & Evidence (#9847, #9848)

Epic #9843 complete. Branch `docs/9848-capture-rig-evidence` documents responsive
compact mode in `docs/motion_capture/capture_rig.md`, records before/after metrics
in `docs/motion_capture/evidence/capture_rig_responsive_evidence.md` (minimum width
reduced from 3276 px to 478 px against the <= 900 px budget; preview width 656 px vs
controls 240 px at 1280x800), and provides offscreen screenshot
`capture_rig_responsive_layout.png`. PR #9876 merged (`d34a41a2e`).

## Reference Overlay: Comparison Workspace, Saved Layers & Reproducible Exports (#9866)

Branch `feat/9866-reference-comparison-workspace` implements the synchronized comparison
workspace, saved layer configurations, and reproducible video/sidecar exports for calibrated
reference motions and capture takes. See canonical `docs/development/HANDOFF.md` and `DL-#9866`.
All focused unit and UI tests pass; quality gates verified.

## Calibrated Scene Registration & Event Synchronization (#9865)

Branch `feat/9865-scene-registration` implements calibrated reference scene registration,
event-anchor and offset synchronization, bounded time warping, gap masking, and distortion-aware
camera projection for ReferenceMotion and 2D expert videos. PR #9871 merged to main (`10caddd21`).

## Coaching Reference Work in Progress (#9862)

Isolated branch `feat/9862-coaching-drawings` builds on editing/library PR #9868
(`1e2469296`). Draft PR #9869 contains five saved drawing tools, gesture/keyboard controls, source-frame
visibility and common preview/still/video exports are implemented and passed
focused qualification. Dependency #9868 must merge before this PR is ready. See `docs/development/HANDOFF.md`. External reference epic
#9863 and fleet adoption are still open. No changes to shared clones or vendor
code; the concurrent panel-navigation correction is preserved.

## Segment Force Colors: Epic #9833 in Progress

Branch `feat/segment-force-colors` lives in `_codex_worktrees/segment-force-colors`.
See `docs/development/segment_force_color_epic.md` for current scope and evidence.
Shared Python/Three.js policies, native pendulum and MuJoCo reaction sources,
renderer adapters, desktop controls and WebSocket/Scene3D wiring are implemented.
Epic #9833 and children #9834–#9837 are published. Remote main `ff0effa5a` was merged
into the branch. Git works by clearing the stale `http.https://github.com/.extraheader`
per invocation and using `gh auth git-credential`; global settings are unchanged.
PR #9840 is open. C3D user segments now accept explicitly bound, clock-checked
loads and expose the shared controls. Native MuJoCo raster verification passed
blue/red output and pixel-exact off restoration (local output/force-colors).
Remote main `403292ca3` is merged and the SPEC conflict is resolved with both rows
preserved. PR CI cycle 1 exposed LoD storage access, a plotting import in headless
contracts, render-function size and a redundant websocket cast; fixes and a
headless regression are included. MuJoCo MeshCat now uses shared leaf-object
bindings; its native command test passes with meshcat 0.3.2 installed only under
ignored output/native-meshcat. C3D broad tests have one unrelated loader error-text
expectation mismatch; the new force test passes. Remaining gates: wider interface adapters, protected CI
and merge. Native MuJoCo tests on this Windows host must import mujoco before
pytest/Qt to avoid a loader-order DLL failure. Do not claim universal rollout.
Combined focused regression: 575 passed; web: 68 passed with TypeScript/ESLint.
Cycle 2 fixes add suite markers and merge main `39d944540` (CI's shallow direct
diff had falsely reported its new notebook test deleted). All earlier CI failures
are fixed locally; current-head checks remain required. Wider hosts remain open.

Updated: 2026-09-08 02:55 PDT
Updated: 2026-09-08 03:10 UTC
Updated: 2026-09-09 00:20 UTC (unit-gate PDF identity pins re-synced to the refreshed canonical PDF)
Updated: 2026-09-08 09:30 UTC (PR backlog catch-up sweep)
Updated: 2026-09-08 23:59 UTC (wave-2 PR triage, session UD2PRs)
Updated: 2026-09-08 (wave-2 issue backlog sweep, session UD2IssuesA)

## Capability Atlas #9850

Isolated branch `feat/9850-capability-atlas`, commit `SELF`, PR #9919 (open).
See `docs/development/HANDOFF.md` and `DL-#9850` for current validation and
continuation. Generated references consume existing registries and preserve
GUI epic #9843 and optimization-agent file ownership. Product #9849 and
performance review #9851 are separate workstreams. Fleet communication lives
in Repository_Management PR #1580 and has completed a real peer message exchange.

## Wave-2 PR Triage: 2026-09-08 (Agent `claude`, Session UD2PRs)

Repo-wide npm-audit red: advisory GHSA-2883-xcg3-v3hh (js-yaml high,
published 2026-09-08 between the 21:57 main push run and the 22:15 PR runs)
fails `code-quality` (`npm audit --audit-level=high`) on every merge ref
whose lockfile carries js-yaml 4.3.1 - main itself goes red on its next
Standard run. Fix on `bot/claude/npm-audit-jsyaml`: npm `overrides.js-yaml`
= `^4.3.2` in `ui/package.json` (dev-only dep of `@eslint/eslintrc`), audit
drops to 5 moderate, gate passes. Disposition table below is maintained as
PRs settle (in-progress at first commit).

Wave-2 disposition table (REST-verified 2026-09-09 ~00:45 UTC):

| PR                         | Disposition at yield                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #9465                      | **MERGED** (squash `9623a5662`).                                                                                                                                                                                                                                                                                                                                                                                                                     |
| #9827                      | **MERGED** (squash `403292ca3`).                                                                                                                                                                                                                                                                                                                                                                                                                     |
| #9826                      | PDF identity pins in `tests/research/test_publication_quality.py` re-synced to the refreshed artifact (`bf855f79`, 2012367 bytes) on `fix/9825-preserve-reviewed-claims` (commit `30aecd113`); auto-merge armed.                                                                                                                                                                                                                                     |
| #9763-#9767                | Five dependabot /ui bumps; unblocked by this PR's audit fix; auto-merge armed, draining.                                                                                                                                                                                                                                                                                                                                                             |
| #9471                      | Merged main in twice (SPEC row conflicts: dropped the branch-side duplicate #9476 row, kept this PR's #9471 row); repo-structure-gates green; auto-merge armed.                                                                                                                                                                                                                                                                                      |
| #9434                      | Merged main in (WORKFLOW_TRACKING.md conflict: companion entry kept, main's always-on-unit-lane entry kept); auto-merge armed.                                                                                                                                                                                                                                                                                                                       |
| #9440                      | Merged main in: theme modify/delete resolved as PR deletions; shadow ledger 33 -> 17 (stale + this-PR entries dropped); seam rulings = main's retirement narratives + PR cleaned rows for notes/plot_theme/theme; divergence inventory regenerated; SeamRedirectFinder roots synced to merged cleaned rulings (commit `4d3193f81`). Auto-merge armed; the wave-1 #8972/#9037 Tools-palette prerequisite still applies if palette-consumer tests red. |
| #9442                      | Stacked on #9440's branch (base `readiness/p1-9406-delete-tools-canonical-1`, not main); branch merged forward to `b45ec951f` with the seam-root sync; fresh CI running; auto-merge not armable while stacked - retarget to main only after #9440 lands (wave-1 rule).                                                                                                                                                                               |
| #9636, #9633, #9618, #9610 | Conductor drafts, skipped per assignment.                                                                                                                                                                                                                                                                                                                                                                                                            |

## Wave-2 Issue Backlog: 2026-09-08 (Agent `claude`, Session UD2IssuesA)

Second-wave residual-backlog sweep over the older half of the 186 open issues
(#8346–#8930). Sibling PRs from the same sweep (based on the identical main
revision): PR #9828 (#8922, mocap retargeting IK cost) and PR #9831 (#8928,
pendulum result accessor caching).

- **#8842** — `notebooks/bunkershot3d/phase1_mvp.py` was unrunnable: it
  imported a nonexistent top-level `bunkershot3d` package and pointed at a
  nonexistent repo-root `configs/` tree. It now imports via
  `bunkershot3d.*` with a repo-root `sys.path` bootstrap, resolves the
  packaged `src/bunkershot3d/calibration/configs/canonical.yaml`, writes
  artifacts under gitignored `output/bunkershot3d/`, and exits 1 with a
  clear log line when the optional `pychrono` backend is absent (phases 1-2
  still produce their artifacts). Smoke tests:
  `tests/bunkershot3d/test_phase1_mvp_notebook.py`.
- **#8922** (PR #9828) — `MotionRetargeting._solve_frame_ik` called
  `mj_forward` once **per marker per IK iteration** (~40x overcount with the
  golf marker set) and grew the stacked Jacobian with a fresh `np.vstack`
  per marker. The marker→(body, target) pairs are now resolved once per
  frame, `mj_forward` runs exactly once per iteration, and the
  Jacobian/error rows are batched with a single `np.vstack` /
  `np.concatenate`. Regression tests pin the forward-call bound and
  multi-marker error reduction
  (`tests/unit/engines/mujoco/test_motion_capture.py::TestMotionRetargetingIKCost`).
- **#8928** — pendulum simulation result accessors re-integrated the
  trajectory on every call (`all_energies` once per energy key = 3 passes;
  `extract_series` fresh per plot per joint). `TrajectoryResultMixin`
  now computes every `all_*` batch in a single pass and memoizes it on
  the result object, and `energy_at` derives `total` arithmetically
  (E = T + V) instead of re-evaluating both terms. Benchmark (400-step
  golfer result): repeated `all_accelerations` 0.64 s → ~6 µs; repeated
  `all_energies` 0.15 s → ~6 µs.

## Capture Rig Multiview Epic #9818: 2026-09-08 (Agent `claude`)

The Capture Rig tile went from "no live view at all" to a recording station.
Merged in order: #9809 (live preview, movable/scrollable panes with saved dock
layouts, transport-style recording), #9819 (#9816 theme and layout standards:
`styling.py`, `header.py`, `action_grid.py`, `playback.py` split out of
`gui.py`), #9820 (#9810/#9811 `layout_model.py` compositor + `layout_presets.py`
store), #9821 (#9815 `mosaic.py` + `rig multipicture`), #9823 (#9813/#9814 live
preview and playback rendered through a `LayoutSpec`). PR #9822 was closed as
superseded: it was #9823's base, so its files landed byte-identically with that
squash, and merging it would have reverted the multiview panes. Only #9817
(this documentation pass) remained.

Hardware facts worth keeping, all measured on the three-camera rig and written
up in `docs/motion_capture/evidence/capture_rig_multiview.md`:

- A DirectShow camera opens once. During a take the recorder tees its own
  preview (`--live-preview DIR`); the tile shows those snapshots.
- That tee must decode cheaply. At full resolution it starved the stream copy
  (95/393/74 frames in ~7 s); with `-lowres:v 2` and a 256 MB real-time buffer
  all three cameras hold 60 fps (493/494/462 frames in 8 s).
- Enumeration costs ~30 s; `--camera VIEW=INSTANCE_ID` reuses what the preview
  already bound.
- OpenCV cannot drive these cameras: Media Foundation hangs on the third unit
  and DirectShow-by-index refuses 1920x1200@60.

Two defects were found while integrating the parallel branches, not by CI:
`workflow.py` briefly held two `ACTION_HELP` tables where the second silently
won, and three new entry points each took nine parameters against a budget of
eight (now `MosaicOptions` / `MultipictureArgs`). A wall-clock assertion in
`test_layout_model.py` that flaked under load now takes the best of several
rounds.

## PR Backlog Catch-Up Sweep: 2026-09-08 (Agent `claude`, Session UpstreamPRs)

Disposition of the 38-PR open backlog (REST-verified states at sweep start):

- **Merged (18):** #9513, #9715, #9716, #9717, #9718, #9719, #9721, #9722, #9728, #9734, #9736, #9738, #9739, #9741, #9742, #9743, #9744, #9745. Every branch was brought current with `main` (REST update-branch, never force-push) and its AGENT_HANDOFF/SPEC/DEVELOPMENT_LOG conflicts resolved (union for SPEC rows, main-wins for handoff stamps, DL entries re-added into the Active table).
- **Closed as redundant (3):** #9433 (duplicate of #9723 for #9409), #9437 (superseded by the merged #9412 registry work, `1b53a9bf5`), #9737 (superseded by merged #9740 for #9699). Explanatory comments posted on each.
- **Armed for auto-merge (9):** #9434, #9465, #9471, #9720, #9723, #9724, #9725, #9726, #9729. Branches are conflict-free and current; auto-merge (merge/squash per repo allowance) merges each as `quality-gate` passes under strict up-to-date. No action needed; they drain serially.
- **Blocked (2):**
  - **#9440** — the `split` ruling deletes UD `theme/__init__.py`+`palette.py`, but main landed UD-only palette/typography extensions (#8972/#9037, `ThemePalette`/`get_current_colors`, ~217 lines) that the pinned Tools tree (`eab74a901a`) does not contain; 10+ launchers/API modules consume them. Next steps: land #8972/#9037 in D-sorganization/Tools tools-canonical side, bump the `vendor/ud-tools` pin, re-merge main, regenerate `docs/shared_tools/divergence_inventory{.md,.v1.json}` via `scripts/shared_tools/divergence_inventory.py`, then re-run `tests/unit/shared_python/test_seam_redirect.py` + quality-gate.
  - **#9442** — stacked on #9440's base branch (retarget to `main` only after #9440 lands); its own CI failures overlap open #9607 / PR #9726 (pinocchio authority lock drift) and a self-hosted authority-runner artifact path. Do not merge in isolation.
- **Skipped drafts (4):** #9610, #9618, #9633, #9636 (conductor research drafts, not trivially completable).

No issue was closed in this sweep (redundant-PR closures do not close issues). Fleet-wide handoff/lease state at sweep start: `C:/tmp/backlog/UpstreamDrift.md`.

## Impact Dynamics and Acoustics: #9700

- Current checkpoint: docs/9700-impact-handoff, SELF; PR #9962. Canonical [HANDOFF.md](docs/development/HANDOFF.md) records merged sources, validation boundaries and ordered takeover. Provider #9916/#9920 are merged; Python/Rust/gitlink agree on e83bd2e4a7a29a2dcd8145ef2d1efa07123324f0. This checkpoint changes no pin or installed runtime.
- Tools contact/load/FRF/calibration foundations and Affine force-regularity #4356 are merged. Friction #5162 requires exact-head hosted qualification and merge at this checkpoint. Coordinate the next combined pin and installed-wheel checks with the context/capture owner.
- Physical, radiation and blinded-perception requirements remain open. Preserve #8557 authority, manufactured-data limits, numerical ceilings and all 328 reviewed claim outcomes. Synthetic convergence is not empirical validation.
- Historical source-specific receipts remain in [provider turnover](docs/development/impact_provider_import_turnover.md), [renderer review](docs/development/renderer_reference_9783_turnover.md), [claim preservation](docs/development/claim_preservation_9825_turnover.md), [shooting convergence](docs/development/shooting_convergence_9830_turnover.md) and [program design](docs/development/impact_acoustics_program.md). Bioptim/CasADi 3.6.7 qualification does not qualify the unsuccessful 3.8 RK4 case.

Epic #8557 is canonical; issue state, local files, and checkpoints are not completion
evidence. UP-D0 (#9066) and UP-D1 (#9067) remain a separate design-manual program.

Detailed takeover context, the merge-versus-quarantine boundary, exact smoke contract,
recovery constraints, and next commands are in `docs/development/proximal_distal_program_turnover.md`.

## Import Bootstrap Fail-Fast: #9733

- Open PR fixes the pytest livelock in fresh worktrees with `vendor/ud-tools`
  uninitialized: `src/__init__.py` now raises an actionable ImportError naming
  `git submodule update --init vendor/ud-tools` instead of installing the
  fallback finder into an unbounded `find_spec` recursion. Regression tests:
  `tests/unit/repo_hygiene/test_src_fallback_fail_fast_9733.py` (probe simulated
  by monkeypatch; never touches the real submodule).

Epic #8557 is canonical; issue state, local files, and checkpoints are not
completion evidence. UP-D0 (#9066) and UP-D1 (#9067) remain a separate
design-manual program.

Detailed takeover context (merge-versus-quarantine boundary, exact smoke contract, recovery constraints, next commands): `docs/development/proximal_distal_program_turnover.md`.

Seam (#9406) and failure triage (#9474): see
`docs/development/readiness_seam_handoff.md` before retiring a shared cluster.
UD #9492 (branch `claude/issue-9492-decompose-timer`): `_on_timer` and
`_add_live_kinematics_overlays` are decomposed into focused helpers, both dated
`architecture_budget.json` exceptions are removed, behavior pinned by tests.

## In-Flight Tool Migration: #9470 (Launch-Monitor Async Analytics)

- Branch `claude/issue-9470-async-analytics` migrates the seven launch-monitor
  analysis handlers onto the #8880 `async_action` worker: one shared
  `AsyncActionBar` for all trigger buttons, widget-free `_compute_*` halves
  with cancellation checkpoints, synchronous `present(compute())` paths kept,
  and the embed adapter `cleanup()` cancelling and joining the worker.
- The branch is stacked on PR #9472 (the #8880 mechanism, `readiness/p2-8880-async-action-worker`);
  merge #9472 first, then this PR applies cleanly.
- Remaining #9470 checklist follow-ups, in pain order: `bunker_shot_gui/gui.py`
  (`_guarded` wait cursor), `putting_green_gui`, `ball_flight_gui`,
  `swing_flight_pipeline`, `terrain_engine`, then the audit-only tools.
- Gate: `python -m pytest -q tests/ui/tools/launch_monitor tests/tools/test_async_action.py`

## Unit-Gate `src`-Identity Sentinel: #9387 (Merged)

- The worker-corruption class from #9099 (a test mutating
  `sys.modules['src']`/`src.*` and corrupting later tests on the same
  xdist worker) is covered by a runtime sentinel,
  `tests/unit/repo_hygiene/test_src_identity_sentinel.py`: each of the
  four documented victim files runs in its own serial subprocess
  (`-p no:xdist`) and the sentinel asserts `sys.modules['src']`
  identity plus the `src.*` namespace snapshot are unchanged. RED was
  demonstrated with a scratch pivot module (removed before commit).
- The four judgment-call leak sites from the audit are explicitly
  snapshot/restored (`test_ux_enhancements.py`, pinocchio
  `test_tasks.py`, `test_gui_import_boundaries.py`,
  `test_golf_launcher_integration.py`); the full audit table lives in
  the PR body. `tests/unit/test_ux_enhancements.py` collects 0 test
  functions (dead fixture file since #5753) — deletion candidate for a
  follow-up PR; do not delete it here.
- Known environment limitation: in `git worktree` checkouts (`.git` is
  a pointer file), `tests/scripts/test_validate_suite.py` fails two
  `.git`-inspecting tests; the sentinel tolerates exactly those two ids
  there and requires them to pass in normal checkouts and CI.

## Pre-Commit on Windows — Resolved (#9494)

- The hook environment works on Windows; no `--no-verify` exception exists
  or is needed. `default_language_version: python: python` resolves to the
  PATH interpreter (3.13.3 here); the old `python3.11` pin is gone since
  #1792/#2720. Do not pin a minor version — workstations without it cannot
  build hook virtualenvs.
- Verified 2026-09-08 (pre-commit 4.6.2, from-scratch env build): all
  commit-stage hooks pass; pre-push `mypy`/`bandit` pass on scoped files;
  `pytest-unit` is slow locally (CI owns the full suite). CLAUDE.md
  "Hook bypass policy" documents this resolution.

## `bioptim` Optimal-Control Layer and the Swing-Dynamics Fixes (#9762)

- Branch `claude/fixes-epic-implementation-x2bu36`, PR #9768 (open). Epic doc:
  `docs/issues/EPIC_BIOPTIM_OCP_INTEGRATION.md`; decision: ADR-0050.
- Prerequisite issues #9755-#9761 are filed; #9755-#9760 are implemented on
  this branch, #9761 (upstream PR to pyomeca/bioptim) is external and open.
- Epic phases 0-3 are implemented and tested; phases 4 (parameter block) and
  5 (moving-horizon wrapper) are not started. `ocp/tracking_ocp` already
  accepts a `parameters` list, which is the seam phase 4 builds on.
- **Do not** import `bioptim` outside `src/shared/python/optimization/ocp/`:
  `tests/architecture/test_bioptim_isolation.py` fails on it. bioptim is
  git-pinned to `Release_3.4.0` (SHA `fdafe4d9`) in the `[bioptim]` extra;
  re-pinning is a ticket that re-runs the phase 0-3 tests.
- Two findings that constrain how results may be read:
  maximising terminal clubhead speed is a concave objective and converges in
  no backend once the dynamics are enforced, so the OCP defaults to a convex
  target-speed objective; and the six-marker set cannot observe the full
  seven-DOF chain (`hip_rotation` and `trunk_rotation` are an exact null
  direction), so tracking results report their own identifiability.
- Gate commands: `MPLBACKEND=Agg pytest tests/integration/optimization/ocp
tests/architecture/test_bioptim_isolation.py -m "not slow"`;
  `pytest tests/unit/optimization tests/unit/estimation`;
  `MPLBACKEND=Agg PYTHONPATH=src python -m benchmarks.bioptim_parity --nodes 8
--duration 0.6`. The benchmark needs `PYTHONPATH=src` (pytest's conftest adds
  it, `python -m` does not) or it dies importing `bunkershot3d`.
- The ocp tests live in `tests/integration/optimization/ocp/`, NOT under
  `src/`. `scripts/check_test_layout.py` (the Test Layout Guard inside
  `repo-structure-gates`) rejects any new `tests` directory under `src/`
  because root pytest does not collect it -- its `LEGACY_SRC_TEST_DIRS`
  allowlist is grandfathered debt, so do not add to it. They sit beside
  `test_casadi_swing_live.py`, the other suite that needs the real optional
  stack rather than the mocks `tests/unit/conftest.py` installs.
- The lane that really exercises them is the `bioptim OCP Tests` step in
  `ci-optional-stack.yml`, which installs the extra and runs only
  `tests/integration/optimization/ocp` plus the isolation test under
  `-m "requires_bioptim or integration"`. It is NOT fail-soft. Locally that
  exact selection is 26 passed in about 150 s; a timeout there is contention,
  not a hang -- the slowest test on its own is 80 s against a 600 s budget.
- Run the ocp tests on their own. Co-running them with
  `tests/unit/optimization` makes `bioptim_available()` return False and
  silently skips 16 of them, in either collection order and on `main` as well
  as here -- something in that directory poisons the import. Pre-existing, not
  this branch's, and worth its own issue: a skip that only appears in a
  combined run is exactly the kind CI hides.
- Two gates only reachable once the earlier ones passed, both fixed:
  `code-quality` fails at its **mypy** step, not ruff, on
  `crocoddyl_backend.py:316` -- `pin.Motion` is missing from the repo's own
  `stubs/pinocchio/__init__.pyi`, which `mypy_path = "stubs"` makes
  authoritative whether or not pinocchio is installed. Reproduce it in an
  environment WITHOUT pinocchio; a venv that has the real package hides
  nothing, but it is the stub mypy reads either way. `unit-test-gate` fails
  `test_divergence_inventory` until `python -m
scripts.shared_tools.divergence_inventory --write` re-records the 16 new
  `optimization/ocp/` files; that regeneration also rewrites unrelated
  authorship rows, which is expected -- the file is generated, not hand-edited.
- Architecture budget: the nine violations this branch authored were fixed by
  decomposition -- `CasadiSolveOptions` and `MaxSpeedOcpOptions` group the
  keyword arguments that pushed `solve_swing_casadi` and `build_max_speed_ocp`
  over the parameter budget, `_SwingModelSurface` moves the bioptim-independent
  half of the adapter to module level, and the parity benchmark is one function
  per backend. `crocoddyl_backend.solve_swing_ddp` is byte-identical to
  `origin/main` and carries a dated exception instead.
- CI state on PR #9768: the `hatchling` direct-reference fix (`52a8710`)
  cleared the eleven jobs that could not build the package at all. The
  `dependency-consistency` gate is red on `main` too, because #9716 added
  `openpyxl` and `imageio` to the `dev` extra without regenerating the
  locks; this branch carries the regenerated `requirements-dev.lock` and
  `environment.yml` so the gate passes here and no-ops once `main` catches
  up. Regenerate them only with Python 3.12 (`make sync-deps`) — that is
  the interpreter the gate runs, and 3.11 produces a different lock.
- `code-quality` runs `ruff` unpinned, so it floats ahead of the
  `ruff>=0.15.10` floor in `pyproject.toml`. 0.15.17 reformats a
  parenthesised lambda body in `benchmarks/bioptim_parity.py` that 0.15.8
  left alone; the committed form is stable under both.

## Vendor Pin & Alias Predicate: #9631

- `vendor/ud-tools` = Tools `eab74a901a`, carrying the Tools#5049 flattened-install alias fix
  (`f8b94bfe` is an ancestor); do NOT rewind — that drops Tools #5051-#5057 incl. the Sentinel
  ShellTool injection fix. Cargo rev, companion `pinned_commit`, `check_tools_pins.py` agree.
- `tests/unit/repo_hygiene/test_pinned_import_alias_contract.py` pins the flattened-install
  contract (RED at pre-#9657 pin `3d93bb2c`, GREEN at the current pin). Child copy stays
  unconverged per #9657. Remaining: re-cut the 2.1.3 release.

## In-Flight Issue Work

- `claude/issue-9612-video-suffix` (PR pending): `src/api/routes/video.py`
  derives the upload temp-file suffix from the filename against
  `SUPPORTED_VIDEO_SUFFIXES` and fails closed with 400 on unknown/missing
  extensions; state tracked in `docs/development/DEVELOPMENT_LOG.md` (DL-#9612).

## Protected Authority

- UpstreamDrift protected `main` is
  `3503674a90e3ca6d75e81f084f011299f5e95794`, the verified squash merge of
  turnover PR #9309. Its reviewed head and squash merge share exact tree
  `827608120154ecdfb6dc8b9f0c53b988a0454343`.
- #9306 adds signed-gap retention, bracketed opening/reattachment location on a
  declared linear state interpolant, duplicate-time event alignment, and the
  distributed replay adapter into the protected attribution kernel.
- #9308 adds the prospective six-case current-main smoke registration without
  executing or promoting an outcome. CI Standard run `33322181043`, Optional
  Stack run `33322180991`, and Bot CI Trigger run `33322180972` passed. Exact
  turnover evidence is recorded on #9153 in comment `issuecomment-5469958455`.
- PR #9302 remains the manufactured-evidence qualification parent at
  `7dc5f86af68907f19bc953c509d96d05f505cdab`; PR #9299 remains its CRBA
  requalification parent at `c8a283f4ffb408d5932bdc2da3f2f0c64665ef83`.
- The requalified paper has 253 pages, 2,011,818 bytes, and SHA-256
  `554fca211786ac5a06959f41b9f7d75720c89155168faeaac9d648524e8c9e36`.
  Tagged-PDF and embedded-font gates remain open archival limitations.
- AffineDrift #3993 pins the exact #9152 authority as protected squash
  `9b9cbcc2199f1fbf8cd281beb08c57d543b552b1`; handoff correction #3995 merged
  as `6cc909273d63147392b17078a35c6c4da034e1da`.
- Tools force-source frame #4873 merged as
  `cc883cbaf63157b58c71cba385a683df2762b0cb`; Tools #4142 remains the broader
  reusable-variation completion authority.

## Impact Explorer Web Route Producer: #9484

- PR: #9724 (open against `main`, `Fixes #9484`).
- The `rate_of_closure` tile declares `web.mode: route` for
  `/tools/impact-explorer`; `src/api/local_server.py` mounts
  `vendor/ud-tools/src/rate_of_closure/web/dist` when it exists. CI Standard's
  `impact-explorer-web-build` job now builds that bundle from the pinned Tools
  tree (`npm ci`, then `npm run build -- --base=/impact-explorer-app/`) and
  `scripts/check_declared_route_producers.py` fails any launcher route that no
  pipeline produces (unit tests in `tests/scripts/test_declared_route_producers.py`).
- Open decision (maintainer, #9417): whether the built bundle ships inside the
  wheel/image or is fetched as a Tools release artifact; this work deliberately
  does not re-architect distribution. The vendored build authority is Tools'
  `.github/workflows/rate-of-closure-web-distribution.yml` (node 22, `npm ci`
  in `src/rate_of_closure/web`); the base path comes from the route fallback
  in `ui/src/pages/ImpactExplorer.tsx`.

## Active Hybrid Authority Repair: #9236

- The candidate adds a study-scoped, hash-locked CPython 3.11.15 manylinux
  environment and an isolated runner-temp native job. The canonical serializer
  is deterministic, rejects NaN, writes atomically, and records the exact
  profile. Semantic comparison requires the complete six-gate tolerance set,
  the exact 18-field rolling compatibility policy, every governed result path
  in both records, finite numeric leaves, and an internally derived maximum for
  each record. Missing paths, policy entries, stale profiles, inconsistent
  maxima, negative residuals, and non-monotone convergence fail closed.
- The committed scientific record, checksum, release manifest, and
  claim-evidence manifest are byte-identical to `origin/main`. The next step is
  an ordinary protected PR that runs the exact locked Linux CPython 3.11.15
  native job. Accept a regenerated record only when two same-environment builds
  are byte-identical, the declared dependency/profile identity matches the job,
  all numeric evidence satisfies the declared comparison policy, and the
  checksum/manifest cascade is regenerated by canonical commands.

## Protected Prospective Smoke Registration: #9153

- PR #9308 protected-squash-merged as
  `651de90a4cf8e1195ec7f3ab3ae16883ec8f6172`. Reviewed head
  `142662fb3de86fbb8086b83772df09425897f9c2` and merge share tree
  `a0dd17ed057c0aea45f112172d406ace971610df`; remote `main` equals the squash.
- PR #9309 changed only this root handoff and the detailed turnover document;
  it introduced no runtime, smoke outcome, scientific promotion, human
  validation, or coaching authority. Any post-merge turnover correction PR is
  recorded in the latest #9153 comment rather than self-referentially embedded
  here; verify it is protected before implementation resumes.
- New registration source:
  `scripts/research/proximal_distal_energy/articulated_distributed_smoke_registration.py`.
- New frozen protocol:
  `docs/research/proximal_distal_energy_transfer/data/articulated_distributed_smoke_registration.json`.
- The prospective matrix contains six cases: MuJoCo and Pinocchio at 1.0,
  0.5, and 0.25 ms, using source case 0/sample 6, one station per hand, 1.5 mm
  slack, zero generalized initial velocity, 1 mm club displacement, and
  -0.8 m/s initial club velocity over 50 ms.
- Execution is explicitly `not_started`; retained outcomes are empty and
  promotion authority is none. The registration binds the protected evaluator
  revision/tree, seven evaluator-source hashes, and the 35,568-byte input NPZ
  with SHA-256 `9fa4364571ba5535995c63226289c0711ee1ebf37c58b7a3b4e4d14a98561779`.
- Only opening and reattachment are eligible. Friction-limit, static stick/slip,
  inferred discrete impact, causal counterfactual, biological, human, and
  coaching interpretations are prohibited.
- TDD evidence: the missing module produced the expected RED import failure;
  after implementation, four registration tests pass serially.

## Spec Check Reminder Fail-Safe Extraction (#9499)

- The `Verify SPEC.md freshness` job no longer carries its comment-posting
  logic as an inline `actions/github-script` heredoc (an unescaped backtick
  from the RM #1520 wording once aborted it with `SyntaxError: Invalid or
unexpected token`, swallowing the finding). Posting now runs
  `scripts/post_spec_reminder.py`, which prints the full diagnostic into the
  job log and exits 0 on any posting failure; the `always()`-guarded
  "Fail if spec is stale" step owns the non-zero exit.
- Contracts pinned by `tests/ci/test_spec_check_workflow.py`.

## Immediate Order

1. Verify the latest turnover-correction PR recorded on #9153 is protected on
   remote `main`; preserve its existing runs and auto-merge.
2. From that exact protected base, implement the current-main single-worker
   atomic smoke runner with RED contracts for registration identity, complete
   case enumeration, atomic resume, typed failures, and no outcome promotion.
3. Execute the six registered cases only after runner code and tests merge.
   Never import or relabel legacy checkpoints as outcomes.
4. #9483: the stale 15-tile nav-gap audit `reports/feature_navigation_gaps/` is deleted (findings dispositioned in the deleting PR); live tile truth is the `src/config/models.yaml` registry plus the generated launcher manifest (#9412/#9437/#9478).
5. Regenerate `requirements*.lock`/`environment.yml` via dispatch-only `lock-refresh.yml`
   for #9533 (DL-#9533, PR #9716) once it merges; the locks were left untouched there.

## Spec Merge-Driver Vendoring (#9476)

- `scripts/install_spec_merge_driver.py` and `shared_scripts/spec_changelog.py`
  are re-vendored from Repository_Management#1521's corrected copies and pinned
  byte-identical by `tests/unit/scripts/test_spec_merge_driver_vendor_drift.py`;
  `scripts/setup_hooks.py` now calls the installer (issue #9476), so the
  documented setup registers the `spec-rows` driver, and the installer
  docstring names this repository's entry point.

## Scientific Boundaries

- Event locations qualify the retained discrete trajectory only; they are not
  the continuous integrator's exact event solution.
- Compliant opening/reattachment records zero discrete event impulse and work by
  model definition; it is not evidence of a physical impact.
- Same-trajectory attribution is descriptive. Divergent forward
  counterfactuals require a separately registered design.
- Energy transfer, momentum redistribution, joint work, contact power, event
  timing, and clubhead speed are distinct estimands.
- Native-engine agreement verifies declared operators and the common contact
  law; it does not calibrate anatomy, grip, shaft, ground, equipment, or human
  strategy.
- #8556/#9004 remain governed human-data boundaries. Synthetic evidence cannot
  substitute for bilateral six-axis participant grip wrenches.

## Frozen External Boundary

- #8800 remains frozen at source `1bd4d57da7bd257b76b42b3cc19524b283b5f748`; only 93/830 checkpoints exist.
- ControlTower ground stopped at 45/48 and shaft at 48/48. Its WSL VHDX is unreadable (`0x80070570`). Do not retry WSL, repair/mount/copy/mutate the VHDX, restart services, or launch a replacement without explicit approval and a recoverability plan.
- DeskComputer remains runner-drained; keep tests serial and web tests at no more than two workers.
- The accumulated campaign worktree is `UpstreamDrift-worktrees/9153-forward-impulse-work`, last remote-equal at `1e5e823ca2fa9391134e8a0ccf140a36036a88a7`, with 233 commits ahead and 48 behind at the last audit. Preserve its evidence and quarantine.

## Active AffineDrift and External Program Boundaries

- Markerless Mocap Program (#9063): Tools #4706 owns capture/contract schemas;
  UpstreamDrift #9069 (folded into #9422) owns app orchestration; makes no physical-lab qualification claim.
  Rig bring-up evidence: `docs/motion_capture/usb_camera_rig_bringup.md` (#9586); consumer slices #9589–#9592.
- Foundation #9180 merged as `1af18489e8755933a0d189aa8edafe787fa94d0f`; publication #9214 merged as `a8073c42edc811522c5d5709744f55c5cbd0fa8e`.
- Governed companion workflows (#9190) define the 15-record registry, public executor, and CI execution evidence across 10 success and 4 failure fixtures.
- #9222 has exact tree `c468c0db`, but its protected-main run was cancelled with no jobs or artifacts. #9192 remains open pending post-#9236 exact bytes; #9174 remains open.
- ADR-0043 and schema v1 are one-way UpstreamDrift software-fact authority for AffineDrift #4010. #9064 remains design-manual authority and #9070 remains typed calculation-manifest authority.

## Validation

Use `C:\Users\diete\AppData\Local\Programs\Python\Python312\python.exe` with `-n 0` for pytest.

```powershell
python -m pytest -n 0 -q tests/research/test_articulated_distributed_smoke_registration.py
python -m pytest -n 0 -q tests/research/test_articulated_contact_events.py
python -m pytest -n 0 -q tests/research/test_articulated_forward_attribution.py
python -m scripts.research.proximal_distal_energy.articulated_distributed_smoke_registration validate
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/ci/check_file_size_budget.py
python scripts/ci/check_architecture_budget.py
```

Also run claim/evidence integrity, release qualification, PDF inspection, and
affected full gates after publication changes. Never force-push, bypass branch
protection, relax tolerances after inspecting results, or create capacity-only
reruns. Do not restart the Actions runner or start WSL.

## UI Dependency Pin: #9249

- Dependabot now ignores `@vitejs/plugin-react` major updates (`.github/dependabot.yml`): 6.x needs Vite 8 (`peerDependencies.vite: "^8.0.0"`; Vite 7 exports no `./internal`), so a lone bump cannot merge. Stay on plugin-react ^5 with vite ^7.3.2 until a paired Vite-8 upgrade; pairing note lives in `ui/README.md`. Branch `claude/issue-9249-ui-pin`.

## OCP Compat Robust to Poisoned `sys.modules` (#9771)

- `tests/unit/conftest.py` installs spec-less `casadi`/`pinocchio` MagicMocks
  process-wide in `pytest_configure` and never removes them, so a lane
  collecting `tests/unit/optimization` and `tests/integration/optimization/ocp`
  together made `_compat.bioptim_available()` report the genuinely installed
  stack as absent and silently skip 16 bioptim ocp tests.
- `src/shared/python/optimization/ocp/_compat.py` now probes top-level names
  through `importlib.machinery.PathFinder` (ignores `sys.modules`), so a mock
  hides a real distribution only when no real one exists behind it;
  `require_bioptim()` evicts mocked `casadi`/`bioptim`/`biorbd_casadi`
  entries before importing so bioptim binds the genuine modules (the unit
  tree's autouse fixture reinstalls its own mocks per test, so unit-tree
  degradation semantics are unchanged).
- Contracts: `tests/integration/optimization/ocp/test_compat_poisoned_sys_modules.py`.
  Note the local reproducer cannot exercise real casadi/bioptim (Python 3.14
  has no casadi wheel); CI lanes with the `[bioptim]` extra must show the
  ocp bioptim legs running (not skipping) in combined lanes.

PR #9840 CI cycle 3: full unit gate passed 14,431 tests and failed only the companion feature-count expectation and shared divergence inventory. Updated counts for the new controls and regenerated the inventory with the pinned Tools tree. Both affected modules pass (34 tests). Await the refreshed remote gate before merging.

Pinocchio and Drake now expose View > Segment Force Colors via a shared MeshcatForceColorSession. Both native GUI tests pass on isolated Linux (Pinocchio 4.1.0, Drake 1.56.0). Explicit bindings and synchronous caller-supplied axial frames are required; automatic native reaction inference is not claimed. Session regression covers stale frames, model replacement and toggling. OpenSim currently has result plots, not a 3D animation host.

Final native-host CI exposed optional Qt menu return annotations. Menus now use explicit QMenu construction after validating the menu bar. Pinocchio synchronizes force colors in its GUI coordinator override, leaving the legacy visualization mixin unchanged; native Linux host tests still pass (2 tests). Current required gate remains red until the type correction is validated remotely.

The final host CI type check passes after menu contracts and the native COM matrix correction. The DRY gate then identified duplicated menu setup; install_force_color_menu now owns that validated setup for both GUIs. Native GUI tests pass after extraction. Await the corrected head's aggregate quality gate before closing the epic.

## Capture Editing and Reference Work

- Product #9849 now includes trim/crop #9860, library/notes #9861 and coaching drawings #9862; separate advanced reference epic #9863 has children #9864–#9866. All are active goal scope.
- Isolated branch feat/9860-swing-editing contains the edit recipe, ingestion mapping, native editor and visible capture library with notes, imports, archive/storage, rename recovery and editable copies. See docs/development/capture_editing_integration.md for reuse audit and remaining work. Selected-swing export and downstream trimmed-timeline qualification are implemented; PR #9868 is open after normal push checks; protected CI/merge remains.
- Preserve the concurrent #9843 GUI dock/layout work; new header entry points use existing wrapping layout without changing dock policy. Raw recordings and prior analyses must remain intact.

## Capture Product Review (#9851, #9857)

See `docs/development/HANDOFF.md` and `docs/development/capture_product_review.md`. Bounded duplicate-frame cache and child startup recovery are covered by six focused tests. The camera suite passed 241 tests before the recovery change. GUI #9843 remains independently owned; hardware qualification remains outstanding.

Expert reference imports are in draft PR #9870 (30 integrated tests, 49 atlas/registry tests and scoped quality gates passed). Registration/time mapping #9865 and native comparison #9866 remain next. Preserve native drawing dependency #9869.

## Verified Agent Context (#9915)

[Component map](docs/agent_context/README.md). CI tests repaired.
