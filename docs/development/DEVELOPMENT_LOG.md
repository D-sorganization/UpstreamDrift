# Development Log — UpstreamDrift

State table for every feature in flight in this repository. Update entries
**in place**; never append dated sections. One entry per feature, from proposal
to ship. See the `development-logs` section of `AGENTS.md` for the binding rules
and `shared_scripts/development_log.py` for the validator.

- **Portfolio:** golf
- **WIP limit:** 8
- **Last audited:** 2026-09-12 by claude
- **Current metadata review:** Last verified commit identifiers below bind path inventory only to e68baeabde8ff639f846a167ca7703ace43bd0a9; no tests or native runs were performed. Full original row text before narrative condensation is retained in [the pre-compression snapshot](evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_PRE_COMPRESSION_20261011.md.snapshot). Prior verification wording and per-path Git blob identities are preserved in [the source metadata audit](evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json).

## States

`proposed` → `in_progress` → `in_review` → `shipped`, with `parked` reachable
from any live state and `abandoned` from `parked`. `shipped` never returns to
`in_progress`; open a new entry instead.

## Active

### DL-#15539 · Wait for Async Results Load Without Spinning the Qt Event Loop (#15539)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15539
- **Branch:** codex/15539-results-wait
- **PR:** #15541
- **Paths:** see #15541
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`5a02b0c1`; collated from changes/15539-wait-for-async-results-load-without-spin.md)
- **Summary:** Wait for async results load without spinning the Qt event loop (#15539)
- **Next step:** Review the PR; operator handles merge authorization.

### DL-#15538 · Project Playback Ball Overlay Using the View Look-At (#15538)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15538
- **Branch:** codex/15538-overlay-lookat
- **PR:** #15542
- **Paths:** see #15542
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`ccbeee4e`; collated from changes/15538-project-playback-ball-overlay-using-the.md)
- **Summary:** Project playback ball overlay using the view look-at (#15538)
- **Next step:** Independent acceptance at the new head; production wiring remains tracked by open #15538.

### DL-#15524 · Preserve the Simulation Time Origin When Exporting Contact-Interval Playback Frames.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15524
- **Branch:** codex/15524-impact-playback-timebase
- **PR:** #15581, #15598, #15615, #15572, #15548, #15626, #15589
- **Paths:** see #15581
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`cc53d0ba`; collated from changes/15524-separation-gate.md)
- **Summary:** Preserve the simulation time origin when exporting contact-interval playback frames.
- **Next step:** Independent review; PR intentionally unarmed. Wiring the signal to ImpactZonePlayback (#15615) and the window source (#15598) is a follow-up once those land.

### DL-#15497 · Connect Bunker Release to Ball Flight and Exploratory Green Rollout

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15497
- **Branch:** codex-oglaptop-1/15497-flight-adapter
- **PR:** #15512, #15619, #15590, #15604, #15597, #15629, #15623
- **Paths:** see #15512
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`d4595e12`; collated from changes/15497-validate-bunker-flight-integration-time.md)
- **Summary:** Connect bunker release to ball flight and exploratory green rollout
- **Next step:** Independent review; keep PR unarmed per owner instruction.

### DL-#15494 · Add Sampled Feasible Component-Range Classification (Model Invariant, Prior-Constrained Narrow, Unmoved, Variable) With Tolerance Dependence (Slice B)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15494
- **Branch:** claude/15494-component-ranges
- **PR:** #15611, #15569, #15562
- **Paths:** see #15611
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`4b720f2b`; collated from changes/15494-strategy-torque-margins.md)
- **Summary:** Add sampled feasible component-range classification (model invariant, prior-constrained narrow, unmoved, variable) with tolerance dependence (Slice B)
- **Next step:** Merge the PR; wire the manifest into the video exporter.

### DL-#15399 · dE/Dt Diverging Segment Shading, Interface Transfer Glyphs, Swing-Derived Limits

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15399
- **Branch:** cursor/segment-net-power-shading-15399-4895
- **PR:** #15431
- **Paths:** see #15431
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`0c874082`; collated from changes/15399-segment-net-power-shading.md)
- **Summary:** dE/dt diverging segment shading, interface transfer glyphs, swing-derived limits
- **Next step:** CI quality-gate on PR.

### DL-#15390 · Robertson–Winter Interface Power (F·V + M·Ω), Per-Segment Balance Residual, MuJoCo Adapter

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15390
- **Branch:** cursor/rw-interface-power-15390-a314
- **PR:** #15393
- **Paths:** see #15393
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`73ec5d73`; collated from changes/15390-robertson-winter-interface-power.md)
- **Summary:** Robertson–Winter interface power (F·v + M·ω), per-segment balance residual, MuJoCo adapter
- **Next step:** CI on PR for #15390.

### DL-#15389 · Segment Kinetic-Energy Globes, Per-Frame KE Time Series API, and Translational/Rotational Split

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15389
- **Branch:** cursor/segment-ke-globe-view-f14a
- **PR:** #15391
- **Paths:** see #15391
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`889deecd`; collated from changes/15389-segment-ke-globe-overlay.md)
- **Summary:** Segment kinetic-energy globes, per-frame KE time series API, and translational/rotational split
- **Next step:** CI on PR #15391.

### DL-#15383 · Measure Native Skinned Body in Showpiece Lighting Checks

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15383
- **Branch:** codex-oglaptop-2/15383-showpiece-lighting
- **PR:** #15392
- **Paths:** see #15392
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`636f6e1a`; collated from changes/15383-measure-native-skinned-body-in-showpiece.md)
- **Summary:** Measure native skinned body in showpiece lighting checks
- **Next step:** Review the metric/test change, then qualify showpiece lighting (ragged shadow and clipping) against the acceptance bounds and remove the strict xfail.

### DL-#15244 · Derive Matched Swing Qualification Ladder From AcceptanceGates + Ledger Receipts

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15244
- **Branch:** cursor/matched-swing-ladder-15244-2d97
- **PR:** #15570
- **Paths:** see #15570
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`4c9f89a3`; collated from changes/15244-matched-swing-qualification-ladder.md)
- **Summary:** Derive matched swing qualification ladder from AcceptanceGates + ledger receipts
- **Next step:** Merge after CI green; regenerate README with generate_matched_swing_status --write when ledger changes.

### DL-#15238 · Split Positive and Negative Work and Impulse at the Zero Crossing Instead of Clipping the Trapezoid

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15238
- **Branch:** claude-desk/t4-work-zero-crossing
- **PR:** #15239
- **Paths:** see #15239
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`b0e4880c`; collated from changes/15238-power-work-zero-crossing-split.md)
- **Summary:** Split positive and negative work and impulse at the zero crossing instead of clipping the trapezoid
- **Next step:** Review the PR; operator handles merge authorization.

### DL-#15233 · Unify Spatial_Algebra Jcalc on Featherstone Passive X_J for Revolute and Prismatic Joints

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15233
- **Branch:** cursor/jcalc-featherstone-15233-2d97
- **PR:** #15560
- **Paths:** see #15560
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`1493bc1a`; collated from changes/15233-jcalc-featherstone-passive-convention.md)
- **Summary:** Unify spatial_algebra jcalc on Featherstone passive X_J for revolute and prismatic joints
- **Next step:** Merge after CI green; dynamics callers inherit the convention via shared jcalc.

### DL-#15232 · Motion Pipeline Resample Keeps the 1/Target_Fps Step and Rejects Invalid Target_Fps

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15232
- **Branch:** claude-desk/t4-resample-grid
- **PR:** #15234
- **Paths:** see #15234
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`f9877dfd`; collated from changes/15232-resample-grid-keeps-target-fps-step.md)
- **Summary:** Motion Pipeline Resample Keeps the 1/target_fps Step and Rejects Invalid target_fps
- **Next step:** Review the PR; operator handles merge authorization.

### DL-#15231 · Fix Replay Impact_to_Send_Latency_ms (Replay Quantization Plus Detect-to-Send Overhead)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15231
- **Branch:** cursor/impact-send-latency-15231-2d97
- **PR:** #15567
- **Paths:** see #15567
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`e93d70af`; collated from changes/15231-impact-to-send-latency-metric.md)
- **Summary:** Fix replay impact_to_send_latency_ms (replay quantization plus detect-to-send overhead)
- **Next step:** Merge after CI green; benchmark now measures presentation-to-send budget.

### DL-#15221 · Handle Overflow Through Safe Numeric Conversion Fallback and Strict Contracts

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15221
- **Branch:** codex/burn1116-15221-numeric-overflow
- **PR:** #15222
- **Paths:** src/shared/python/core/type_utils.py,tests/unit/core/test_type_conversion_overflow.py
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`e0535f8b`; collated from changes/15221-safe-conversion-overflow.md)
- **Summary:** Handle overflow through safe numeric conversion fallback and strict contracts
- **Next step:** CI and automated review

### DL-#15218 · Preserve Fresh Caller Defaults After JSON and YAML Load Failures

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15218
- **Branch:** codex/burn1116-15218-config-default
- **PR:** #15220
- **Paths:** src/shared/python/config/config_utils.py,tests/unit/config/test_config_load_failure_defaults.py
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`517d71b7`; collated from changes/15218-config-load-defaults.md)
- **Summary:** Preserve fresh caller defaults after JSON and YAML load failures
- **Next step:** CI and automated review

### DL-#15216 · Floor Grid Indices so Points Just Below the Raster Origin Are Off-Grid

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15216
- **Branch:** claude-desk/t4-puttsg
- **PR:** #15217
- **Paths:** see #15217
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`0c3ea407`; collated from changes/15216-floor-grid-indices-so-points-just-below.md)
- **Summary:** Floor grid indices so points just below the raster origin are off-grid
- **Next step:** Review the PR.

### DL-#15196 · Validate JSON and YAML Serialization Before Changing Destination Files

- **State:** in_review
- **Owner:** codex
- **Issue:** #15196
- **Branch:** fix/json-serialization-preserve
- **PR:** #15198
- **Paths:** see #15198
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`ff1575d1`; collated from changes/15196-validate-json-and-yaml-serialization-bef.md)
- **Summary:** Validate JSON and YAML serialization before changing destination files
- **Next step:** Review serialization-failure preservation and merge through the supervised pipeline

### DL-#15193 · Resolve Dotted Import Names and Version Checks on the Requested Module

- **State:** in_review
- **Owner:** codex
- **Issue:** #15193
- **Branch:** fix/dotted-module-imports
- **PR:** #15195
- **Paths:** see #15195
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`f424f4e3`; collated from changes/15193-resolve-dotted-import-names-and-version.md)
- **Summary:** Resolve dotted import names and version checks on the requested module
- **Next step:** Review dotted-module regression fix through the supervised pipeline

### DL-#15191 · Reject Invalid Seed Types Before Mutating Global RNG State

- **State:** in_review
- **Owner:** codex
- **Issue:** #15191
- **Branch:** fix/seed-validation-atomic
- **PR:** #15194
- **Paths:** see #15194
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`75c19da6`; collated from changes/15191-reject-invalid-seed-types-before-mutatin.md)
- **Summary:** Reject invalid seed types before mutating global RNG state
- **Next step:** Review seed precondition and regression evidence through the supervised pipeline

### DL-#15189 · Report Filesystem Errors Through Async and Background Save Callbacks

- **State:** in_review
- **Owner:** codex
- **Issue:** #15189
- **Branch:** fix/async-save-io-errors
- **PR:** #15190
- **Paths:** src/shared/python/data_io/_async_io.py; tests/output/wave5_output/test_async_io.py
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`45e57a32`; collated from changes/15189-report-filesystem-errors-through-async-a.md)
- **Summary:** Report filesystem errors through async and background save callbacks
- **Next step:** Review regression fix through the supervised pipeline

### DL-#15187 · Refresh the Full-Source Mypy Diagnostic Baseline and Mark Its Regression Test

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15187
- **Branch:** cursor/fix-full-src-mypy-baseline-4b58
- **PR:** #15188
- **Paths:** see #15188
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`ca9d8887`; collated from changes/15187-refresh-the-full-source-mypy-diagnostic.md)
- **Summary:** Refresh the Full-Source Mypy Diagnostic Baseline and Mark Its Regression Test
- **Next step:** Review PR #15188; operator handles merge authorization.

### DL-#15148 · Skinned Human Hand Visual Backend: MakeHuman CC0 Rig and Weights, Hierarchical LBS, Shared Curl-to-Contact Posing, EGL Evidence Renders.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15148
- **Branch:** cursor/skinned-human-hand-backend-751f
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`f637f68b`; collated from changes/15148-skinned-human-hand-visual-backend-makehu.md)
- **Summary:** Skinned human hand visual backend: MakeHuman CC0 rig and weights, hierarchical LBS, shared curl-to-contact posing, EGL evidence renders.
- **Next step:** Cup the palm and solve the trail hand against the lead hand; see PR #15149 score table.

### DL-#15144 · Calibrate MyoHand Anatomical Contact Solver, Bilateral Mirroring, Skeleton JSON Export, and 74% Z-Buffer Framing

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15144
- **Branch:** cursor/myohand-visual-backend-dcca
- **PR:** #15147
- **Paths:** see #15147
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`4c18f5cc`; collated from changes/15144-myohand-skeleton-visual-calibration.md)
- **Summary:** Calibrate MyoHand anatomical contact solver, bilateral mirroring, skeleton JSON export, and 74% Z-buffer framing
- **Next step:** Integrate shared posing.py (#15164) and render.py (#15154) when landed while keeping research_only=True until #13763 owner sign-off

### DL-#15138 · HRG-GB1b: Per-Hand Shaft Roll on Grip Attachment via Apply_Shaft_Roll and Cross-Engine Parity.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15138
- **Branch:** cursor/hrg-gb1b-shaft-roll-2882
- **PR:** #15139
- **Paths:** see #15139
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`4f30c6dc`; collated from changes/15138-hrg-gb1b-per-hand-shaft-roll-on-grip-att.md)
- **Summary:** HRG-GB1b: per-hand shaft roll on grip attachment via apply_shaft_roll and cross-engine parity.
- **Next step:** Merge after GB-1 #15133; GB-2 wires roll into the procedural builder.

### DL-#15128 · HRG-GB5b: Offline Grip-Loss Evaluator With Configurable Hysteresis and Committed-Series Prediction Reports.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15128
- **Branch:** cursor/hrg-gb5b-grip-loss-evaluator-4e08
- **PR:** #15136
- **Paths:** see #15136
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`6db6bf8b`; collated from changes/15128-hrg-gb5b-grip-loss-evaluator.md)
- **Summary:** HRG-GB5b: offline grip-loss evaluator with configurable hysteresis and committed-series prediction reports.
- **Next step:** Stack merges after GB-1; GB-6 MuJoCo clamp consumes the same threshold config pattern.

### DL-#15125 · HRG-GB3 Palm Wrap, Thumb Press, Trail Overlap/Interlock/Ten-Finger Linking, Golden Checksums

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15125
- **Branch:** cursor/hrg-gb3-grip-builder-fd55
- **PR:** #15137
- **Paths:** see #15137
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`5d7af863`; collated from changes/15125-hrg-gb3-palm-thumb-trail-goldens.md)
- **Summary:** HRG-GB3 palm wrap, thumb press, trail overlap/interlock/ten-finger linking, golden checksums
- **Next step:** Review stacked on GB-2 (#15134); merge queue after parent chain.

### DL-#15124 · HRG-GB2 Engine-Free Flat-Hand Curl Builder and Build Receipt Stacked on GB-1 GripSpec

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15124
- **Branch:** cursor/hrg-gb2-grip-builder-0932
- **PR:** #15137
- **Paths:** see #15137
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`5d7af863`; collated from changes/15124-hrg-gb2-engine-free-flat-hand-curl-build.md)
- **Summary:** HRG-GB2 engine-free flat-hand curl builder and build receipt stacked on GB-1 GripSpec
- **Next step:** Merge GB-1 (#15133) then GB-2 (#15134).

### DL-#15123 · HRG-GB1: Versioned Engine-Free GripSpec, GripPose Projection, Force/Contact Providers, ADR-0057 Proposed Addendum.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15123
- **Branch:** cursor/hrg-gb1-grip-spec-96ca
- **PR:** #15133
- **Paths:** see #15133
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`e652d497`; collated from changes/15123-hrg-gb1-grip-spec.md)
- **Summary:** HRG-GB1: versioned engine-free GripSpec, GripPose projection, force/contact providers, ADR-0057 Proposed addendum.
- **Next step:** Review and merge; GB-2 may consume GripSpec for the procedural builder.

### DL-#15115 · Local API App Applies the Shared Allowed-Host Policy

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15115
- **Branch:** claude-og/local-server-host-check
- **PR:** #15116
- **Paths:** see #15116
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`795caaad`; collated from changes/15115-local-api-app-applies-the-shared-allowed.md)
- **Summary:** Local API app applies the shared allowed-host policy
- **Next step:** Merge the PR.

### DL-#15112 · Fail Closed on Per-Crate Maturin Build Errors in Dockerfile and Enforce in check_dockerfile_contracts.py.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15112
- **Branch:** gemini-og/15112-dockerfile-maturin-fail-closed
- **PR:** #15114
- **Paths:** see #15114
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`7699de0c`; collated from changes/15112-dockerfile-maturin-fail-closed.md)
- **Summary:** Fail closed on per-crate maturin build errors in Dockerfile and enforce in check_dockerfile_contracts.py.
- **Next step:** Merge PR for #15112.

### DL-#15107 · Fix Broken Repo Paths in Curated Testing, Architecture, and Manual Docs and Enforce in check_spec_paths.py.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15107
- **Branch:** gemini-og/15107-docs-broken-repo-paths
- **PR:** #15113
- **Paths:** see #15113
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`6b454ac0`; collated from changes/15107-docs-broken-repo-paths.md)
- **Summary:** Fix broken repo paths in curated testing, architecture, and manual docs and enforce in check_spec_paths.py.
- **Next step:** Merge PR #15113.

### DL-#15086 · Restore Per-Workflow Docker CI Concurrency Groups and Per-Run Image Tags in Docker-Smoke (Fix-Forward).

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15086
- **Branch:** cursor/docker-ci-concurrency-fix-f3f5
- **PR:** #15167
- **Paths:** see #15167
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`a0409003`; collated from changes/15086-docker-ci-concurrency-fix-forward.md)
- **Summary:** Restore per-workflow Docker CI concurrency groups and per-run image tags in docker-smoke (fix-forward).
- **Next step:** Merge after CI green, including Docker Size and Health Check.

### DL-#13888 · Add HILL_MTU Drive Mode to 3DOF FTLE Study With Matched Nominal Excitation and Four-Mode Receipt.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13888
- **Branch:** cursor/hill-mtu-drive-mode-3dof-ftle-f296
- **PR:** #15076
- **Paths:** see #15076
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`1b4ce670`; collated from changes/13888-hill-mtu-drive-mode-3dof-ftle-study.md)
- **Summary:** Add HILL_MTU drive mode to 3DOF FTLE study with matched nominal excitation and four-mode receipt.
- **Next step:** Merge after CI; full t_end=0.2 Hill receipt rows remain NOT_QUALIFIED until MTU equilibrium converges on full swing.

### DL-#13782 · ML-WS-5 MuJoCo Corpus Run Receipt, Manifest-Verified Loader, and CI Evidence

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13782
- **Branch:** cursor/ml-ws5-corpus-receipt-loader-791f
- **PR:** #15075
- **Paths:** see #15075
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`0adc2797`; collated from changes/13782-ml-ws-5-mujoco-corpus-run-receipt-manife.md)
- **Summary:** ML-WS-5 MuJoCo corpus run receipt, manifest-verified loader, and CI evidence
- **Next step:** Barb review / CI

### DL-#13761 · Bump Dnspython to 2.9.2 and Tornado to 6.6.1; Remove Stale Tornado Pip-Audit Waivers.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13761
- **Branch:** cursor/pip-audit-dnspython-tornado-54d2
- **PR:** #15086
- **Paths:** see #15086
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`b6abbfbf`; collated from changes/13761-bump-dnspython-to-2-9-2-and-tornado-to-6.md)
- **Summary:** Bump dnspython to 2.9.2 and tornado to 6.6.1; remove stale tornado pip-audit waivers.
- **Next step:** Merge after CI green.

### DL-#13427 · Regression-Test Address Reserve Share; Wire Study Allow_Implausible and OS-9 Asset Gate

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13427
- **Branch:** cursor/spn-d9-address-reserve-share-ab3e
- **PR:** #15077
- **Paths:** see #15077
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`d56bdccf`; collated from changes/13427-regression-test-address-reserve-share-wi.md)
- **Summary:** Regression-test address reserve share; wire study allow_implausible and OS-9 asset gate
- **Next step:** Owner decides address trunk bound; merge qualified OS-9 native receipt when measured

### DL-#15045 · Hill F_V Zero Beyond Vmax; Exponential Activation Substeps for Mocap Dt

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #15045
- **Branch:** cursor/hill-activation-sci-15045-1ec6
- **PR:** #15054
- **Paths:** see #15054
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`59cb1fcb`; collated from changes/15045-hill-f-v-zero-beyond-vmax-exponential-ac.md)
- **Summary:** Hill f_v zero beyond Vmax; exponential activation substeps for mocap dt

### DL-#15044 · SO(3) Quat_Inverse_Distance (Relative-Quaternion Angle), Interpolated Ball-Flight Ground Landing

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #15044
- **Branch:** cursor/quat-landing-sci-15044-e727
- **PR:** #15056
- **Paths:** see #15056
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`67fef889`; collated from changes/15044-so-3-quat-inverse-distance-and-interpola.md)
- **Summary:** SO(3) quat_inverse_distance (relative-quaternion angle), interpolated ball-flight ground landing

### DL-#15043 · Align Rigid-Body Off-Center COR Impulse With Validator Budget via Club Torque; Solver Integration Test.

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #15043
- **Branch:** cursor/validate-energy-balance-15043-4fad
- **PR:** #15051
- **Paths:** see #15051
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`37a0e84f`; collated from changes/15043-energy-validator-uses-effective-club-mas.md)
- **Summary:** Align rigid-body off-center COR impulse with validator budget via club torque; solver integration test.

### DL-#15040 · Fix Turf Contact Impact: Joule Energy Loss, Receding Guard, Coupled Friction-Spin; Release-Safe Mass/Radius Validation

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #15040
- **Branch:** cursor/turf-contact-impact-15040-eeaa
- **PR:** #15048
- **Paths:** see #15048
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`b2212bc8`; collated from changes/15040-fix-turf-contact-impact-joule-energy-los.md)
- **Summary:** Fix turf contact impact: joule energy loss, receding guard, coupled friction-spin; release-safe mass/radius validation

### DL-#15022 · Agent API V1 Conformance CI and FastMCP Migration for Ud-Sim-Mcp

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #15022
- **Branch:** cursor/agent-api-v1-fastmcp-15022-1251
- **PR:** #15073, #15163
- **Paths:** see #15073
- **Started:** 2026-11-14
- **Last verified:** 2026-11-24 (`26334a68`; collated from changes/15022-agent-api-issue-closure.md)
- **Summary:** Agent API v1 conformance CI and FastMCP migration for ud-sim-mcp

### DL-#15015 · Fail-Closed Motion-Matching Gates; Archive Run-102 Evidence Script With Restore Ref.

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #15015
- **Branch:** cursor/motion-matching-gate-honesty-15015-2352
- **PR:** #15037, #15078, #15173, #15160, #15172, #15178, #15171, #15170
- **Paths:** see #15037
- **Started:** 2026-11-14
- **Last verified:** 2026-11-24 (`3a36793e`; collated from changes/15015-two-window-pelvis-yaw-nan-00d4.md)
- **Summary:** Fail-closed motion-matching gates; archive run-102 evidence script with restore ref.

### DL-#15014 · Fix Dempster/Zatsiorsky Segment Mass Expansion and C3D Export Placeholder (Part A)

- **State:** shipped
- **Shipped:** 2026-11-14
- **Owner:** unassigned
- **Issue:** #15014
- **Branch:** cursor/segment-mass-c3d-fix-e135
- **PR:** #15036, #15035, #15159, #15088
- **Paths:** see #15036
- **Started:** 2026-11-14
- **Last verified:** 2026-11-24 (`504086c0`; collated from changes/15014-mocap-conventions-and-de-leva-guard.md)
- **Summary:** Fix Dempster/Zatsiorsky segment mass expansion and C3D export placeholder (part a)

### DL-#13887 · Add Hill MTU Joint Actuator (Agonist/Antagonist, Compliant Tendon, Linearisation API).

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #13887
- **Branch:** cursor/hill-mtu-joint-actuator-cce3
- **PR:** #15062
- **Paths:** see #15062
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`1b0ef44a`; collated from changes/13887-add-hill-mtu-joint-actuator-agonist-anta.md)
- **Summary:** Add Hill MTU joint actuator (agonist/antagonist, compliant tendon, linearisation API).

### DL-#13781 · ML-WS-4 Native MuJoCo Episode Corpus Generator, Manifest Hashes, and CI Fixture

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #13781
- **Branch:** cursor/ml-ws4-mujoco-episode-corpus-b024
- **PR:** #15059
- **Paths:** see #15059
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`aa6cfb89`; collated from changes/13781-ml-ws-4-native-mujoco-episode-corpus-gen.md)
- **Summary:** ML-WS-4 native MuJoCo episode corpus generator, manifest hashes, and CI fixture

### DL-#12723 · NSX-94: Commit Cross-Engine Co-Contraction Trade-Off Evidence JSON and Validation Harness

- **State:** shipped
- **Shipped:** 2026-11-13
- **Owner:** unassigned
- **Issue:** #12723
- **Branch:** cursor/nsx94-tradeoff-report-3f62
- **PR:** #15067
- **Paths:** see #15067
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`90bfee5b`; collated from changes/12723-nsx-94-commit-cross-engine-co-contractio.md)
- **Summary:** NSX-94: commit cross-engine co-contraction trade-off evidence JSON and validation harness

### DL-#12690 · R9c: Catalog MATLAB R2025b Appearance Acceptance as Deferred; Extend Headless Export and MATLAB Consumer Contract Tests.

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #12690
- **Branch:** cursor/simscape-appearance-r9c-c25e
- **PR:** #15065
- **Paths:** see #15065
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`bbaf03c0`; collated from changes/12690-r9c-catalog-matlab-r2025b-appearance-acc.md)
- **Summary:** R9c: catalog MATLAB R2025b appearance acceptance as deferred; extend headless export and MATLAB consumer contract tests.

### DL-#15027 · Re-Bootstrap Pip With Ensurepip Before Force-Overlay-Parity Engine Installs (#15027).

- **State:** shipped
- **Shipped:** 2026-11-11
- **Owner:** unassigned
- **Issue:** #15027
- **Branch:** cursor/force-overlay-pip-ensurepip-a939
- **PR:** #15032
- **Paths:** see #15032
- **Started:** 2026-11-12
- **Last verified:** 2026-11-12 (`3afe09c1`; collated from changes/15027-re-bootstrap-pip-with-ensurepip-before-f.md)
- **Summary:** Re-bootstrap pip with ensurepip before force-overlay-parity engine installs (#15027).

### DL-#15011 · Label Injury Spinal-Load Heuristics Honestly; Disclaim Unvalidated Screening Bands.

- **State:** shipped
- **Shipped:** 2026-11-11
- **Owner:** unassigned
- **Issue:** #15011
- **Branch:** cursor/injury-heuristic-labels-118d
- **PR:** #15033
- **Paths:** see #15033
- **Started:** 2026-11-12
- **Last verified:** 2026-11-12 (`a1a26523`; collated from changes/15011-label-injury-spinal-load-heuristics-hone.md)
- **Summary:** Label injury spinal-load heuristics honestly; disclaim unvalidated screening bands.

### DL-#14934 · Fix Syngas Water Unit Conversions and Router Fallback on 60F_14.696Psia Basis (Paired With Tools#6009)

- **State:** shipped
- **Shipped:** 2026-11-10
- **Owner:** unassigned
- **Issue:** #14934
- **Branch:** agy/6004-syngas-water-router-pair
- **PR:** #14976, #14985, #14990, #14995
- **Paths:** see #14976
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`71515bb6`; collated from changes/14934-default-true-removal.md)
- **Summary:** Fix syngas water unit conversions and router fallback on 60F_14.696psia basis (paired with Tools#6009)

### DL-#14906 · Correct SHOW-A10 Evidence Provenance Manifest and Summary Attribution Without Regenerating Artifacts

- **State:** shipped
- **Shipped:** 2026-11-09
- **Owner:** unassigned
- **Issue:** #14906
- **Branch:** agy/ct/14906-show-a10-provenance-correction
- **PR:** #14920
- **Paths:** see #14920
- **Started:** 2026-11-09
- **Last verified:** 2026-11-09 (`94f794b0`; collated from changes/14906-show-a10-provenance-correction.md)
- **Summary:** Correct SHOW-A10 evidence provenance manifest and summary attribution without regenerating artifacts

### DL-#14397 · Canonical Float Rounding Before Hashing for Ubuntu 26.04

- **State:** shipped
- **Shipped:** 2026-11-08
- **Owner:** unassigned
- **Issue:** #14397
- **Branch:** agy/14397-ubuntu-2604-hash-drift
- **PR:** #14899
- **Paths:** see #14899
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`604503f6`; collated from changes/14397-ubuntu-2604-hash-drift.md)
- **Summary:** physics_spec_sha256 and the saved-comparison receipts compare floats at a canonical precision (12 significant digits, 1e-12 absolute) so BLAS/libm last-bit drift cannot move them (owner option A)

### DL-#14896 · CI Checkout Depth Qualification and Transfer Budget Policy

- **State:** shipped
- **Shipped:** 2026-11-08
- **Owner:** fleet-orchestrator
- **Issue:** #14896
- **Branch:** agy/14896-ci-checkout-depth-contracts
- **PR:** #14915, #14916
- **Paths:** scripts/ci/qualify_checkout_depth.py, tests/ci/test_qualify_checkout_depth.py, docs/development/evidence/CI_CHECKOUT_DEPTH_AND_TRANSFER_BUDGET_POLICY.md
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`0ec24365`; collated from changes/14896-ci-checkout-depth-contracts.md)
- **Summary:** Qualify job-specific CI checkout depth, transfer bandwidth budgets, and sparse-state regression contracts

### DL-#14895 · Source Asset Selective LFS Policy and Inventory Contracts

- **State:** shipped
- **Shipped:** 2026-11-08
- **Owner:** fleet-orchestrator
- **Issue:** #14895
- **Branch:** agy/14895-source-asset-lfs-policy
- **PR:** #14914
- **Paths:** scripts/assets/source_asset_inventory.py, tests/unit/assets/test_source_asset_inventory.py, docs/development/evidence/SOURCE_ASSET_LFS_POLICY.md
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`c6375e53`; collated from changes/14895-source-asset-lfs-policy.md)
- **Summary:** Selective Git LFS policy, source asset inventory contracts, and fail-closed hydration verification

### DL-#14894 · Research Artifacts: Checksum-Bound Storage Contracts

- **State:** shipped
- **Owner:** fleet-orchestrator
- **Issue:** #14894
- **Branch:** agy/14894-research-artifact-storage
- **PR:** #14911, #14912
- **Paths:** scripts/research/shot_pattern_analysis/fetch_release_assets.py, tests/unit/research/test_fetch_release_assets.py, docs/development/evidence/RESEARCH_ARTIFACT_STORAGE_POLICY.md
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`781dcc0b`; collated from changes/14894-research-artifact-storage-contracts.md)
- **Summary:** Research artifacts: checksum-bound durable storage, fail-closed offline/integrity verification, and evidence preservation policy
- **Next step:** Shipped.

### DL-#13158 · OSL-19: Grip Closure and OpenSim Reach Receipt

- **State:** shipped
- **Shipped:** 2026-11-08
- **Owner:** fleet-orchestrator
- **Issue:** #13158
- **Branch:** agy/13158-osl19-grip-closure-reach
- **PR:** #14802, #14907
- **Paths:** src/shared/python/osl_scoping/osl19.py, tests/fixtures/lifting/osl_scoping/osl19_scoping_v1.json, tests/unit/lifting/osl_scoping/test_osl19_scoping_contract.py
- **Started:** 2026-11-06
- **Last verified:** 2026-11-08 (`9e90db40`; collated from changes/13158-osl-19-grip-closure-reach.md)
- **Summary:** OSL-19: grip closure contracts and OpenSim reach hook for back-squat bundles

### DL-#13131 · OSL-17: Back-Squat Bar Rack and Rest Window

- **State:** shipped
- **Shipped:** 2026-11-08
- **Owner:** fleet-orchestrator
- **Issue:** #13131
- **Branch:** agy/13131-osl17-bar-rack-rest
- **PR:** #14802, #14903
- **Paths:** src/shared/python/lifting/back_rack.py, src/shared/python/osl_scoping/osl17.py, tests/unit/lifting/test_back_rack.py, tests/unit/lifting/osl_scoping/test_osl17_scoping_contract.py, docs/development/lifting/evidence/opensim_ik_id/receipts.json
- **Started:** 2026-11-06
- **Last verified:** 2026-11-08 (`ccd72956`; collated from changes/13131-osl-17-back-squat-bar-rack-and-rest.md)
- **Summary:** OSL-17: put back-squat bar on BackTop high-bar rack, verify lead-in settle and whole-window OpenSim Hicks and marker gates

### DL-#13093 · OSL-15: Regenerate Source Back Squat With Bar Load

- **State:** shipped
- **Shipped:** 2026-11-08
- **Owner:** fleet-orchestrator
- **Issue:** #13093
- **Branch:** agy/13093-osl15-load-balance-squat
- **PR:** #14904
- **Paths:** src/shared/python/osl_scoping/osl15.py, tests/fixtures/lifting/osl_scoping/osl15_scoping_v1.json, tests/unit/lifting/osl_scoping/test_osl15_scoping_contract.py
- **Started:** 2026-11-06
- **Last verified:** 2026-11-08 (`7b629286`; collated from changes/13093-osl-15-load-balance-squat.md)
- **Summary:** OSL-15: verify bar-load balance in IK provenance (combined CoM constrained over mid-foot) on back-squat barbell and source bundles

### DL-#11746 · LIFT-6: OpenSim Lift Composition and Static Hold Contracts

- **State:** shipped
- **Shipped:** 2026-11-08
- **Owner:** fleet-orchestrator
- **Issue:** #11746
- **Branch:** agy/11746-lift6-composition-static-hold
- **PR:** #14805, #14910
- **Paths:** src/shared/python/lift_scoping/lift6.py, tests/fixtures/lifting/lift_scoping/lift6_epic_v1.json, tests/unit/lifting/lift_scoping/test_lift6_epic_contract.py
- **Started:** 2026-11-06
- **Last verified:** 2026-11-08 (`c3813358`; collated from changes/11746-lift-6-composition-static-hold.md)
- **Summary:** LIFT-6: OpenSim full-body lift composition, 80-muscle count, static hold, and kinetics qualification contracts

### DL-#11745 · LIFT-5: Four-Engine Bar Path Parity Contract

- **State:** shipped
- **Shipped:** 2026-11-08
- **Owner:** fleet-orchestrator
- **Issue:** #11745
- **Branch:** agy/11745-lift5-four-engine-bar-path-parity
- **PR:** #14805, #14877, #14909
- **Paths:** src/shared/python/lift_scoping/lift5.py, tests/fixtures/lifting/lift_scoping/lift5_epic_v1.json, tests/unit/lifting/lift_scoping/test_lift5_epic_contract.py
- **Started:** 2026-11-06
- **Last verified:** 2026-11-08 (`960d2084`; collated from changes/11745-lift-5-four-engine-bar-path-parity.md)
- **Summary:** LIFT-5: four-engine bar path parity contract and tests covering MuJoCo, Drake, Pinocchio, and OpenSim

### DL-#13991 · Docs: ADR-0060 OpenSim Scaling and Fit Plan (#13991)

- **State:** shipped
- **Shipped:** 2026-11-06
- **Owner:** unassigned
- **Issue:** #13991
- **Branch:** cursor/opensim-scaling-panel-plan-13991-94df
- **PR:** #14803
- **Paths:** see #14803
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`320872fc`; collated from changes/13991-docs-adr-0060-opensim-scaling-and-fit-pl.md)
- **Summary:** docs: ADR-0060 OpenSim scaling and fit plan (#13991)

### DL-#14419 · Avoid Merge_Group Changed-Paths Checkout Hang and Sync Admission Publisher

- **State:** shipped
- **Shipped:** 2026-11-05
- **Owner:** unassigned
- **Issue:** #14419
- **Branch:** fix/14419-merge-group-checkout-depth
- **PR:** #14420
- **Paths:** see #14420
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`d100e477`; collated from changes/14419-avoid-merge-group-changed-paths-checkout.md)
- **Summary:** Avoid merge_group changed-paths checkout hang and sync admission publisher

### DL-#14417 · Sync the Admission Publisher Workflow and Provenance Digest With the Canonical Provider

- **State:** shipped
- **Shipped:** 2026-11-05
- **Owner:** unassigned
- **Issue:** #14417
- **Branch:** codex/14417-admission-publisher-sync
- **PR:** #14418
- **Paths:** see #14418
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`982d34d5`; collated from changes/14417-sync-the-admission-publisher-workflow-an.md)
- **Summary:** Sync the admission publisher workflow and provenance digest with the canonical provider

### DL-#14429 · Avoid OpenSim Classify Checkout Hang With Shallow Fetch

- **State:** shipped
- **Shipped:** 2026-11-05
- **Owner:** unassigned
- **Issue:** #14429
- **Branch:** agy/14429-opensim-classify-shallow-checkout
- **PR:** #14430
- **Paths:** see #14430
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`a9657f01`; collated from changes/14429-avoid-opensim-classify-checkout-hang-wit.md)
- **Summary:** Avoid OpenSim classify checkout hang with shallow fetch

### DL-#14535 · Size-Aware Development-Log Archiver: Archive Oldest Terminal Entries When the Log Nears the Byte Cap

- **State:** shipped
- **Shipped:** 2026-11-05
- **Owner:** unassigned
- **Issue:** #14535
- **Branch:** claude-ct/14535-devlog-size-trigger
- **PR:** #14592
- **Paths:** see #14592
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`8c6f82c1`; collated from changes/14535-size-aware-development-log-archiver-arch.md)
- **Summary:** Size-aware development-log archiver: archive oldest terminal entries when the log nears the byte cap

### DL-#14274 · Config/Secret Separation and Upgrade/Rollback Guarantees

- **State:** shipped
- **Shipped:** 2026-11-05
- **Owner:** unassigned
- **Issue:** #14274
- **Branch:** feat/14274-config-secrets
- **PR:** #14414, #14403
- **Paths:** see #14414
- **Started:** 2026-11-03
- **Last verified:** 2026-11-05 (`b281ba3e`; collated from changes/14274-root-dockerignore-secret-contract.md)
- **Summary:** Config/secret separation and upgrade/rollback guarantees

### DL-#14176 · Add Putting Reference Data and Validation Harness

- **State:** shipped
- **Shipped:** 2026-11-07
- **Owner:** unassigned
- **Issue:** #14176
- **Branch:** claude/putting-validation-14176
- **PR:** #14198, #14232, #14827
- **Paths:** see #14198
- **Started:** 2026-11-01
- **Last verified:** 2026-11-07 (`4ef29833`; collated from changes/14176-putting-make-rate-calibration.md)
- **Summary:** Add Putting Reference Data and Validation Harness

### DL-#11161 — Capture Registry, Capture Export and Swing Comparison (Owner Swing vs Tour Average)

- **Next step:** Coordinate the separately tracked engine deliveries in #11165–#11169 and cross-engine comparison #11170; keep this parent active until its scoped acceptance is met.
- **PR:** not recorded in source
- **State:** in_review
- **Owner:** claude
- **Issue:** #11161
- **Branch:** `feat/capture-registry-11161`
- **Paths:** `data/capture_registry.json`, `src/motion_capture/capture_registry.py`, `src/motion_capture/capture_export.py`, `src/shared/python/swing_comparison/`, `src/shared/python/motion_matching/leaderboard.py`, `src/shared/python/motion_matching/pipeline/`
- **Started:** 2026-09-29
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Capture-O parent coordinates a neutral capture registry/export and engine-independent swing comparison. Engine slices #11165 through #11169 and cross-engine comparison #11170 remain separately tracked.

### DL-#11421 · Dynamics-Informed Mocap Matching With ZTCF Prediction and Continuous Forward Replay

- **State:** in_progress
- **Owner:** antigravity
- **Issue:** #11421 (children #11422–#11430, #11434, #11435, #11437 closed; #11436 active)
- **PR:** #11436
- **Branch:** `feat/dime-11436-learned-initializers`
- **Paths:** `src/shared/python/estimation/dime_learned_initializers.py`, `src/shared/python/estimation/__init__.py`, `tests/unit/estimation/test_dime_learned_initializers.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Epic #11421: ZTCF drift prediction, kinematics-informed mocap matching, calibrated uncertainty weighting, uninterrupted forward simulation replay without frame-to-frame resetting.
- **Next step:** Create PR, verify CI Standard, squash-merge into origin/main.

### DL-#11313 · Necromatcher Video Export Force/Torque Layer

- **State:** in_review
- **Owner:** claude (Sonnet 5 CLI delegate)
- **Issue:** #11313; parent #11285
- **PR:** draft from `claude/issue-11313` (number on the PR; `Fixes #11313`)
- **Branch:** `claude/issue-11313`
- **Paths:** `src/shared/python/workspace/necromatcher_video_forces.py`, `src/shared/python/workspace/necromatcher_video.py`, `src/shared/python/workspace/necromatcher_video_worker.py`, `src/shared/python/workspace/necromatcher_video_jobs.py`, `src/api/routes/necromatcher.py`, `src/tools/necromatcher/video_dialog.py`, `ui/src/components/necromatcher/VideoExportControls.tsx`, `tests/unit/workspace/test_necromatcher_video_forces.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Off-by-default `force_layer {enabled, kinds, scale, segment_shading}` drawn through the shared FTO-8 renderer with the fit's camera; analytic v/a from the preserved spline (kinematic-only fits report the layer unavailable); legend states research-fit, not measured forces; manifest and per-frame receipts are deterministic.
- **Next step:** Frontier review of the draft PR, then rebase against paused draft #11359 (also edits `necromatcher_video.py`) before merge.

### DL-#11278 · COV-10 Model-Candidate Video Comparison

- **State:** in_progress
- **Owner:** claude
- **Issue:** #11278; parent #11268; blocked (real data) on #11165
- **PR:** not created (Refs #11278)
- **Branch:** `claude/issue-11278`
- **Paths:** `src/motion_capture/reference/model_candidate_source.py`, `tests/unit/motion_capture/test_cov10_model_candidate_source.py`, `docs/development/capture-o-video/procedure.md`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Python-only engine-generic path projecting any model candidate into video views and comparing it through the COV-7 L2 receipt; real Simscape evidence needs a stored `capture-O` candidate from #11165 and an R2025b host.
- **Next step:** After #11165 stores a candidate, run `compare_candidate_2d` on an R2025b host for every paired swing and record aggregate results on #11278.

### DL-#11268 · Capture-O Video Companion

- **State:** in_progress
- **Owner:** local
- **Issue:** #11268; children #11269–#11279; parent program #11161
- **PR:** #11502 (Closes #11271, Refs #11268)
- **Branch:** `feat/cov-3-protocol-decision-11271`
- **Paths:** `docs/development/capture-o-video/comparison-protocol.md`, `src/config/cov_comparison_profile.v1.json`, `src/config/cov_landmark_correspondence.json`, `tests/unit/motion_capture/test_cov3_protocol_and_correspondence.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Ratify comparison protocol L0-L3, frozen comparison profile, and landmark correspondence table.
- **Next step:** Merge PR for COV-3, complete Epic #11268.

- **PR:** #11499 (Closes #11272, #11274, #11275, #11276, #11277, #11279, Refs #11268)
- **Branch:** `feat/cov-benchmark-suite-11268`
- **Paths:** `src/motion_capture/reference/**`, `docs/development/capture-o-video/**`, `tests/unit/motion_capture/test_cov*`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f
- **Summary:** Complete implementation and verification of Capture-O Video benchmark suite spanning levels L0–L3, virtual camera projection, pairing confidence, marker-anchored subject scaling, 2D/3D residuals, and error budget reporting.
- **Next step:** Merge consolidated PR, close superseded individual draft PRs, and close Epic #11268.

- **PR:** #11416 (Closes #11270, Refs #11268)
- **Branch:** `feat/cov-2-register-sources-11270`
- **Started:** 2026-10-02
- **Last verified:** 2026-10-11 at e68baeabde8f
- **Summary:** Bring the owner's capture-session video album into the markerless, Necromatcher and engine-matching pipelines, compare against the `capture-O` marker capture at graded levels L0–L3, and publish a single-view reconstruction error budget. [Procedure](capture-o-video/procedure.md).
- **Next step:** Advance to COV-3 comparison protocol specification and level definitions (#11271).

- **PR:** #11413 (Closes #11269, Refs #11268)
- **Branch:** `feat/cov-1-acquisition-11269`
- **Started:** 2026-10-02
- **Last verified:** 2026-10-11 at e68baeabde8f
- **Summary:** Bring the owner's capture-session video album into the markerless, Necromatcher and engine-matching pipelines, compare against the `capture-O` marker capture at graded levels L0–L3, and publish a single-view reconstruction error budget. [Procedure](capture-o-video/procedure.md).
- **Next step:** Advance to COV-2 (#11270) video source registration and swing window grading.

- **PR:** #11417 (Closes #11273, Refs #11268)
- **Branch:** `feat/cov-5-runner-matrix-11273`
- **Started:** 2026-10-02
- **Last verified:** 2026-10-11 at e68baeabde8f
- **Summary:** Bring the owner's capture-session video album into the markerless, Necromatcher and engine-matching pipelines, compare against the `capture-O` marker capture at graded levels L0–L3, and publish a single-view reconstruction error budget. [Procedure](capture-o-video/procedure.md).
- **Next step:** Advance to COV-6 validation harness and error budget report (#11274).

### DL-#11095 — Qualify OpenSim Native Dual-Club Dynamics and Replay

- **Next step:** Native qualification remains unachieved. Run only on the corresponding pinned native lane with complete source-bound evidence; this metadata review did not run tests or native code.
- **Started:** not recorded in source
- **PR:** not recorded in source
- **Owner:** not recorded in source
- **State:** in_review
- **Issue:** #11095
- **Branch:** `feat/mmr-10o-opensim-dual-club-11095`
- **Paths:** `src/engines/physics_engines/opensim/python/native_qualification.py`, `tests/unit/engines/opensim/test_opensim_dual_club_qualification.py`, `docs/development/matched_swing_program/evidence/opensim/`
- **Last verified:** 2026-09-29 at d5dd606ffa6dafc11408614676a2576764a5e092 — 17 unit tests passed; red evidence recorded for the 7 new fail-closed tests against the pre-fix placeholder path; ruff check and format clean on changed files. Native OpenSim qualification NOT achieved: opensim bindings unavailable on all reachable hosts.
- **Summary:** OpenSim receipts now fail closed on missing runtime/evidence; fabricated marker metrics, unenforced tolerances and placeholder hashes were removed. Native qualification remains blocked without the pinned bindings.

### DL-#11096 — Qualify MyoSuite Native Dual-Club Dynamics and Replay

- **Next step:** Native qualification remains unachieved. Run only on the corresponding pinned native lane with complete source-bound evidence; this metadata review did not run tests or native code.
- **Started:** not recorded in source
- **PR:** not recorded in source
- **Owner:** not recorded in source
- **State:** in_review
- **Issue:** #11096
- **Branch:** `feat/mmr-10m-myosuite-dual-club-11096`
- **Paths:** `src/engines/physics_engines/myosuite/python/native_qualification.py`, `tests/unit/engines/myosuite/test_myosuite_dual_club_qualification.py`, `docs/development/matched_swing_program/evidence/myosuite/`
- **Last verified:** 2026-09-29 at 613b0e28ed88b5af4719ecccfe876d5d9db90419 — 19 unit tests passed; red evidence recorded for the 7 new fail-closed tests against the pre-fix placeholder path; ruff check and format clean on changed files. Native MyoSuite qualification NOT achieved: myosuite/MuJoCo unavailable on all reachable hosts.
- **Summary:** MyoSuite receipts fail closed on unavailable runtime and missing/invalid rollout, marker, derivative or quaternion evidence; fabricated metrics and unenforced tolerances were removed. Native qualification remains unavailable.

### DL-#11094 — Qualify Drake Native Dual-Club Dynamics and Replay

- **Started:** not recorded in source
- **PR:** not recorded in source
- **State:** in_review
- **Owner:** UDFixTrio10x
- **Issue:** #11094
- **Branch:** `feat/mmr-10d-drake-dual-club-11094`
- **Paths:** `src/engines/physics_engines/drake/python/native_qualification.py`, `tests/unit/engines/drake/test_drake_dual_club_qualification.py`, `scripts/ci/run_native_engine_lane.py`, `scripts/ci/run_native_engine_lane.sh`, `tests/scripts/test_run_native_engine_lane.py`, `docs/development/matched_swing_program/evidence/nightly/drake_receipt.json`, `docs/development/matched_swing_program/evidence/drake/`
- **Last verified:** 2026-09-29 at 3b23bd41502a7b7d028f291564997b6460f14cbc — 36 focused tests passed (test_drake_dual_club_qualification.py 16, test_run_native_engine_lane.py + test_native_lane_freshness.py 20); red evidence recorded for the 7 new fail-closed tests against the pre-fix placeholder path; ruff check and format clean on changed files. Native Drake qualification NOT achieved: pydrake unavailable on all reachable hosts.
- **Summary:** Drake receipts now fail closed on missing runtime/evidence; fabricated marker metrics and placeholder hashes were removed, and Drake entered the nightly lane. Native qualification remains blocked without pydrake.
- **Next step:** Merge drivers follow; native qualification remains blocked on engine availability (recorded in receipts, not silently).

### DL-#10286 · Native Swing ZTCF/ZVCF Fail-Closed Dynamics and DTACK Semantics

- **State:** in_review
- **Owner:** claude (agy executor)
- **Issue:** #10286
- **PR:** draft PR from `agy/ud-10286-fail-closed`
- **Branch:** `agy/ud-10286-fail-closed`
- **Paths:** `src/engines/physics_engines/myosuite/python/_drift_control.py`, `src/engines/physics_engines/pendulum/python/golf_swing_physics_engine.py`, `src/engines/physics_engines/pinocchio/python/dtack/gui/main_window.py`, `src/engines/physics_engines/pinocchio/python/dtack/sim/dynamics.py`, `tests/engines/physics_engines/test_golf_swing_pendulum.py`, `tests/engines/physics_engines/test_myosuite_engine.py`, `tests/unit/engines/pinocchio/dtack/sim/test_dynamics.py`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Remove success-shaped empty/zero array returns from uninitialized ZTCF/ZVCF counterfactual methods (raising StateError and ValueError for invalid dimensions), update dtack compute_zvcf to canonical (v=0, tau=0) semantics, add compute_zero_velocity_controlled for control-preserved dynamics, and update GUI caller and button label to "Zero-velocity (control kept)".
- **Next step:** CI green, review, merge.

### DL-#9619 · Amend ADR-0041 to Record Consumer-Side Fitter Decision

- **PR:** not recorded in source
- **State:** in_review
- **Owner:** local (session `claude-deskcomputer-20260925-ud`)
- **Issue:** #9619 (companion #9630)
- **Branch:** `agy/ud-9619-adr0041`
- **Paths:** `docs/adr/0041-markerless-mocap-consumer-authority.md`, `tests/architecture/test_markerless_mocap_authority.py`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Amended ADR-0041 to record that consumer-side self-calibration fitters live in UpstreamDrift (`src/motion_capture/reconstruct/`) while Tools maintains vendor-neutral reference geometry (#9630, #9619).
- **Next step:** CI green, review, merge.

### DL-#9688 · Bunker P1: Decide and Benchmark a Genuine 3D Sand-Motion Tier

- **State:** in_review
- **Owner:** antigravity
- **Issue:** #9688
- **PR:** draft PR from `agy/ud-9688-capability-register`
- **Branch:** `agy/ud-9688-capability-register`
- **Paths:** `src/bunkershot3d/solvers/capability.py`, `src/bunkershot3d/solvers/exceptions.py`, `src/bunkershot3d/solvers/__init__.py`, `docs/adr/0044-out-of-plane-fidelity-for-bunkershot3d.md`, `tests/bunkershot3d/solvers/test_capability_9688.py`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Added fail-closed sand-motion capability register (SandMotionKind, SandMotionCapability, register mapping, and require\_\* guards) in bunkershot3d.solvers to prevent presenting F0/F1/proxy/tracer outputs as genuine 3-D individual-grain trajectories or spherical ball spin.
- **Next step:** CI green, review, merge.

### DL-#9613 · Rig Soak: Multi-Camera Repeat Record Mode

- **State:** in_review
- **Owner:** antigravity
- **Issue:** #9613
- **PR:** draft PR from `agy/ud-9613-record-repeat`
- **Branch:** `agy/ud-9613-record-repeat`
- **Paths:** `src/motion_capture/rig/__main__.py`, `src/motion_capture/rig/soak.py`, `tests/motion_capture/rig/test_record_repeat.py`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Added `record --repeat N [--pause S]` soak mode to camera rig CLI, supporting N back-to-back takes in take subdirectories, machine-readable soak summary, and worst-take exit code.
- **Next step:** CI green, review, merge.

### DL-#9411 · Keep MyPy Exclusion Budget and Unit-Gate Quarantine Ratchets Green

- **State:** in_review
- **Owner:** claude (agy executor)
- **Issue:** #9411
- **PR:** draft PR from `claude/ud-9411-unquarantine-passing`
- **Branch:** `claude/ud-9411-unquarantine-passing`
- **Paths:** `scripts/config/unit_gate_quarantine.json`, quarantined unit test files, `src/shared/python/validation_pkg/data_fitting.py`, `src/tools/model_explorer/mujoco_viewer.py`, `src/shared/python/ai/adapters/`, `src/api/routes/__init__.py`, `src/shared/python/physics/impact_model/utils.py`, `src/shared/python/pendulum_simulator/physics.py`
- **Started:** 2026-09-25
- **Last verified:** 2026-09-26 — #10997 Linux run: kernel built and imported, none of the 12 retired BunkerShot IDs failed; the tests the kernel un-skipped exposed #11000 (fixed in #11004) and #11003 (fixed in #11010), and an untracked root `Cargo.lock` (fixed); replayed onto main d716c698f, ledger contract passes at 55 IDs.
- **Summary:** Retire unit-gate quarantine IDs whose tests pass on the Linux CI lane after fixing real defects (shoulder FK origin), deduplicating the `data_fitting` and `mujoco_viewer` coordinators onto their existing helper modules, and retargeting stale tests. Tools-owned `ai` adapter changes and the gear-effect sign question are out of scope here (Tools#5362 / a physics-convention decision). Earlier slice (#10973) landed the mypy budget and 43 retirements. Rust slice (#10997): unit-test-gate maturin-builds upstream-physics into its venv (ball_simulator enforces strict Rust parity), retiring the 12 BunkerShot workbench/GUI IDs. Consolidated delivered subscopes from the preserved source rows: Claude #10995 (claude/ud-9411-vendor-bump-safe-eval) advanced the Tools pin and retired 10 quarantined tests; Antigravity #10999 (feat/ud-9411-quarantine-burndown-batch1) retired 6; #11005 (feat/ud-9411-ai-adapters-convergence) converged Gemini/BitNet adapters and retired 5; #11013 (feat/ud-9411-quarantine-batch2) retired 1 and narrowed optional-dependency mock scope. The source snapshot retains original owners, branches, paths, verification text and next steps. Broader parent mypy/quarantine readiness remains in review; no full ratchet completion is claimed.
- **Next step:** Complete parent-level mypy-exclusion and quarantine acceptance against a source-bound Linux unit-gate run, then reconcile the remaining eligible IDs. The delivered child tranches do not establish full parent readiness; the original child next steps are preserved in the source snapshot.

### DL-#9703 · Versioned Pre-Impact Bundle for Tools Impact Kernels

- **State:** in_review
- **Owner:** claude
- **Issue:** #9703 (IA-U2; parent #9700)
- **PR:** draft (see branch)
- **Branch:** `feat/9703-pre-impact-bundle`
- **Paths:** `src/shared/python/physics/pre_impact_bundle.py`, `src/shared/python/physics/_pre_impact_contracts.py`, `src/shared/python/physics/_pre_impact_frames.py`, `tests/shared_contracts/test_pre_impact_bundle.py`, `docs/development/impact_acoustics_program.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-09-25 — 48/48 tests pass against vendor pin a9ed0e7c5 (incl. python -O); pre-push mypy and unit hooks pass (Rust wheel hidden, #10946)
- **Summary:** Consumer-side PreImpactBundle v1 composing Tools conventions: provenance, bounded timebase, world/head/grip poses, head COM inertia and contact, ball, reduced shaft modal state and prestress field, per-hand wrench/impedance, per-field origin; absent fields raise; power-invariant transforms; energy-reporting modal projection. Engine adapters and installed-wheel fixtures remain.
- **Next step:** Review the draft PR, then implement the first engine adapter against the bundle under #9703.

### DL-#10342 · OpenSim/MyoSuite Native Nightly Lane Receipts

- **PR:** not recorded in source
- **State:** in_review
- **Owner:** local
- **Issue:** #10342 (MS-43, epic #10363)
- **Branch:** fix/issue-10342-ms-43-native-lane-local
- **Paths:** scripts/ci/run_native_engine_lane.py; scripts/ci/run_native_engine_lane.sh; docs/development/matched_swing_program/evidence/nightly/; tests/docs/test_native_lane_freshness.py; tests/scripts/test_run_native_engine_lane.py
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Added idempotent native-engine lane runner emitting hashed nightly receipts for OpenSim and MyoSuite (`requires_opensim` / `requires_myosuite` pytest markers), freshness gate warning at seven days and failing at thirty days, and fleet native-host verification docs without editing `.github/workflows`.
- **Next step:** Refresh receipts on the fleet native host opensim-10003 venv; merge PR closing #10342.
- **Evidence:** docs/development/matched_swing_program/evidence/nightly/opensim_receipt.json; docs/development/matched_swing_program/evidence/nightly/myosuite_receipt.json; tests/docs/test_native_lane_freshness.py.

### DL-#9162 · Local Branch Triage

- **PR:** not recorded in source
- **State:** in_progress
- **Owner:** claude
- **Issue:** #9162 (rollout Repository_Management#1460; no-seed bootstrap #9161)
- **Branch:** conductor/issue-9162
- **Paths:** docs/development/branch_triage_9162.md
- **Started:** 2026-09-15
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Disposition ledger for every local branch in the primary checkout: 3 protected, 27 deferred to the worktree pass, 34 live (merged sweep then DL entry), 178 stale and 144 abandoned scratch branches (bundle snapshot then delete). The runbook in the ledger executes the deletions and the coordination notice for agent-owned branches.
- **Next step:** Execute the runbook in docs/development/branch_triage_9162.md from the primary checkout and fill in its Outcome section.

### DL-#9410 · Adversarial Product Review Remediation Epic

- **PR:** not recorded in source
- **State:** in_progress
- **Owner:** claude
- **Issue:** #9410 (program Repository_Management#1505; children #8820–#8943, #8360, #8641, #8843, #8846, #8853, #8861–#8870, #8874–#8876, #8894)
- **Branch:** conductor/issue-9410
- **Paths:** docs/development/adversarial_review_remediation_9410.md
- **Started:** 2026-09-16
- **Last verified:** 2026-09-16 (SELF; every child re-checked against `db4fe88c4` by first-parent history search plus source reads of each residual)
- **Summary:** Reconciliation ledger for the 2026-08-21 adversarial review: 34 of 61 children landed on `main` (SHAs recorded), 27 residual. Three residual fixes already exist on unmerged branches (`readiness/p0-9412-one-tile-registry`, `conductor/issue-8865`, `conductor/issue-8866`). Epic acceptance (one registry, one API factory, one C3D reader, one pose type, no GUI-thread simulation) is not met on `main`.
- **Next step:** Rebase and merge `readiness/p0-9412-one-tile-registry`, then re-verify the Cluster B rows in the ledger.

### DL-#10602 · Club-Only Motion Matching Plan

- **State:** in_review
- **Owner:** codex
- **Issue:** #10602
- **Branch:** `agy/ud-10602-real-matrix`
- **PR:** draft PR from `agy/ud-10602-real-matrix`
- **Paths:** docs/plans/club_neural_review/; docs/plans/club_only_matching/; docs/plans/neural_motion_matching/; src/shared/python/motion_matching/club_only/
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Published bounded implementation issues with TDD/DbC/LoD/DRY prompts, dependency ordering, native validation gates and shared technical review. Planning artifacts do not qualify physical results or speedup.
- **Next step:** CI green, review, merge.
- **Evidence:** docs/plans/club_neural_review/REVIEW.md; docs/plans/club_neural_review/excel_audit.json.

### DL-#10363 · Matched Swing Continuation Review & Drake G1 Retraction

- **State:** in_review
- **Owner:** claude-deskcomputer-20260925-ud
- **Issue:** #10363
- **Branch:** agy/ud-10363-drake-retraction
- **PR:** draft PR from `agy/ud-10363-drake-retraction`
- **Paths:** docs/development/matched_swing_program; src/shared/python/motion_matching/acceptance.py; src/engines/physics_engines/drake/python/full_body_fit.py; evidence/matched/driver_g1_drake/reevaluation.json; docs/development/full_body_models/evidence/acceptance/verdicts_2026-09.json; reports/matched_swing_ledger.json
- **Started:** 2026-09-18
- **Last verified:** 2026-09-25 at ff9fbee62 (retracted Drake G1 fabricated PASS via reevaluation.json; closed synthesis paths in full_body_fit.py; added fail-closed integrity gates in acceptance.py; 0 PASSED rows in ledger; motion_matching/tour_baselines/opensim ladder/browser suites green except 13 pre-existing local failures)
- **Summary:** 2026-09-18 continuation review (codex) preserved checkpoint/source recovery evidence and bounded the cheaper-agent continuation, with no physical acceptance claimed. Then: retracted the fabricated Drake G1 receipt (PR #10506) under MS-100; added fail-closed evidence integrity gates (\_evaluate_evidence_integrity) rejecting placeholder hashes and zero residuals without replay evidence; removed physical_audit numeric literals, np.zeros target fallback, and silent warm-start fallback in Drake full_body_fit.py; regenerated ledger and status matrices confirming 0 PASSED rows.
- **Next step:** CI green, review, merge.
- **Evidence:** docs/development/matched_swing_program/evidence/continuation_20260918/matching-handoff-snapshot.json; evidence/matched/driver_g1_drake/reevaluation.json; reports/matched_swing_ledger.json; tests/unit/motion_matching/test_acceptance.py; tests/unit/motion_matching/test_drake_full_body_fit.py.

### DL-#10362 · Tools Dependency Gate for Matched Swing Program

- **State:** in_progress
- **Owner:** local
- **Issue:** #10362 (epic #10363)
- **Branch:** feat/ms95-tools-dependency-gate-10362
- **PR:** open
- **Paths:** vendor/ud-tools; Cargo.toml; requirements-tools.txt; docs/shared_tools/divergence_inventory.md; docs/shared_tools/divergence_inventory.v1.json; docs/agent_context/README.md; docs/agent_context/index.html
- **Started:** 2026-09-17
- **Last verified:** 2026-09-17 at 62e8cdbf9 (SELF; Tools main green, Tools #4494 and #4262 closed, Tools #5227 landed; four-way pin bumped to 62e8cdbf9; divergence inventory regenerated; agent context verified; test_no_shadow_of_tools_shared and run_checks pass)
- **Summary:** UpstreamDrift ownership of humanoid_character_builder and model_generation ruled per Tools #4494; Tools #4262 and #4494 closed; vendor/ud-tools, requirements-tools.txt, Cargo.toml, and divergence inventory repinned to Tools main 62e8cdbf9.
- **Next step:** Open PR referencing Closes #10362, enable auto-merge, release lease.
- **Evidence:** docs/shared_tools/divergence_inventory.v1.json; tests/unit/repo_hygiene/test_no_shadow_of_tools_shared.py; tests/fixtures/reference_calibration/run_checks.py.

### DL-#9967 · Native Simscape Tour Matching

- **PR:** not recorded in source
- **State:** in_progress
- **Owner:** codex (turnover review; execution ownership by next lease)
- **Issue:** #9967 (parent #9921)
- **Branch:** feat/9967-native-simscape-pinocchio
- **Paths:** src/shared/python/motion_matching; docs/development/simscape_tour_matching
- **Started:** 2026-09-09
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Run101 improves yaw and has measured R2025b–Pinocchio prefix agreement of 0.0605 mm maximum. Terminal RMS 40.31 mm fails the 35 mm gate; full 1.814 s capture is incomplete. No optimizer launched by this review.
- **Next step:** Follow COMPLETION_HANDOFF_20260916.md: restore clean-runtime native providers, coordinate #10260 finite-weld derivative qualification, produce articulated feasibility evidence, fit0.90 s and deliver repeatable reports. Run102 remains rejected; no new compute launched by this review.

## Shipped (Last 90 Days)

### DL-#2725 · Remove the Retired Vendored GAAI Framework Tree, Its Launcher and Slash Commands; .claude/ Is Now the Canonical Agent Configuration Root

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #2725
- **Branch:** merged via #15119
- **PR:** #15119
- **Paths:** see #15119
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`e63b3a62`; collated from changes/2725-remove-the-retired-vendored-gaai-framewo.md)
- **Summary:** Remove the retired vendored GAAI framework tree, its launcher and slash commands; .claude/ is now the canonical agent configuration root
- **Next step:** Shipped in PR #15119.

### DL-#15670 · The Model Explorer Demo Swing (Trunk_Visuals/Driver_V2_Bundle.Npz) No Longer Falls Over About 0.8 s in. Its V2 Spec Now Carries the Run's Hip, Ankle, Shank and Ground Calibration (Carry_Calibration), so the Pelvis Stays Within 2.3 cm (Height) of the Run's Target. the Evidence Generator's Repository Root Is Fixed as Well.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15670
- **Branch:** merged via #15680
- **PR:** #15680
- **Paths:** see #15680
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`60bf6549`; collated from changes/15670-driver-v2-bundle-calibrated.md)
- **Summary:** The Model Explorer demo swing (trunk_visuals/driver_v2_bundle.npz) no longer falls over about 0.8 s in. Its v2 spec now carries the run's hip, ankle, shank and ground calibration (carry_calibration), so the pelvis stays within 2.3 cm (height) of the run's target. The evidence generator's repository root is fixed as well.
- **Next step:** Shipped in PR #15680.

### DL-#15659 · Canonical Golfer Gate Refuses Capsule, Loft and Unrevised Specs Before Simulation; Solution Bundles Carry a Plant/Controller/Solver Model Lock and Unlocked Bundles Are Replay-Only

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15659
- **Branch:** merged via #15668
- **PR:** #15668
- **Paths:** see #15668
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`36aec7db`; collated from changes/15659-canonical-golfer-gate.md)
- **Summary:** Canonical golfer gate refuses capsule, loft and unrevised specs before simulation; Solution Bundles carry a plant/controller/solver model lock and unlocked bundles are replay-only
- **Next step:** Shipped in PR #15668.

### DL-#15654 · The Spec Builder Moves Specs With Physical Grip Picks to the Current Golfer Model, Posing the Skinned Hands From Subject.Grip_Config; Only Left-Handed Specs Keep the Deprecated Fallback.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15654
- **Branch:** merged via #15677
- **PR:** #15677, #15656
- **Paths:** see #15677
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`7ee0859b`; collated from changes/15654-skinned-hands-grip-config.md)
- **Summary:** The spec builder moves specs with physical grip picks to the current golfer model, posing the skinned hands from subject.grip_config; only left-handed specs keep the deprecated fallback.
- **Next step:** Shipped in PR #15677.

### DL-#15652 · Ratchet Test: Committed Specs and Bundles on a Deprecated Golfer Model Part May Only Leave Src/Config/Golfer_Model_Legacy_Assets.Json, Never Join It.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15652
- **Branch:** merged via #15658
- **PR:** #15658, #15655
- **Paths:** see #15658
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`eab8469b`; collated from changes/15652-left-handed-skinned-hands.md)
- **Summary:** Ratchet test: committed specs and bundles on a deprecated golfer model part may only leave src/config/golfer_model_legacy_assets.json, never join it.
- **Next step:** Shipped in PR #15658.

### DL-#15650 · Resolve Four Mypy Arg-Type and Operator Errors in cross_engine_replay.py That Blocked Pre-Push Hooks

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15650
- **Branch:** merged via #15653
- **PR:** #15653
- **Paths:** see #15653
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`4fb75a6b`; collated from changes/15650-fix-mypy-cross-engine-replay.md)
- **Summary:** Resolve four mypy arg-type and operator errors in cross_engine_replay.py that blocked pre-push hooks
- **Next step:** Shipped in PR #15653.

### DL-#15648 · Build_Anthropometric_Spec Writes Right-Handed Specs on the Current Golfer Model (Skinned Hand Wrists, Skinned Body); Left-Handed Specs Log the Deprecated Fallback Until Left-Handed Skinned Hands Land.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15648
- **Branch:** merged via #15657
- **PR:** #15657, #15651, #15678
- **Paths:** see #15657
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`937979e1`; collated from changes/15648-golfer-reference-specs.md)
- **Summary:** build_anthropometric_spec writes right-handed specs on the current golfer model (skinned hand wrists, skinned body); left-handed specs log the deprecated fallback until left-handed skinned hands land.
- **Next step:** Shipped in PR #15657.

### DL-#15523 · Add Iron Divot Model With Fat-Shot Pre-Ball Turf Contact (Physics/divot_model.py)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15523
- **Branch:** merged via #15545
- **PR:** #15545, #15617, #15583, #15606, #15596
- **Paths:** see #15545
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`1bcb6da4`; collated from changes/15523-iron-ground-contact.md)
- **Summary:** Add iron divot model with fat-shot pre-ball turf contact (physics/divot_model.py)
- **Next step:** Shipped in PR #15545.

### DL-#15522 · Add Headless Qt End-to-End Tests Driving the Ball Flight GUI Simulate Click Through Real Physics

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15522
- **Branch:** merged via #15543
- **PR:** #15543, #15631, #15577, #15568, #15647, #15633, #15634, #15622, #15593, #15628, #15640, #15614, #15618, #15621, #15637, #15561, #15636, #15612, #15563, #15643, #15645, #15601, #15599, #15579, #15605, #15627, #15638, #15588, #15639, #15591, #15607, #15644, #15608
- **Paths:** see #15543
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`c220a014`; collated from changes/15522-unified-dashboard-qtest-e2e.md)
- **Summary:** Add headless Qt end-to-end tests driving the Ball Flight GUI Simulate click through real physics
- **Next step:** Shipped in PR #15543.

### DL-#15493 · Add Versioned Solution Bundle Schema and Strict JSON+NPZ Codec for Replayable Optimized Swings (Slice 2).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15493
- **Branch:** merged via #15564
- **PR:** #15564, #15595, #15624, #15625, #15573, #15565, #15552, #15600
- **Paths:** see #15564
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`49723453`; collated from changes/15493-solution-bundle-time-domain.md)
- **Summary:** Add versioned Solution Bundle schema and strict JSON+NPZ codec for replayable optimized swings (Slice 2).
- **Next step:** Shipped in PR #15564.

### DL-#15485 · Regression Test and Citation Tightening for Wind Consumers in the Flight Shortcomings Register (#15482 Review)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15485
- **Branch:** merged via #15550
- **PR:** #15550
- **Paths:** see #15550
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`d1a76787`; collated from changes/15485-qualify-wind-consumer-register.md)
- **Summary:** Regression test and citation tightening for wind consumers in the flight shortcomings register (#15482 review)
- **Next step:** Shipped in PR #15550.

### DL-#15444 · Normalize MuJoCo Geom_Rgba Ndarrays Before MeshCat Material Binding (#15444).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15444
- **Branch:** merged via #15525
- **PR:** #15525
- **Paths:** see #15525
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`23d638b5`; collated from changes/15444-normalize-mujoco-geom-rgba-ndarrays-befo.md)
- **Summary:** Normalize MuJoCo geom_rgba ndarrays before MeshCat material binding (#15444).
- **Next step:** Shipped in PR #15525.

### DL-#15443 · Add Reference-Shot Regression for the Ball-Flight Model: Published Driver/Iron/Wedge Launches With Carry, Apex and Descent Bands; Rows the Model Misses Are Strict Xfails With Measured Errors

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15443
- **Branch:** merged via #15450
- **PR:** #15450, #15498, #15482, #15500, #15501, #15489, #15488, #15460, #15473, #15474, #15492, #15499, #15490, #15466, #15468, #15454, #15506, #15502, #15646, #15486, #15475, #15496
- **Paths:** see #15450
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`87d68a6f`; collated from changes/15443-shot-shape-views-aim-at-the-finish-plan.md)
- **Summary:** Add reference-shot regression for the ball-flight model: published driver/iron/wedge launches with carry, apex and descent bands; rows the model misses are strict xfails with measured errors
- **Next step:** Shipped in PR #15450.

### DL-#15435 · S4 Angmom Independent Oracle; Ball Excluded From Whole-Body H; Root_Body_Id Deprecated

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15435
- **Branch:** merged via #15544
- **PR:** #15544
- **Paths:** see #15544
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`5795bbc0`; collated from changes/15435-s4-angmom-oracle-ball-com-consistency.md)
- **Summary:** S4 angmom independent oracle; ball excluded from whole-body H; root_body_id deprecated
- **Next step:** Shipped in PR #15544.

### DL-#15429 · Reject Non-Finite Flight Inset Positives; Pin Headless Native Verification Clip

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15429
- **Branch:** merged via #15527
- **PR:** #15527
- **Paths:** see #15527
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`0173406c`; collated from changes/15429-reject-non-finite-flight-inset-positives.md)
- **Summary:** Reject non-finite flight inset positives; pin headless native verification clip
- **Next step:** Shipped in PR #15527.

### DL-#15398 · S4: Straight Angular-Momentum Arrows (Whole-Body + Groups, Fixed Scale) for MuJoCo/MeshCat

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15398
- **Branch:** merged via #15409
- **PR:** #15409
- **Paths:** see #15409
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`d26d6e2a`; collated from changes/15398-angular-momentum-vector-overlay.md)
- **Summary:** S4: straight angular-momentum arrows (whole-body + groups, fixed scale) for MuJoCo/MeshCat
- **Next step:** Shipped in PR #15409.

### DL-#15397 · Engine-Agnostic Segment Angular-Momentum Frame and MuJoCo Provider (About Whole-Body COM, World Frame)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15397
- **Branch:** merged via #15400
- **PR:** #15400
- **Paths:** see #15400
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`a439780d`; collated from changes/15397-segment-angular-momentum-provider-mujoco.md)
- **Summary:** Engine-agnostic segment angular-momentum frame and MuJoCo provider (about whole-body COM, world frame)
- **Next step:** Shipped in PR #15400.

### DL-#15396 · Segment Kinetic-Energy Surface Shading (Cividis, Fixed Limits); Demote KE Globes to KE Proxy Debug Overlay

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15396
- **Branch:** merged via #15427
- **PR:** #15427
- **Paths:** see #15427
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`510beb20`; collated from changes/15396-segment-ke-surface-shading.md)
- **Summary:** Segment kinetic-energy surface shading (cividis, fixed limits); demote KE globes to KE proxy debug overlay
- **Next step:** Shipped in PR #15427.

### DL-#15395 · Generalize Segment Tension/Compression Shading Into Quantity-Agnostic Scalar Series and Colour Scale

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15395
- **Branch:** merged via #15403
- **PR:** #15403
- **Paths:** see #15403
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`0ad23f4f`; collated from changes/15395-segment-scalar-shading-core.md)
- **Summary:** Generalize segment tension/compression shading into quantity-agnostic scalar series and colour scale
- **Next step:** Shipped in PR #15403.

### DL-#15345 · Bundle Adjustment FC-011 Guards, Feasible Tangent Projection, Bounded FD, and Positive Prior Sigmas

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15345
- **Branch:** merged via #15537
- **PR:** #15537
- **Paths:** see #15537
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`38425e26`; collated from changes/15345-bundle-adjustment-fc-011-guards-inward-f.md)
- **Summary:** Bundle adjustment FC-011 guards, feasible tangent projection, bounded FD, and positive prior sigmas
- **Next step:** Shipped in PR #15537.

### DL-#15344 · Guard Degenerate GP Radius Window in Compute_Correlation_Dimension

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15344
- **Branch:** merged via #15530
- **PR:** #15530
- **Paths:** see #15530
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`8994c0e2`; collated from changes/15344-guard-degenerate-gp-radius-window-in-com.md)
- **Summary:** Guard degenerate GP radius window in compute_correlation_dimension
- **Next step:** Shipped in PR #15530.

### DL-#15343 · Guard Hysteresis, Bound Power Durations, Defensive Work-Metrics Cache Copies

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15343
- **Branch:** merged via #15531
- **PR:** #15531
- **Paths:** see #15531
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`c981f575`; collated from changes/15343-guard-hysteresis-bound-power-durations-d.md)
- **Summary:** Guard hysteresis, bound power durations, defensive work-metrics cache copies
- **Next step:** Shipped in PR #15531.

### DL-#15342 · Require a List for Subject.Dropped_Markers in Marker_Selection; Dict, Int, Bool, Tuple and Stringified Values Now Raise ValueError

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15342
- **Branch:** merged via #15513
- **PR:** #15513
- **Paths:** see #15513
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`fa228f3a`; collated from changes/15342-require-a-list-for-subject-dropped-marke.md)
- **Summary:** Require a list for subject.dropped_markers in marker_selection; dict, int, bool, tuple and stringified values now raise ValueError
- **Next step:** Shipped in PR #15513.

### DL-#15333 · C3D Viewer Embed Loader Registers the Tool Again When the Registry Was Reset After the First Import

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15333
- **Branch:** merged via #15334
- **PR:** #15334
- **Paths:** see #15334
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`224fae7e`; collated from changes/15333-c3d-viewer-embed-loader-registers-the-to.md)
- **Summary:** C3D viewer embed loader registers the tool again when the registry was reset after the first import
- **Next step:** Shipped in PR #15334.

### DL-#15329 · A Hand May Be Given a Thumb Target of Its Own (Thumb.Toward_Index_Target_of, None Set): The Trail Thumb Can Close a V on Its Forefinger While the Lead Thumb Keeps Its Azimuth. A Hand's Thumb May Be Held Flexed, a Short Thumb (Thumb.Held_Flexion_Deg_of, None Set): Tried for the Lead Thumb Under the Squarer Trail Seats and Not Set, Because the Raised Knuckle Lifts the Trail Palm Off the Thumb (Crown Gap 12.6 to 15.3-17.9 mm); Drawn Hands Unchanged (Appearance Only)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15329
- **Branch:** merged via #15372
- **PR:** #15372, #15364, #15368, #15377, #15362
- **Paths:** see #15372
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`0f125125`; collated from changes/15329-set-metacarpal-arch-for-the-skinned-hand.md)
- **Summary:** A hand may be given a thumb target of its own (thumb.toward_index_target_of, none set): the trail thumb can close a V on its forefinger while the lead thumb keeps its azimuth. A hand's thumb may be held flexed, a short thumb (thumb.held_flexion_deg_of, none set): tried for the lead thumb under the squarer trail seats and not set, because the raised knuckle lifts the trail palm off the thumb (crown gap 12.6 to 15.3-17.9 mm); drawn hands unchanged (appearance only)
- **Next step:** Shipped in PR #15372.

### DL-#15322 · Pelvis Skin Map Scales Lateral Hip Spacing to Match Subject Hip_Half_Spacing

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15322
- **Branch:** merged via #15556
- **PR:** #15556
- **Paths:** see #15556
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`45491e1d`; collated from changes/15322-pelvis-map-hip-half-spacing.md)
- **Summary:** Pelvis skin map scales lateral hip spacing to match subject hip_half_spacing
- **Next step:** Shipped in PR #15556.

### DL-#15299 · Add a Chest Depth Scale and a Thigh-Front Shift to the CC0 Body Skin Fit as Named Settings, Neutral by Default and Set by the Showpiece Profile

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15299
- **Branch:** merged via #15325
- **PR:** #15325, #15324, #15317, #15321, #15335, #15314, #15384, #15331, #15316, #15367, #15366, #15303, #15310, #15309, #15328, #15338, #15313, #15336, #15370, #15308, #15386, #15307, #15365, #15371, #15375, #15356, #15376, #15385, #15360, #15353, #15369, #15379, #15347, #15341, #15339, #15349, #15363, #15305, #15326, #15311
- **Paths:** see #15325
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`1918c7f4`; collated from changes/15299-wrist-ball-elbow-and-one-tone-skin.md)
- **Summary:** Add a chest depth scale and a thigh-front shift to the CC0 body skin fit as named settings, neutral by default and set by the showpiece profile
- **Next step:** Shipped in PR #15325.

### DL-#15296 · Add a Selectable Club-Only Driver Club Marker Registration (Club_Only); the Default Driver.Json and the Committed Documents Are Unchanged

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15296
- **Branch:** merged via #15302
- **PR:** #15302, #15374, #15304, #15320, #15361, #15351, #15340, #15346, #15337, #15332, #15354, #15348, #15318, #15330, #15315, #15358, #15312
- **Paths:** see #15302
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`7684bb42`; collated from changes/15296-thigh-shank-and-hip-spacing-as-optional.md)
- **Summary:** Add a selectable club-only driver club marker registration (club_only); the default driver.json and the committed documents are unchanged
- **Next step:** Shipped in PR #15302.

### DL-#15292 · Slanted Crew-Neck Collar and Shoulder Caps Lowered Below the Neck Point (Trunk, Yoke and Upper-Arm Geometry)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15292
- **Branch:** merged via #15298
- **PR:** #15298, #15300, #15294
- **Paths:** see #15298
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`74469014`; collated from changes/15292-visual-neck-solid-for-the-cc0-head-thora.md)
- **Summary:** Slanted crew-neck collar and shoulder caps lowered below the neck point (trunk, yoke and upper-arm geometry)
- **Next step:** Shipped in PR #15298.

### DL-#15280 · Hide Contact Spheres by Default, Light Golf Shoes With Dark Sole, Tapered Legs

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15280
- **Branch:** merged via #15281
- **PR:** #15281
- **Paths:** see #15281
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`2b02c618`; collated from changes/15280-hide-contact-spheres-by-default-light-go.md)
- **Summary:** Hide contact spheres by default, light golf shoes with dark sole, tapered legs
- **Next step:** Shipped in PR #15281.

### DL-#15279 · Shoulder Caps Sit Lower and Smaller so the Shoulders Read as Shoulders, Not Pads

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15279
- **Branch:** merged via #15290
- **PR:** #15290, #15282, #15283
- **Paths:** see #15290
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`5033cda6`; collated from changes/15279-upper-body-builder-can-take-the-clavicle.md)
- **Summary:** Shoulder caps sit lower and smaller so the shoulders read as shoulders, not pads
- **Next step:** Shipped in PR #15290.

### DL-#15278 · Knuckle Sideways Range as a Setting (Off by Default): A Bent Knuckle's Sideways Swing Is Held to a Range That Shrinks With Flexion

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15278
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`f637f68b`; collated from changes/15278-trail-thumb-set-40-degrees-to-the-target.md)
- **Summary:** Knuckle sideways range as a setting (off by default): a bent knuckle's sideways swing is held to a range that shrinks with flexion
- **Next step:** Shipped in PR #15288.

### DL-#15277 · Arm Refit Takes Posture Targets; Address Posture Is a Neutral Lumbar Joint With a Slight Thoracic Bend

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15277
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`f637f68b`; collated from changes/15277-arm-refit-takes-posture-targets-address.md)
- **Summary:** Arm refit takes posture targets; address posture is a neutral lumbar joint with a slight thoracic bend
- **Next step:** Shipped in PR #15288.

### DL-#15276 · Golfer Head: Dirty Blonde Short-Crop Hair With a Natural Taper, Eyes, Brows and Mouth

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15276
- **Branch:** merged via #15287
- **PR:** #15287
- **Paths:** see #15287
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`117df07e`; collated from changes/15276-golfer-head-dirty-blonde-short-crop-hair.md)
- **Summary:** Golfer head: dirty blonde short-crop hair with a natural taper, eyes, brows and mouth
- **Next step:** Shipped in PR #15287.

### DL-#15275 · Add Per-Frame Swing Wrist and Forearm Readout (Flexion/Extension, Radial/Ulnar Deviation, Pronation/Supination) and 3-Row Plot Script

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15275
- **Branch:** merged via #15284
- **PR:** #15284
- **Paths:** see #15284
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`4237aa18`; collated from changes/15275-add-per-frame-swing-wrist-and-forearm-re.md)
- **Summary:** Add per-frame swing wrist and forearm readout (flexion/extension, radial/ulnar deviation, pronation/supination) and 3-row plot script
- **Next step:** Shipped in PR #15284.

### DL-#15272 · Skinned Hand: Typed Landmarks and Joint Frames Export for the Other Hand Tracks

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15272
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`f637f68b`; collated from changes/15272-skinned-hand-typed-landmarks-and-joint-f.md)
- **Summary:** Skinned hand: typed landmarks and joint frames export for the other hand tracks
- **Next step:** Shipped in PR #15288.

### DL-#15270 · Skinned Hand: Palm Fold Joint so the Trail Palm Closes on the Lead Thumb

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15270
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`f637f68b`; collated from changes/15270-palm-fold-joint-so-the-trail-palm-closes.md)
- **Summary:** Skinned hand: palm fold joint so the trail palm closes on the lead thumb
- **Next step:** Shipped in PR #15288.

### DL-#15265 · Rest the Trail Palm on the Lead Thumb: Owner Trail Cross Angle, Thumb Contact Check

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15265
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`f637f68b`; collated from changes/15265-rest-the-trail-palm-on-the-lead-thumb-ow.md)
- **Summary:** Rest the trail palm on the lead thumb: owner trail cross angle, thumb contact check
- **Next step:** Shipped in PR #15288.

### DL-#15264 · Size the Forearm's Wrist End From the Skinned Hand's Wrist When the Skinned Hands Are Drawn

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15264
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`f637f68b`; collated from changes/15264-size-the-forearm-s-wrist-end-from-the-sk.md)
- **Summary:** Size the forearm's wrist end from the skinned hand's wrist when the skinned hands are drawn
- **Next step:** Shipped in PR #15288.

### DL-#15262 · Draw the Forearm as One Continuous Tapered Shape Across Its Rotation Joint, so the Golfer No Longer Appears to Have Two Elbows

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15262
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`f637f68b`; collated from changes/15262-draw-the-forearm-as-one-continuous-taper.md)
- **Summary:** Draw the forearm as one continuous tapered shape across its rotation joint, so the golfer no longer appears to have two elbows
- **Next step:** Shipped in PR #15288.

### DL-#15258 · Arm Scale Estimated From Capture Marker Distances With Stated Skin-to-Joint Offsets

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15258
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`f637f68b`; collated from changes/15258-arm-scale-estimated-from-capture-marker.md)
- **Summary:** Arm scale estimated from capture marker distances with stated skin-to-joint offsets
- **Next step:** Shipped in PR #15288.

### DL-#15255 · Arm Refit That Returns the Club to Its Pose After the Wrist Joints Move to the Skinned Hands, With Optional Trunk Coordinates

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15255
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`f637f68b`; collated from changes/15255-arm-refit-that-returns-the-club-to-its-p.md)
- **Summary:** Arm refit that returns the club to its pose after the wrist joints move to the skinned hands, with optional trunk coordinates
- **Next step:** Shipped in PR #15288.

### DL-#15253 · Spec Revision That Moves the Golfer's Wrist Joints to the Skinned Hands' Wrists; Skinned Club Hands Cut and Capped at the Wrist

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15253
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`f637f68b`; collated from changes/15253-spec-revision-that-moves-the-golfer-s-wr.md)
- **Summary:** Spec revision that moves the golfer's wrist joints to the skinned hands' wrists; skinned club hands cut and capped at the wrist
- **Next step:** Shipped in PR #15288.

### DL-#15251 · Skinned Human Hands Selectable on the Golfer's Club, Squared to the Clubface (Subject.Hand_Visual_Backend)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15251
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`f637f68b`; collated from changes/15251-skinned-human-hands-selectable-on-the-go.md)
- **Summary:** Skinned human hands selectable on the golfer's club, squared to the clubface (subject.hand_visual_backend)
- **Next step:** Shipped in PR #15288.

### DL-#15248 · Remove Unused Numpy Import From Motion_Pipeline Resample Preprocessing (Ruff F401)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15248
- **Branch:** barb/remove-unused-numpy-import
- **PR:** #15248
- **Paths:** see #15248
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`53dd87f7`; collated from changes/15248-remove-unused-numpy-import-from-motion-p.md)
- **Summary:** Remove unused numpy import from motion_pipeline resample preprocessing (ruff F401)
- **Next step:** Shipped in PR #15248.

### DL-#15246 · Conflict Disarm Starts Again: Pass Only the Secrets the Pinned Provider Declares

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15246
- **Branch:** merged via #15247
- **PR:** #15247
- **Paths:** see #15247
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`caeb99c2`; collated from changes/15246-conflict-disarm-starts-again-pass-only-t.md)
- **Summary:** Conflict Disarm starts again: pass only the secrets the pinned provider declares
- **Next step:** Shipped in PR #15247.

### DL-#15241 · Preserve JSON Recording Files on Serialization or Publication Failure by Reusing the Atomic Text Writer.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15241
- **Branch:** merged via #15243
- **PR:** #15243
- **Paths:** see #15243
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`93797bb7`; collated from changes/15241-preserve-json-recording-files-on-seriali.md)
- **Summary:** Preserve JSON recording files on serialization or publication failure by reusing the atomic text writer.
- **Next step:** Shipped in PR #15243.

### DL-#15240 · Honor Package Anchors in Safe_Import Relative Module Resolution

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15240
- **Branch:** merged via #15242
- **PR:** #15242
- **Paths:** see #15242
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`a401215a`; collated from changes/15240-safe-relative-imports.md)
- **Summary:** Honor package anchors in safe_import relative module resolution
- **Next step:** Shipped in PR #15242.

### DL-#15236 · Reject Unsupported Analysis Report Formats Before Creating Directories or Capturing Provenance, Preserving JSON and HTML Exports.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15236
- **Branch:** merged via #15237
- **PR:** #15237
- **Paths:** see #15237
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`c38937eb`; collated from changes/15236-reject-unsupported-analysis-report-forma.md)
- **Summary:** Reject unsupported analysis report formats before creating directories or capturing provenance, preserving JSON and HTML exports.
- **Next step:** Shipped in PR #15237.

### DL-#15227 · RQA Determinism Counts Diagonal Lines in Both Triangles and Excludes the Main Diagonal by Its Content

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15227
- **Branch:** merged via #15230
- **PR:** #15230
- **Paths:** see #15230
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`55dbf8d1`; collated from changes/15227-rqa-determinism-counts-diagonal-lines-in.md)
- **Summary:** RQA determinism counts diagonal lines in both triangles and excludes the main diagonal by its content
- **Next step:** Shipped in PR #15230.

### DL-#15226 · Compare_Scalars No Longer Reports 0% for Opposite-Sign or NaN Values

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15226
- **Branch:** merged via #15229
- **PR:** #15229
- **Paths:** see #15229
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`af7c1593`; collated from changes/15226-compare-scalars-no-longer-reports-0-for.md)
- **Summary:** compare_scalars no longer reports 0% for opposite-sign or NaN values
- **Next step:** Shipped in PR #15229.

### DL-#15225 · Pair Tools Cache Count-Pressure Correction While Preserving Integrity and Minimum-Age Checks

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15225
- **Branch:** merged via #15228
- **PR:** #15228
- **Paths:** see #15228
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`4a46a5ed`; collated from changes/15225-pair-tools-cache-count-pressure-correcti.md)
- **Summary:** Pair Tools cache count-pressure correction while preserving integrity and minimum-age checks
- **Next step:** Shipped in PR #15228.

### DL-#15223 · SwingComparator No Longer Scores 100 When the Reference Peak Speed, Club Head Speed or Tempo Ratio Is Zero.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15223
- **Branch:** merged via #15224
- **PR:** #15224
- **Paths:** see #15224
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`b1d00398`; collated from changes/15223-swing-comparison-zero-reference-score.md)
- **Summary:** SwingComparator no longer scores 100 when the reference peak speed, club head speed or tempo ratio is zero.
- **Next step:** Shipped in PR #15224.

### DL-#15214 · Adopt Tools Cumulative Integral Initial Offset via Coherent Provider Pin and Consumer Regressions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15214
- **Branch:** merged via #15575
- **PR:** #15575
- **Paths:** see #15575
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`54b5f908`; collated from changes/15214-adopt-paired-cumulative-integral-offset.md)
- **Summary:** Adopt Tools cumulative integral initial offset via coherent provider pin and consumer regressions
- **Next step:** Shipped in PR #15575.

### DL-#15213 · Compute_Screw_Axis Refuses Non-Finite Twists, Bad Thresholds and Bad Endpoint Lengths Instead of Returning NaN Screw Axes.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15213
- **Branch:** merged via #15215
- **PR:** #15215
- **Paths:** see #15215
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`b208d7aa`; collated from changes/15213-screw-axis-refuses-non-finite-twists.md)
- **Summary:** compute_screw_axis refuses non-finite twists, bad thresholds and bad endpoint lengths instead of returning NaN screw axes.
- **Next step:** Shipped in PR #15215.

### DL-#15209 · Reject NaN Environment Floats When Numeric Bounds Are Supplied; Pair Tools#6083

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15209
- **Branch:** merged via #15211
- **PR:** #15211
- **Paths:** see #15211
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`3e0db00b`; collated from changes/15209-reject-nan-environment-floats-when-numer.md)
- **Summary:** Reject NaN environment floats when numeric bounds are supplied; pair Tools#6083
- **Next step:** Shipped in PR #15211.

### DL-#15206 · Spectral Arc Length Now Uses the Adaptive Cut-Off From Balasubramanian 2015, so It No Longer Depends on Fc

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15206
- **Branch:** merged via #15210
- **PR:** #15210
- **Paths:** see #15210
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`2de7b64d`; collated from changes/15206-spectral-arc-length-adaptive-cutoff.md)
- **Summary:** Spectral arc length now uses the adaptive cut-off from Balasubramanian 2015, so it no longer depends on fc
- **Next step:** Shipped in PR #15210.

### DL-#15205 · Capture an Independent Deep Snapshot of Provenance Parameters so Caller Mutations Cannot Rewrite Run Metadata or Later Export Headers.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15205
- **Branch:** merged via #15235
- **PR:** #15235
- **Paths:** see #15235
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`09b84de8`; collated from changes/15205-capture-an-independent-deep-snapshot-of.md)
- **Summary:** Capture an independent deep snapshot of provenance parameters so caller mutations cannot rewrite run metadata or later export headers.
- **Next step:** Shipped in PR #15235.

### DL-#15204 · Use Transverse Spin in the Python Magnus/Lift Force Laws (BallFlightSimulator, LiftModel/MagnusModel, ODE Flight Models) as #15042 Did for Rust.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15204
- **Branch:** merged via #15208
- **PR:** #15208
- **Paths:** see #15208
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`8a715591`; collated from changes/15204-use-transverse-spin-in-python-magnus-lift.md)
- **Summary:** Use transverse spin in the Python Magnus/lift force laws (BallFlightSimulator, LiftModel/MagnusModel, ODE flight models) as #15042 did for Rust.
- **Next step:** Shipped in PR #15208.

### DL-#15202 · Reject Zero-Row CSV Frames and Trajectory Arrays Before Reporting Successful Loads

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15202
- **Branch:** merged via #15207
- **PR:** #15207
- **Paths:** see #15207
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`a565e63f`; collated from changes/15202-reject-zero-row-csv-frames-and-trajector.md)
- **Summary:** Reject zero-row CSV frames and trajectory arrays before reporting successful loads
- **Next step:** Shipped in PR #15207.

### DL-#15200 · Preserve Exact NumPy Integer Values and JSON Integer Types in Output Exports

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15200
- **Branch:** merged via #15203
- **PR:** #15203
- **Paths:** see #15203
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`c2c88b51`; collated from changes/15200-preserve-exact-numpy-integer-values-and.md)
- **Summary:** Preserve exact NumPy integer values and JSON integer types in output exports
- **Next step:** Shipped in PR #15203.

### DL-#15199 · Stage Legacy Launcher Configuration Copies so Partial Failures Remain Retryable

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15199
- **Branch:** merged via #15201
- **PR:** #15201
- **Paths:** see #15201
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`a75eb07c`; collated from changes/15199-stage-legacy-launcher-configuration-copi.md)
- **Summary:** Stage legacy launcher configuration copies so partial failures remain retryable
- **Next step:** Shipped in PR #15201.

### DL-#15150 · Add a Dispersion-on-the-Hole Still (--Hole, --Hole-Preset) to the Shot Dispersion CLI

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15150
- **Branch:** merged via #15507
- **PR:** #15507, #15441, #15192, #15476, #15423, #15472, #15458, #15674, #15394, #15479, #15381, #15681, #15510, #15417, #15432, #15413, #15407, #15465, #15515, #15481, #15478, #15487, #15428, #15437, #15374, #15449, #15457, #15456, #15436, #15412, #15415, #15434, #15464, #15422, #15533, #15520, #15528, #15526, #15519, #15532, #15470, #15408, #15387, #15455, #15411, #15402, #15433, #15421, #15420, #15426, #15410, #15424, #15469, #15484, #15453, #15401, #15483, #15509, #15418, #15517, #15467, #15461, #15504, #15438, #15414, #15406, #15405, #15430, #15440, #15508, #15516, #15642, #15549, #15540, #15566, #15547, #15557, #15529, #15553, #15536, #15480, #15511, #15514, #15477, #15378, #15425, #15380, #15675, #15463, #15679, #15197, #15459, #15451, #15419, #15442, #15382, #15388, #15404, #15301, #15505, #15416, #15350, #15355, #15359, #15503
- **Paths:** see #15507
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`ab4cdc8c`; collated from changes/15150-with-shaft-model-resolves-a-named-club-t.md)
- **Summary:** Add a dispersion-on-the-hole still (--hole, --hole-preset) to the shot dispersion CLI
- **Next step:** Shipped in PR #15507.

### DL-#15130 · Local Desktop API App Requires the Launcher Capability Token on API Requests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15130
- **Branch:** merged via #15132
- **PR:** #15132
- **Paths:** see #15132
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`4a1c5f43`; collated from changes/15130-local-app-requires-the-launcher-capability.md)
- **Summary:** Local desktop API app requires the launcher capability token on API requests
- **Next step:** Shipped in PR #15132.

### DL-#15122 · Fleet Critic 2026-11-20: Grip Epic Critique (Builder Capsule Clearance Uses Max Not Min; Grip-Loss Shear Estimate Drops Axial Pull; Near-Circular Null Result).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15122
- **Branch:** merged via #15352
- **PR:** #15352, #15357
- **Paths:** see #15352
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`cc80419a`; collated from changes/15122-pragmatic-programmer-review-2026-11-20-o.md)
- **Summary:** Fleet Critic 2026-11-20: grip epic critique (builder capsule clearance uses max not min; grip-loss shear estimate drops axial pull; near-circular null result).
- **Next step:** Shipped in PR #15352.

### DL-#15119 · Record the Reviewed GAAI Policy-Test Retirement for Merge-Group CI

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15119
- **Branch:** merged via #15119
- **PR:** #15119
- **Paths:** see #15119
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`e63b3a62`; collated from changes/15119-reviewed-test-deletion.md)
- **Summary:** Record the reviewed GAAI policy-test retirement for merge-group CI
- **Next step:** Shipped in PR #15119.

### DL-#15117 · Desktop UI Sends the Launcher Capability Token on Backend API Requests Through One Shared Fetch Helper

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15117
- **Branch:** merged via #15118
- **PR:** #15118
- **Paths:** see #15118
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`800ac553`; collated from changes/15117-desktop-ui-sends-the-launcher-capability.md)
- **Summary:** Desktop UI sends the launcher capability token on backend API requests through one shared fetch helper
- **Next step:** Shipped in PR #15118.

### DL-#15108 · Scope Tests/Unit/conftest.py Engine MagicMocks to Tests/Unit and Guard OpenSim Dynamics Parity (#15108)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15108
- **Branch:** merged via #15109
- **PR:** #15109
- **Paths:** see #15109
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`daf6ce2a`; collated from changes/15108-scope-unit-conftest-engine-mocks.md)
- **Summary:** Scope tests/unit/conftest.py engine MagicMocks to tests/unit and guard OpenSim dynamics parity (#15108)
- **Next step:** Shipped in PR #15109.

### DL-#15106 · Prune Dead Testpaths/LEGACY_SRC_TEST_DIRS, Enable Green Tests/* Suites in Unit-Test-Gate, and Enforce Unrun Suite Exclusions (#15106)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15106
- **Branch:** merged via #15110
- **PR:** #15110
- **Paths:** see #15110
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`f25a6b67`; collated from changes/15106-testpaths-and-unrun-suite-exclusions.md)
- **Summary:** Prune dead testpaths/LEGACY_SRC_TEST_DIRS, enable green tests/* suites in unit-test-gate, and enforce unrun suite exclusions (#15106)
- **Next step:** Shipped in PR #15110.

### DL-#15105 · Guard Drain_Process_Streams Against Infinite MagicMock Readline Loops in Launcher Pipe Drains

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15105
- **Branch:** merged via #15157
- **PR:** #15157
- **Paths:** see #15157
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`e2cecf0c`; collated from changes/15105-drain-magicmock-guard.md)
- **Summary:** Guard drain_process_streams against infinite MagicMock readline loops in launcher pipe drains
- **Next step:** Shipped in PR #15157.

### DL-#15103 · Guard Pytest Makereport and Teardown Against Leaked Os.Name Patches

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15103
- **Branch:** merged via #15168
- **PR:** #15168
- **Paths:** see #15168
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`b8159ea3`; collated from changes/15103-guard-pytest-makereport-and-teardown-aga.md)
- **Summary:** Guard pytest makereport and teardown against leaked os.name patches
- **Next step:** Shipped in PR #15168.

### DL-#15101 · Join or Retain Retiring BinderThread and CameraWorker Instances Until Qt Finishes Thread Teardown so Capture Rig Preview Never Aborts in ~QThread.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15101
- **Branch:** merged via #15102
- **PR:** #15102
- **Paths:** see #15102
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`8123908e`; collated from changes/15101-preview-qthread-lifetime.md)
- **Summary:** Join or retain retiring BinderThread and CameraWorker instances until Qt finishes thread teardown so Capture Rig preview never aborts in ~QThread.
- **Next step:** Shipped in PR #15102.

### DL-#15097 · Isolate Sidekick Sys.Modules and Sys.Meta_Path State Across Overlay Tests (#15097)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15097
- **Branch:** merged via #15099
- **PR:** #15099
- **Paths:** see #15099
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`eb2d9b7e`; collated from changes/15097-sidekick-overlay-module-isolation.md)
- **Summary:** Isolate sidekick sys.modules and sys.meta_path state across overlay tests (#15097)
- **Next step:** Shipped in PR #15099.

### DL-#15091 · Drain Child Output Streams Before Transitioning ManagedProcess Exit State and Exposing Exit_Code

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15091
- **Branch:** merged via #15092
- **PR:** #15092
- **Paths:** see #15092
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`257654e0`; collated from changes/15091-stderr-drain-before-exit.md)
- **Summary:** Drain child output streams before transitioning ManagedProcess exit state and exposing exit_code
- **Next step:** Shipped in PR #15092.

### DL-#15013 · Restore UdAttachmentManifestMixin so Rejected File_Upload Clears Dedup SHA and Retry Sends WebSocket Again; Re-Add Reject-Then-Retry Regression Tests.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15013
- **Branch:** merged via #15072
- **PR:** #15072, #15060, #15156
- **Paths:** see #15072
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`d7bdbc56`; collated from changes/15013-sg-analytics-documentation-close.md)
- **Summary:** Restore UdAttachmentManifestMixin so rejected file_upload clears dedup SHA and retry sends WebSocket again; re-add reject-then-retry regression tests.
- **Next step:** Shipped in PR #15072.

### DL-#14850 · Redact Host/Home Literals in Batch 5 Docs and Add Shell Placeholder Regression Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14850
- **Branch:** merged via #14850
- **PR:** #14850
- **Paths:** see #14850
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`9a0dbab7`; collated from changes/14850-redact-host-home-literals-in-batch-5-doc.md)
- **Summary:** Redact host/home literals in batch 5 docs and add shell placeholder regression test
- **Next step:** Shipped in PR #14850.

### DL-#14849 · Parameterize Remote Evidence Runners and Finish Batch 3 Host Redaction

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14849
- **Branch:** merged via #14849
- **PR:** #14849
- **Paths:** see #14849
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`c60c445c`; collated from changes/14849-parameterize-remote-evidence-runners-and.md)
- **Summary:** Parameterize remote evidence runners and finish batch 3 host redaction
- **Next step:** Shipped in PR #14849.

### DL-#14310 · Headless Qt/Matplotlib Teardown After Each Test and MuJoCo Offscreen Renderer Release (#14310)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14310
- **Branch:** merged via #15098
- **PR:** #15098
- **Paths:** see #15098
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`180394fd`; collated from changes/14310-headless-gui-teardown.md)
- **Summary:** Headless Qt/matplotlib teardown after each test and MuJoCo offscreen renderer release (#14310)
- **Next step:** Shipped in PR #15098.

### DL-#13575 · Retire Jules Push/PR Noise Workflows and Use FLEET_GENERAL_AGENT App Credentials for Fleet Policy Gates.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13575
- **Branch:** merged via #14999
- **PR:** #14999
- **Paths:** see #14999
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`310c1327`; collated from changes/13575-retire-jules-workflows.md)
- **Summary:** Retire Jules push/PR noise workflows and use FLEET_GENERAL_AGENT App credentials for fleet policy gates.
- **Next step:** Shipped in PR #14999.

### DL-#13314 · Unit-Test Gate Runs in Three Shards; Leg Two of Two Was Cancelled at Its 35 Minute Limit

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13314
- **Branch:** merged via #15319
- **PR:** #15319
- **Paths:** see #15319
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`f7827f0a`; collated from changes/13314-unit-test-gate-runs-in-three-shards-leg.md)
- **Summary:** Unit-test gate runs in three shards; leg two of two was cancelled at its 35 minute limit
- **Next step:** Shipped in PR #15319.

### DL-#12678 · Preserve Smooth CC0 Hair Shading in Shared Native-Viewer OBJ Exports, Matching MuJoCo's Hair Selection

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12678
- **Branch:** merged via #15571
- **PR:** #15571
- **Paths:** see #15571
- **Started:** 2026-11-24
- **Last verified:** 2026-11-24 (`af612ed8`; collated from changes/12678-native-hair-normals.md)
- **Summary:** Preserve smooth CC0 hair shading in shared native-viewer OBJ exports, matching MuJoCo's hair selection
- **Next step:** Shipped in PR #15571.

### DL-#15095 · Fix SwingMetricsMixin Test Harness After Velocity Delegation and Add Unit Marker (#15095)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15095
- **Branch:** merged via #15096
- **PR:** #15096
- **Paths:** see #15096
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`c0d0f6c9`; collated from changes/15095-dbc-swing-phase-unit-marker.md)
- **Summary:** Fix SwingMetricsMixin test harness after velocity delegation and add unit marker (#15095)
- **Next step:** Shipped in PR #15096.

### DL-#15093 · Guard NP-05 Lexical Test Against Collated Changelog Fragment Removal (#15093)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15093
- **Branch:** merged via #15094
- **PR:** #15094
- **Paths:** see #15094
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`a011577c`; collated from changes/15093-np05-collated-changes-guard.md)
- **Summary:** Guard NP-05 lexical test against collated changelog fragment removal (#15093)
- **Next step:** Shipped in PR #15094.

### DL-#15046 · SCI-10/12/13/14: Shaft FEM Scope Docs, Pendulum Inertia Param, Static-Opt Cost, Spatial Ordering

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15046
- **Branch:** merged via #15052
- **PR:** #15052
- **Paths:** see #15052
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`e25c7b5b`; collated from changes/15046-sci-10-12-13-14-shaft-fem-scope-docs-pen.md)
- **Summary:** SCI-10/12/13/14: shaft FEM scope docs, pendulum inertia param, static-opt cost, spatial ordering
- **Next step:** Shipped in PR #15052.

### DL-#15042 · Use Transverse Spin in Magnus/Lift Magnitude; Document Frame and Parity Fixtures (#15042).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15042
- **Branch:** merged via #15053
- **PR:** #15053
- **Paths:** see #15053
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`9c8f8834`; collated from changes/15042-use-transverse-spin-in-magnus-lift-magni.md)
- **Summary:** Use transverse spin in Magnus/lift magnitude; document frame and parity fixtures (#15042).
- **Next step:** Shipped in PR #15053.

### DL-#15041 · Evaluate Constrained Per-Actuator Columns at Measured (Q, V) With ZTCF Baseline

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15041
- **Branch:** merged via #15049
- **PR:** #15049
- **Paths:** see #15049
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`c6acff9d`; collated from changes/15041-evaluate-constrained-per-actuator-column.md)
- **Summary:** Evaluate constrained per-actuator columns at measured (q, v) with ZTCF baseline
- **Next step:** Shipped in PR #15049.

### DL-#15030 · Quantize Spatial Model Canonical Hashes for Cross-Runtime Manufactured Authority Parity, Regenerate Publication Evidence, and Re-Pick Optional-Stack Shooting Defect Floor After #14227 Gravity Alignment.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15030
- **Branch:** merged via #15069
- **PR:** #15069
- **Paths:** see #15069
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`3e7c72e8`; collated from changes/15030-quantize-spatial-model-canonical-hashes.md)
- **Summary:** Quantize spatial model canonical hashes for cross-runtime manufactured authority parity, regenerate publication evidence, and re-pick optional-stack shooting defect floor after #14227 gravity alignment.
- **Next step:** Shipped in PR #15069.

### DL-#15028 · Wave 0.1: Independent Triage of 67 Budget Exceptions That Expired 2026-11-30 (Measured Sizes, Regression Tests).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15028
- **Branch:** merged via #15074
- **PR:** #15074
- **Paths:** see #15074
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`175e7bc6`; collated from changes/15028-wave0-architecture-budget-triage.md)
- **Summary:** Wave 0.1: independent triage of 67 budget exceptions that expired 2026-11-30 (measured sizes, regression tests).
- **Next step:** Shipped in PR #15074.

### DL-#15021 · Expose ZTCF and Golf-Simulator Golden Capabilities via Sim_Run With REST/MCP Parity and 64 KiB Artifact Spill.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15021
- **Branch:** merged via #15061
- **PR:** #15061
- **Paths:** see #15061
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`b1bc35ea`; collated from changes/15021-expose-ztcf-and-golf-simulator-golden-ca.md)
- **Summary:** Expose ZTCF and golf-simulator golden capabilities via sim_run with REST/MCP parity and 64 KiB artifact spill.
- **Next step:** Shipped in PR #15061.

### DL-#15016 · Link Tile Help to AffineDrift Theory; Register Six Provisional Core Calculations and Dedupe Registry Blocker Ids

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15016
- **Branch:** merged via #15063
- **PR:** #15063, #15080, #15079, #15090
- **Paths:** see #15063
- **Started:** 2026-11-14
- **Last verified:** 2026-11-24 (`d54c0c57`; collated from changes/15016-spec-input-map-manifest-honesty.md)
- **Summary:** Link tile help to AffineDrift theory; register six provisional core calculations and dedupe registry blocker ids
- **Next step:** Shipped in PR #15063.

### DL-#15012 · Fix WBC Zero Torques With Contact and Hierarchical Task Enforcement

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15012
- **Branch:** merged via #15025
- **PR:** #15025, #15083
- **Paths:** see #15025
- **Started:** 2026-11-14
- **Last verified:** 2026-11-24 (`339be5a5`; collated from changes/15012-friction-cone-rl-honesty.md)
- **Summary:** Fix WBC zero torques with contact and hierarchical task enforcement
- **Next step:** Shipped in PR #15025.

### DL-#15010 · Correct Putting Slope Rolling (5/7 G), Unify Stimp Friction, Impact Club Spin, Flight Spin Sign Docs, Rust Air Temperature K

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15010
- **Branch:** merged via #15026
- **PR:** #15026, #15084
- **Paths:** see #15026
- **Started:** 2026-11-14
- **Last verified:** 2026-11-24 (`6996475c`; collated from changes/15010-flight-aero-impact-honesty.md)
- **Summary:** Correct putting slope rolling (5/7 g), unify stimp friction, impact club spin, flight spin sign docs, Rust air temperature K
- **Next step:** Shipped in PR #15026.

### DL-#14949 · Managed Local Child-Process Lifecycle; Concurrent Stdout/Stderr Drain Avoids Pipe Deadlock (Refs #14949).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14949
- **Branch:** merged via #14956
- **PR:** #14956, #15081
- **Paths:** see #14956
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`450becd6`; collated from changes/14949-wsl-matlab-docker-lifecycle.md)
- **Summary:** Managed local child-process lifecycle; concurrent stdout/stderr drain avoids pipe deadlock (Refs #14949).
- **Next step:** Shipped in PR #14956.

### DL-#14942 · EngineManager Reloads LOADED Engines; Cleanup Resets Status and Closes Active Engine (#14942).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14942
- **Branch:** merged via #15017
- **PR:** #15017, #14961
- **Paths:** see #15017
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`38c09978`; collated from changes/14942-enginemanager-reloads-loaded-engines-cle.md)
- **Summary:** EngineManager reloads LOADED engines; cleanup resets status and closes active engine (#14942).
- **Next step:** Shipped in PR #15017.

### DL-#14936 · JaxSim Canonical-V2 Round-Trip; Rename Velocity Constant to SPATIAL_WRENCH_REPRESENTATION

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14936
- **Branch:** merged via #15038
- **PR:** #15038, #14966
- **Paths:** see #15038
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`1f3202e7`; collated from changes/14936-wire-canonical-v2-get-set-on-basephysics.md)
- **Summary:** JaxSim canonical-v2 round-trip; rename velocity constant to SPATIAL_WRENCH_REPRESENTATION
- **Next step:** Shipped in PR #15038.

### DL-#14935 · Committed Evidence Scan Calls Receipt Taxonomy Validation; Missing Fields and Conflicts Are Tested (#14935).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14935
- **Branch:** merged via #14960
- **PR:** #14960, #15120, #15082
- **Paths:** see #14960
- **Started:** 2026-11-14
- **Last verified:** 2026-11-24 (`318242c0`; collated from changes/14935-evidence-recompute-and-ratchets.md)
- **Summary:** Committed evidence scan calls receipt taxonomy validation; missing fields and conflicts are tested (#14935).
- **Next step:** Shipped in PR #14960.

### DL-#14502 · NP-09 Policy Input-Output Sensitivity at Stated Operating Points (#14502)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14502
- **Branch:** merged via #14825
- **PR:** #14825, #15100
- **Paths:** see #14825
- **Started:** 2026-11-14
- **Last verified:** 2026-11-24 (`30f937bf`; collated from changes/14502-np09-lexical-fragment-guard.md)
- **Summary:** NP-09 policy input-output sensitivity at stated operating points (#14502)
- **Next step:** Shipped in PR #14825.

### DL-#14427 · Event-Relative Time Normalisation (101-Point Resampling) Between Named Swing Events

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14427
- **Branch:** merged via #14834
- **PR:** #14834
- **Paths:** see #14834
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`e64ae5e7`; collated from changes/14427-gm4-event-time-normalisation.md)
- **Summary:** Event-relative time normalisation (101-point resampling) between named swing events
- **Next step:** Shipped in PR #14834.

### DL-#14426 · GM-3 Cheetham X-Factor Stretch and Kinematic Sequence (#14426)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14426
- **Branch:** merged via #14836
- **PR:** #14836
- **Paths:** see #14836
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`8722411f`; collated from changes/14426-gm3-xfactor-kinematic-seq.md)
- **Summary:** GM-3 Cheetham X-factor stretch and kinematic sequence (#14426)
- **Next step:** Shipped in PR #14836.

### DL-#14425 · GRF and COP Observability, Event-Anchored Peaks, and Normalisation Contracts (GM-2).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14425
- **Branch:** merged via #14837
- **PR:** #14837
- **Paths:** see #14837
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`120e4803`; collated from changes/14425-gm2-grf-cop-observability.md)
- **Summary:** GRF and COP observability, event-anchored peaks, and normalisation contracts (GM-2).
- **Next step:** Shipped in PR #14837.

### DL-#12727 · NSX-98: Document A(t)/B(t) Assembly in Derivative Conformance Harness

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12727
- **Branch:** merged via #15066
- **PR:** #15066
- **Paths:** see #15066
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`f7edaa35`; collated from changes/12727-nsx-98-document-a-t-b-t-assembly-in-deri.md)
- **Summary:** NSX-98: document A(t)/B(t) assembly in derivative conformance harness
- **Next step:** Shipped in PR #15066.

### DL-#12046 · Interim 30-Day Realtime Shadow Ledger Sunset (#12046) so Hygiene Tests Pass Pending Tools Migration

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12046
- **Branch:** merged via #14929
- **PR:** #14929, #15085
- **Paths:** see #14929
- **Started:** 2026-11-14
- **Last verified:** 2026-11-24 (`e47a24aa`; collated from changes/12046-realtime-adr-split-ruling.md)
- **Summary:** Interim 30-day realtime shadow ledger sunset (#12046) so hygiene tests pass pending Tools migration
- **Next step:** Shipped in PR #14929.

### DL-#15047 · README Onboarding: Venv-First Install, Toolchains, Ud-Sim, Engines, Layout Tree.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15047
- **Branch:** cursor/readme-onboarding-15047-1324
- **PR:** #15057
- **Paths:** see #15057
- **Started:** 2026-11-12
- **Last verified:** 2026-11-12 (`95058d94`; collated from changes/15047-readme-onboarding-venv-first-install-too.md)
- **Summary:** README onboarding: venv-first install, toolchains, ud-sim, engines, layout tree.
- **Next step:** Shipped in PR #15057.

### DL-#15029 · Renew or Delete 67 Budget Exceptions Expiring 2026-11-30 (Wave 0 #15028).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15029
- **Branch:** merged via #15031
- **PR:** #15031, #15162, #15174
- **Paths:** see #15031
- **Started:** 2026-11-12
- **Last verified:** 2026-11-24 (`316a2989`; collated from changes/15029-wave0-architecture-issue-field.md)
- **Summary:** Renew or delete 67 budget exceptions expiring 2026-11-30 (Wave 0 #15028).
- **Next step:** Shipped in PR #15031.

### DL-#15009 · Fix OpenSim IAA Gravity Sign and Moment-Arm Mapping; MuJoCo Constraint-Consistent Control Acceleration; Minor Docs/Dead Code

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15009
- **Branch:** merged via #15024
- **PR:** #15024
- **Paths:** see #15024
- **Started:** 2026-11-12
- **Last verified:** 2026-11-12 (`625a16e6`; collated from changes/15009-fix-opensim-iaa-gravity-sign-and-moment.md)
- **Summary:** Fix OpenSim IAA gravity sign and moment-arm mapping; MuJoCo constraint-consistent control acceleration; minor docs/dead code
- **Next step:** Shipped in PR #15024.

### DL-#14939 · Agent Control Stubs Return Honest Not_Implemented Results (#14939).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14939
- **Branch:** merged via #14957
- **PR:** #14957
- **Paths:** see #14957
- **Started:** 2026-11-12
- **Last verified:** 2026-11-12 (`2383b013`; collated from changes/14939-agent-control-stubs-return-honest-not-im.md)
- **Summary:** Agent control stubs return honest not_implemented results (#14939).
- **Next step:** Shipped in PR #14957.

### DL-#14480 · NP-05 Benchmark Localized Edit vs Constrained Fine-Tune (#14480)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14480
- **Branch:** merged via #14828
- **PR:** #14828
- **Paths:** see #14828
- **Started:** 2026-11-12
- **Last verified:** 2026-11-12 (`ecc9b1f0`; collated from changes/14480-np05-edit-vs-finetune-benchmark.md)
- **Summary:** NP-05 benchmark localized edit vs constrained fine-tune (#14480)
- **Next step:** Shipped in PR #14828.

### DL-#15020 · Register Ud-Sim-Mcp Console Script, Generated Capability Inventory, and Agent API Auth Docs.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15020
- **Branch:** merged via #15023
- **PR:** #15023
- **Paths:** see #15023
- **Started:** 2026-11-11
- **Last verified:** 2026-11-11 (`972404f4`; collated from changes/15020-register-ud-sim-mcp-console-script-gener.md)
- **Summary:** Register ud-sim-mcp console script, generated capability inventory, and Agent API auth docs.
- **Next step:** Shipped in PR #15023.

### DL-#14940 · AIP JSON-RPC Stubs Return Not_Implemented; Capabilities Self-Check (#14940).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14940
- **Branch:** merged via #15000
- **PR:** #15000, #14959
- **Paths:** see #15000
- **Started:** 2026-11-10
- **Last verified:** 2026-11-14 (`d91cd296`; collated from changes/14940-aip-json-rpc-stubs-return-not-implemente.md)
- **Summary:** AIP JSON-RPC stubs return not_implemented; capabilities self-check (#14940).
- **Next step:** Shipped in PR #15000.

### DL-#14937 · BallFlightSimulator Reports Per-Step Forces From the Rust Integrator With Legacy Post-Hoc Path Tagged

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14937
- **Branch:** merged via #14962
- **PR:** #14962
- **Paths:** see #14962
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`a423e0d2`; collated from changes/14937-ballflightsimulator-reports-per-step-for.md)
- **Summary:** BallFlightSimulator reports per-step forces from the Rust integrator with legacy post-hoc path tagged
- **Next step:** Shipped in PR #14962.

### DL-#14943 · CounterfactualKind/Result, Simscape/Putting Defaults, Conformance (#14943 Slices 1–4)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14943
- **Branch:** merged via #14998
- **PR:** #14998, #14965, #15005, #14973, #15007
- **Paths:** see #14998
- **Started:** 2026-11-10
- **Last verified:** 2026-11-14 (`96af1f50`; collated from changes/14943-mujoco-tau-generalized-force.md)
- **Summary:** CounterfactualKind/Result, simscape/putting defaults, conformance (#14943 slices 1–4)
- **Next step:** Shipped in PR #14998.

### DL-#14951 · Tauri BackendState Supervisor With Ephemeral Port and Stderr Tail (Part Of #14951).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14951
- **Branch:** merged via #14988
- **PR:** #14988, #14992
- **Paths:** see #14988
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`1d08a591`; collated from changes/14951-tauri-version-consistency-and-dynamic-port.md)
- **Summary:** Tauri BackendState supervisor with ephemeral port and stderr tail (Part of #14951).
- **Next step:** Shipped in PR #14988.

### DL-#14948 · Enforce Required Physics Engine Execution, JUnit Floor Checks, and JaxSim Heavy Test Lane

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14948
- **Branch:** merged via #14977
- **PR:** #14977, #14993, #15004
- **Paths:** see #14977
- **Started:** 2026-11-10
- **Last verified:** 2026-11-14 (`7376c4ab`; collated from changes/14948-conftest-makereport-and-floor.md)
- **Summary:** Enforce required physics engine execution, JUnit floor checks, and JaxSim heavy test lane
- **Next step:** Shipped in PR #14977.

### DL-#14947 · Add Directory-Based Suite Marker Inference, Opt-Out Marker, Gate Selectability Summary, and Audit Utility (Part of #14947, Part Of #14934)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14947
- **Branch:** agy/14947-suite-marker-reach
- **PR:** #14980, #14986
- **Paths:** see #14980
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`b5be04e6`; collated from changes/14947-enable-auto-mark-ci.md)
- **Summary:** Add directory-based suite marker inference, opt-out marker, gate selectability summary, and audit utility (Part of #14947, Part of #14934)
- **Next step:** Shipped in PR #14980.

### DL-#14941 · Workspace Synchronous Optimization Returns Explicit Synthetic Placeholder; RECEIPT Registration Refused for Synthetic Results

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14941
- **Branch:** merged via #14963
- **PR:** #14963
- **Paths:** see #14963
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`2da54750`; collated from changes/14941-workspace-synchronous-optimization-retur.md)
- **Summary:** Workspace synchronous optimization returns explicit synthetic placeholder; RECEIPT registration refused for synthetic results
- **Next step:** Shipped in PR #14963.

### DL-#14938 · \ Marker IK Exposes Convergence Status,Reports NO_MARKERS and NaN RMS for Empty Frames,and Routes LM With Finite Bounds to TRF\

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14938
- **Branch:** agy/14938-marker-ik-convergence
- **PR:** #14974
- **Paths:** see #14974
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`bb977b68`; collated from changes/14938-marker-ik-convergence.md)
- **Summary:** \ Marker IK exposes convergence status,reports NO_MARKERS and NaN RMS for empty frames,and routes LM with finite bounds to TRF\
- **Next step:** Shipped in PR #14974.

### DL-#14946 · Make Coverage Truth Explicit in CI Summary and Remove Dead Cov-Fail-Under Flags

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14946
- **Branch:** merged via #14972
- **PR:** #14972, #14981
- **Paths:** see #14972
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`be9e042d`; collated from changes/14946-sharded-coverage-ratchet.md)
- **Summary:** Make coverage truth explicit in CI summary and remove dead cov-fail-under flags
- **Next step:** Shipped in PR #14972.

### DL-#14950 · Simulation WS Error Frame Handling and isPaused Reset on Stop (Part Of #14950).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14950
- **Branch:** merged via #14958
- **PR:** #14958, #14967, #15001
- **Paths:** see #14958
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`6a566a98`; collated from changes/14950-sim-run-lifecycle-reducer.md)
- **Summary:** Simulation WS error frame handling and isPaused reset on stop (Part of #14950).
- **Next step:** Shipped in PR #14958.

### DL-#14945 · Make Architecture and File-Size Budget Expiry Evaluation Deterministic

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14945
- **Branch:** merged via #14954
- **PR:** #14954
- **Paths:** see #14954
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`268f76ed`; collated from changes/14945-deterministic-budget-expiry.md)
- **Summary:** Make architecture and file-size budget expiry evaluation deterministic
- **Next step:** Shipped in PR #14954.

### DL-#14944 · ZTCF: Guard Multi-Body Misuse of Compute_Ztcf_Forces; Double-Pendulum Regression (#14944, Part Of #14934)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14944
- **Branch:** merged via #14953
- **PR:** #14953, #14979
- **Paths:** see #14953
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`975141a2`; collated from changes/14944-ztcf-rnea-contract.md)
- **Summary:** ZTCF: guard multi-body misuse of compute_ztcf_forces; double-pendulum regression (#14944, Part of #14934)
- **Next step:** Shipped in PR #14953.

### DL-#14227 · Document CI Optional Stack Slice 1 (Pinocchio Gravity) and Add Runbook Contract Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14227
- **Branch:** merged via #14923
- **PR:** #14923, #15089, #15158, #15176
- **Paths:** see #14923
- **Started:** 2026-11-10
- **Last verified:** 2026-11-24 (`cc10309b`; collated from changes/14227-optional-stack-scheduled-green.md)
- **Summary:** Document CI Optional Stack slice 1 (Pinocchio gravity) and add runbook contract tests
- **Next step:** Shipped in PR #14923.

### DL-#13604 · NSX Salvage Tracker: Unit Receipts for Merged NSX-22/NSX-24 Rows on Main (#13604)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13604
- **Branch:** merged via #14930
- **PR:** #14930
- **Paths:** see #14930
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`2d4fde75`; collated from changes/13604-nsx-salvage-tracker-receipts.md)
- **Summary:** NSX salvage tracker: unit receipts for merged NSX-22/NSX-24 rows on main (#13604)
- **Next step:** Shipped in PR #14930.

### DL-#14860 · Redact Fleet Host and Home Literals Batch 4 Recovery

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14860
- **Branch:** merged via #14933
- **PR:** #14933
- **Paths:** see #14933
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`1cc97639`; collated from changes/14860-host-redact-04.md)
- **Summary:** Redact fleet host and home literals batch 4 recovery
- **Next step:** Shipped in PR #14933.

### DL-#14820 · UD Docs and CI Contract Tests Reaffirm PR Size Is Advisory Only (Dieter 2026-11-06)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14820
- **Branch:** merged via #14926
- **PR:** #14926
- **Paths:** see #14926
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`ee388e1a`; collated from changes/14820-ud-pr-size-advisory-review-contract.md)
- **Summary:** UD docs and CI contract tests reaffirm PR size is advisory only (Dieter 2026-11-06)
- **Next step:** Shipped in PR #14926.

### DL-#14928 · Align Realtime Shadow-Ledger Sunset_Date With Q4 Baseline and Validate Against UTC Date (#14928, #12046)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14928
- **Branch:** merged via #14931
- **PR:** #14931
- **Paths:** see #14931
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`a3e2f0f1`; collated from changes/14928-realtime-shadow-sunset.md)
- **Summary:** Align realtime shadow-ledger sunset_date with Q4 baseline and validate against UTC date (#14928, #12046)
- **Next step:** Shipped in PR #14931.

### DL-#12878 · Explorer Contract: Engine-Tagged Playback Archives and Per-Engine MP4 Distinctness Test (#12878 Workstream C)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12878
- **Branch:** merged via #14921
- **PR:** #14921
- **Paths:** see #14921
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`f6c43790`; collated from changes/12878-explorer-contract-engine-tagged-playback.md)
- **Summary:** Explorer contract: engine-tagged playback archives and per-engine MP4 distinctness test (#12878 workstream C)
- **Next step:** Shipped in PR #14921.

### DL-#14905 · Preserve #14733 Proposal Durable Change-History Evidence Across Normal Change-Fragment Collation and Log Archival

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14905
- **Branch:** merged via #14908
- **PR:** #14908
- **Paths:** see #14908
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`a89dd7dd`; collated from changes/14905-preserve-proposal-collation-evidence.md)
- **Summary:** Preserve #14733 proposal durable change-history evidence across normal change-fragment collation and log archival
- **Next step:** Shipped in PR #14908.

### DL-#14897 · Point Driver Motion-Matching Smoke to Canonical Capture-Fit Spec (Full_Body_Spec_Anthro_Driver_Capture.Json) and Align Anthro_Driver Receipt Base_Spec_File Metadata

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14897
- **Branch:** merged via #14901
- **PR:** #14901
- **Paths:** see #14901
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`1c3126cd`; collated from changes/14897-motion-matching-smoke-driver-spec.md)
- **Summary:** Point driver motion-matching smoke to canonical capture-fit spec (full_body_spec_anthro_driver_capture.json) and align anthro_driver receipt base_spec_file metadata
- **Next step:** Shipped in PR #14901.

### DL-#14795 · SHOW-A19: Re-Characterise Wrist and Scapula ROM From Evidence and Evaluate Option (C) Address Reachability

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14795
- **Branch:** merged via #14890
- **PR:** #14890
- **Paths:** see #14890
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`632c36c9`; collated from changes/14795-show-a19-wrist-scapula-rom.md)
- **Summary:** SHOW-A19: re-characterise wrist and scapula ROM from evidence and evaluate option (c) address reachability
- **Next step:** Shipped in PR #14890.

### DL-#14733 · Proposal (Write-Up Only): #13560 OpenSim Net-Moment Gap - Recorder Inconsistency Finding, Conditioning Math, Options

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14733
- **Branch:** merged via #14888
- **PR:** #14888
- **Paths:** see #14888
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`36155e93`; collated from changes/14733-opensim-net-moment-gap-proposal.md)
- **Summary:** Proposal (write-up only): #13560 OpenSim net-moment gap - recorder inconsistency finding, conditioning math, options
- **Next step:** Shipped in PR #14888.

### DL-#14500 · NP-07 Loop-Constraint Projection of Policy Output (Grip Loop, Diagnostic) (#14500)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14500
- **Branch:** merged via #14824
- **PR:** #14824
- **Paths:** see #14824
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`884bf8f5`; collated from changes/14500-np07-loop-projection.md)
- **Summary:** NP-07 loop-constraint projection of policy output (grip loop, diagnostic) (#14500)
- **Next step:** Shipped in PR #14824.

### DL-#14477 · NP-02 Demonstration Fixture and Behavior-Cloned 64X64 MLP Baseline (#14477)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14477
- **Branch:** merged via #14839
- **PR:** #14839
- **Paths:** see #14839
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`e01522fd`; collated from changes/14477-np02-bc-mlp-baseline.md)
- **Summary:** NP-02 demonstration fixture and behavior-cloned 64x64 MLP baseline (#14477)
- **Next step:** Shipped in PR #14839.

### DL-#14763 · SHOW-A18: ROM Diagnostic Decomposing A2 Trail-Wrist Excursions Against Pre-A2 Capture and G2 Options

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14763
- **Branch:** merged via #14886
- **PR:** #14886
- **Paths:** see #14886
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`38cfcf83`; collated from changes/14763-show-a18-rwinputy-range.md)
- **Summary:** SHOW-A18: ROM diagnostic decomposing A2 trail-wrist excursions against pre-A2 capture and G2 options
- **Next step:** Shipped in PR #14886.

### DL-#14796 · SHOW-A20: Refit Capture Club Pose per Trail-Grip Option Removing Copy-Grip Bias

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14796
- **Branch:** merged via #14883
- **PR:** #14883
- **Paths:** see #14883
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`3ee20382`; collated from changes/14796-show-a20-refit-club-pose.md)
- **Summary:** SHOW-A20: refit capture club pose per trail-grip option removing copy-grip bias
- **Next step:** Shipped in PR #14883.

### DL-#14797 · SHOW-A21: Evidence-Based Grip Guards (Dorsal Separation >= 90 Deg and Trail V-Offset Bound in [-75, 75] Deg Grounded in Biomechanical Literature and Tour Data)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14797
- **Branch:** merged via #14882
- **PR:** #14882
- **Paths:** see #14882
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`29f9fac0`; collated from changes/14797-evidence-grip-guards.md)
- **Summary:** SHOW-A21: evidence-based grip guards (dorsal separation >= 90 deg and trail V-offset bound in [-75, 75] deg grounded in biomechanical literature and tour data)
- **Next step:** Shipped in PR #14882.

### DL-#14735 · Key-Frame PNG Writer and Grip-Clock Inset Helper for Styled Renders (SHOW-A10)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14735
- **Branch:** merged via #14880
- **PR:** #14880, #14900
- **Paths:** see #14880
- **Started:** 2026-11-07
- **Last verified:** 2026-11-08 (`df2d7034`; collated from changes/14735-show-a10-evidence.md)
- **Summary:** Key-frame PNG writer and grip-clock inset helper for styled renders (SHOW-A10)
- **Next step:** Shipped in PR #14880.

### DL-#14734 · SHOW-A9: Refit Best-Match MuJoCo Driver on A7 Grip + A8 Trunk Spec; Record Candidate Receipt and Baseline Comparison

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14734
- **Branch:** merged via #14878
- **PR:** #14878
- **Paths:** see #14878
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`f304efdc`; collated from changes/14734-show-a9-mujoco-refit.md)
- **Summary:** SHOW-A9: Refit best-match MuJoCo driver on A7 grip + A8 trunk spec; record candidate receipt and baseline comparison
- **Next step:** Shipped in PR #14878.

### DL-#14738 · Fleet Grip Guard: PR-Tier Test That Every Model and Binding Reads 15° ± 1° Under A7 Convention (SHOW-A13, #14738)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14738
- **Branch:** merged via #14875
- **PR:** #14875
- **Paths:** see #14875
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`62a43192`; collated from changes/14738-show-a13-fleet-grip-guard.md)
- **Summary:** Fleet grip guard: PR-tier test that every model and binding reads 15° ± 1° under A7 convention (SHOW-A13, #14738)
- **Next step:** Shipped in PR #14875.

### DL-#14737 · Apply A7 Grip Roll Zero to Drake, Pinocchio, OpenSim and MyoSuite via the Shared Metric

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14737
- **Branch:** merged via #14873
- **PR:** #14873
- **Paths:** see #14873
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`6823740b`; collated from changes/14737-show-a12-cross-engine-grip.md)
- **Summary:** Apply A7 grip roll zero to Drake, Pinocchio, OpenSim and MyoSuite via the shared metric
- **Next step:** Shipped in PR #14873.

### DL-#14736 · SHOW-A11: Validate Active MuJoCo Spec and Grip Bindings Against A7 Grip Roll Convention (+15 Deg Neutral)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14736
- **Branch:** merged via #14869
- **PR:** #14869
- **Paths:** see #14869
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`910c29fa`; collated from changes/14736-show-a11-apply-a7-grip-convention.md)
- **Summary:** SHOW-A11: Validate active MuJoCo spec and grip bindings against A7 grip roll convention (+15 deg neutral)
- **Next step:** Shipped in PR #14869.

### DL-#14428 · Add GM-5 Posture, Translation, Club-at-Impact, and Han Et Al. (2019) GRF Pivoting Moment Metrics.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14428
- **Branch:** merged via #14832
- **PR:** #14832
- **Paths:** see #14832
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`56341fdd`; collated from changes/14428-gm5-posture-impact-metrics.md)
- **Summary:** Add GM-5 posture, translation, club-at-impact, and Han et al. (2019) GRF pivoting moment metrics.
- **Next step:** Shipped in PR #14832.

### DL-#14424 · Split Address-Relative and Target-Line/Lab-Frame Shoulder and Hip Turn Conventions With Declared Segment Proxies and Chu 2010 Event Reporting (GM-1C).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14424
- **Branch:** merged via #14863
- **PR:** #14863
- **Paths:** see #14863
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`19999b17`; collated from changes/14424-gm1c-shoulder-hip-turn-conventions.md)
- **Summary:** Split address-relative and target-line/lab-frame shoulder and hip turn conventions with declared segment proxies and Chu 2010 event reporting (GM-1c).
- **Next step:** Shipped in PR #14863.

### DL-#14731 · Re-Zero Grip Roll to the Owner's A7 Convention: Wrists on Top of Shaft, + CCW Toward Target

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14731
- **Branch:** merged via #14865
- **PR:** #14865
- **Paths:** see #14865
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`f01bfcee`; collated from changes/14731-grip-roll-zero-a7-convention.md)
- **Summary:** Re-zero grip roll to the owner's A7 convention: wrists on top of shaft, + CCW toward target
- **Next step:** Shipped in PR #14865.

### DL-#14786 · Optional-Stack CI Runs Loop_Nullspace Snapshot vs Spatial Fixture With Real Pinocchio

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14786
- **Branch:** merged via #14788
- **PR:** #14788
- **Paths:** see #14788
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`f88d28a2`; collated from changes/14786-optional-stack-ci-runs-loop-nullspace-sn.md)
- **Summary:** Optional-stack CI runs loop_nullspace snapshot vs spatial fixture with real pinocchio
- **Next step:** Shipped in PR #14788.

### DL-#14753 · SHOW-A17: Mirror the Right Arm in Full_Body_Spec_Anthro_Iron7 and Regenerate Its Two Ground-Support Receipts

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14753
- **Branch:** merged via #14862
- **PR:** #14862
- **Paths:** see #14862
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`b6e846e1`; collated from changes/14753-show-a17-mirror-the-right-arm-in-full-bo.md)
- **Summary:** SHOW-A17: mirror the right arm in full_body_spec_anthro_iron7 and regenerate its two ground-support receipts
- **Next step:** Shipped in PR #14862.

### DL-#14747 · Orient the Drawn Grip Hands From the Wrist Follower Frames: The Drawn Dorsum Shell Now Lies on the Wrist Dorsal Axis (Was 35 Deg Lead / 75 Deg Trail Off)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14747
- **Branch:** merged via #14858
- **PR:** #14858
- **Paths:** see #14858
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`e14357ac`; collated from changes/14747-orient-the-drawn-grip-hands-from-the-wri.md)
- **Summary:** Orient the drawn grip hands from the wrist follower frames: the drawn dorsum shell now lies on the wrist dorsal axis (was 35 deg lead / 75 deg trail off)
- **Next step:** Shipped in PR #14858.

### DL-#14732 · SHOW-A8: V2 Lumbar+Thoracic Trunk Into Best-Match Mujoco_Driver Model With Styled Render

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14732
- **Branch:** merged via #14855
- **PR:** #14855
- **Paths:** see #14855
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`6392e761`; collated from changes/14732-show-a8-v2-trunk-best-match.md)
- **Summary:** SHOW-A8: v2 lumbar+thoracic trunk into best-match mujoco_driver model with styled render
- **Next step:** Shipped in PR #14855.

### DL-#13882 · Catalog Exact Existing Regression Nodes and Merged Provenance for Six Resolved Board Findings

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13882
- **Branch:** merged via #14842
- **PR:** #14842
- **Paths:** see #14842
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`f46f90ed`; collated from changes/13882-regression-catalog.md)
- **Summary:** Catalog exact existing regression nodes and merged provenance for six resolved board findings
- **Next step:** Shipped in PR #14842.

### DL-#14751 · G2: Trail-Hand Grip Option (B15) With Explicit 15 Deg Alpha and Widened Trail V-Offset Bound

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14751
- **Branch:** merged via #14819
- **PR:** #14819
- **Paths:** see #14819
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`832f01c9`; collated from changes/14751-g2-trail-hand-grip-option-b15-with-expli.md)
- **Summary:** G2: trail-hand grip option (b15) with explicit 15 deg alpha and widened trail V-offset bound
- **Next step:** Shipped in PR #14819.

### DL-#12983 · Reload Pinocchio in OpenSim Dynamics Parity When Unit Mocks Left Stubs in Sys.Modules

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12983
- **Branch:** merged via #14843
- **PR:** #14843
- **Paths:** see #14843
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`1c58d8d3`; collated from changes/12983-reload-pinocchio-in-opensim-dynamics-par.md)
- **Summary:** Reload pinocchio in OpenSim dynamics parity when unit mocks left stubs in sys.modules
- **Next step:** Shipped in PR #14843.

### DL-#14743 · TableShotModel Reads the Dispersion Cell for the Shot's Lie and the Conditions' Wind/Slope Bins (CourseConditions Gains Wind/Slope, Default Calm/Flat); Absent Wind/Slope Bins Are Rejected Instead of Silently Using Calm/Flat.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14743
- **Branch:** merged via #14784
- **PR:** #14784
- **Paths:** see #14784
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`b85eb218`; collated from changes/14743-tableshotmodel-reads-the-dispersion-cell.md)
- **Summary:** TableShotModel reads the dispersion cell for the shot's lie and the conditions' wind/slope bins (CourseConditions gains wind/slope, default calm/flat); absent wind/slope bins are rejected instead of silently using calm/flat.
- **Next step:** Shipped in PR #14784.

### DL-#14816 · SHOW-B3b: Omit Ground_Model From Carry-Only Dispersion Tables and Deduplicate Runout Entry Points

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14816
- **Branch:** merged via #14833
- **PR:** #14833
- **Paths:** see #14833
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`a425584c`; collated from changes/14816-carry-table-provenance.md)
- **Summary:** SHOW-B3b: omit ground_model from carry-only dispersion tables and deduplicate runout entry points
- **Next step:** Shipped in PR #14833.

### DL-#14501 · NP-08: Offline SVD-Direction Ablation vs High-Frequency Jitter Benchmark (#14501)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14501
- **Branch:** merged via #14823
- **PR:** #14823
- **Paths:** see #14823
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`d4e3a4eb`; collated from changes/14501-np08-svd-jitter-ablation.md)
- **Summary:** NP-08: offline SVD-direction ablation vs high-frequency jitter benchmark (#14501)
- **Next step:** Shipped in PR #14823.

### DL-#14423 · Map X-Factor Variants to Literature Families and Add Publish Payload + Unit Tests (GM-1B).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14423
- **Branch:** merged via #14818
- **PR:** #14818
- **Paths:** see #14818
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`f0cfeb27`; collated from changes/14423-map-x-factor-variants-to-literature-fami.md)
- **Summary:** Map X-factor variants to literature families and add publish payload + unit tests (GM-1b).
- **Next step:** Shipped in PR #14818.

### DL-#13884 · Controller Synergy Audit Slice 1: ControllerAuditSpec, #13837 Trace Adapter and ControllerAuditReport (UCM, TNC Fail-Closed Below 30 Trials, Null-Space Gain Norm) With an Analytic Lyapunov Golden

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13884
- **Branch:** merged via #14768
- **PR:** #14768, #14856
- **Paths:** see #14768
- **Started:** 2026-11-06
- **Last verified:** 2026-11-07 (`2f7dab23`; collated from changes/13884-controller-synergy-audit-slice-2-minimum.md)
- **Summary:** Controller synergy audit slice 1: ControllerAuditSpec, #13837 trace adapter and ControllerAuditReport (UCM, TNC fail-closed below 30 trials, null-space gain norm) with an analytic Lyapunov golden
- **Next step:** Shipped in PR #14768.

### DL-#14812 · Rewrite Pose_Studio Test_Widgets Against Real Qt Widgets With Qapp Fixture (#14812)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14812
- **Branch:** merged via #14815
- **PR:** #14815
- **Paths:** see #14815
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`5d662cbb`; collated from changes/14812-pose-studio-test-widgets.md)
- **Summary:** rewrite pose_studio test_widgets against real Qt widgets with qapp fixture (#14812)
- **Next step:** Shipped in PR #14815.

### DL-#14810 · Pose_Studio Test_Gui Requests a Real QApplication via an Autouse Qapp Fixture, so It No Longer Crashes an Xdist Worker When No QApplication Exists Yet

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14810
- **Branch:** merged via #14811
- **PR:** #14811
- **Paths:** see #14811
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`02450ff9`; collated from changes/14810-pose-studio-test-gui-requests-a-real-qap.md)
- **Summary:** pose_studio test_gui requests a real QApplication via an autouse qapp fixture, so it no longer crashes an xdist worker when no QApplication exists yet
- **Next step:** Shipped in PR #14811.

### DL-#14807 · Drake UI Abort Regression Tests Request Qapp via an Autouse Fixture, so They No Longer Crash an Xdist Worker When No QApplication Exists Yet

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14807
- **Branch:** merged via #14808
- **PR:** #14808
- **Paths:** see #14808
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`26bb158a`; collated from changes/14807-drake-ui-abort-regression-tests-request.md)
- **Summary:** Drake UI abort regression tests request qapp via an autouse fixture, so they no longer crash an xdist worker when no QApplication exists yet
- **Next step:** Shipped in PR #14808.

### DL-#14767 · Leaderboard Freshness Gate Moved to Scripts/Ci/check_leaderboard_freshness.py; a PR That Changes No Motion_Matching Solver Input Passes Despite a Squash-Orphaned Git_Head, Solver Changes Still Need a Rerun.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14767
- **Branch:** merged via #14783
- **PR:** #14783
- **Paths:** see #14783
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`7031b228`; collated from changes/14767-leaderboard-freshness-gate-moved-to-scri.md)
- **Summary:** Leaderboard freshness gate moved to scripts/ci/check_leaderboard_freshness.py; a PR that changes no motion_matching solver input passes despite a squash-orphaned git_head, solver changes still need a rerun.
- **Next step:** Shipped in PR #14783.

### DL-#14756 · P5: Parity Report Candidate Resolution Fails Closed; Canonical Checksum/Schema/Manifest Failures Never Fall Through to Legacy Loading, Legacy Numeric Archives Detected Explicitly, No Allow_Pickle=True

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14756
- **Branch:** merged via #14770
- **PR:** #14770, #15165, #15373, #15587, #15613, #15594, #15602, #15578, #15576, #15580
- **Paths:** see #14770
- **Started:** 2026-11-06
- **Last verified:** 2026-11-24 (`a7326c81`; collated from changes/14756-trunk-bundle-source.md)
- **Summary:** P5: parity report candidate resolution fails closed; canonical checksum/schema/manifest failures never fall through to legacy loading, legacy numeric archives detected explicitly, no allow_pickle=True
- **Next step:** Independent review; keep PR unarmed.

### DL-#14744 · Strategy CLI: --Shot-Table / --Shot-Model Inject the Selected Shot Model Into the Personal Solve and Report Its Provenance in Text and HTML

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14744
- **Branch:** merged via #14793
- **PR:** #14793
- **Paths:** see #14793
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`87681255`; collated from changes/14744-strategy-cli-shot-table-shot-model-injec.md)
- **Summary:** strategy CLI: --shot-table / --shot-model inject the selected shot model into the personal solve and report its provenance in text and HTML
- **Next step:** Shipped in PR #14793.

### DL-#14727 · Commit the V1 Source Spec of the Trunk_Visuals Driver V2 Bundle and Pin the Generator Against the Committed Rollout on Its True Inputs (Step 1)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14727
- **Branch:** merged via #14785
- **PR:** #14785, #14898
- **Paths:** see #14785
- **Started:** 2026-11-06
- **Last verified:** 2026-11-08 (`4bd0ae50`; collated from changes/14727-drake-v2-reference.md)
- **Summary:** Commit the v1 source spec of the trunk_visuals driver v2 bundle and pin the generator against the committed rollout on its true inputs (step 1)
- **Next step:** Shipped in PR #14785.

### DL-#14633 · Re-Base HX Grip Alpha to Face-On Zero at Flagship Address; Preserve Fitted Euler Triples.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14633
- **Branch:** merged via #14639
- **PR:** #14639
- **Paths:** see #14639
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`25a0c0a6`; collated from changes/14633-re-base-hx-grip-alpha-to-face-on-zero-at.md)
- **Summary:** Re-base HX grip alpha to face-on zero at flagship address; preserve fitted Euler triples.
- **Next step:** Shipped in PR #14639.

### DL-#14621 · Flight Model Validity: Launch Spin Cap, Trajectory Guard, Pipeline Metadata

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14621
- **Branch:** merged via #14641
- **PR:** #14641
- **Paths:** see #14641
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`74f43930`; collated from changes/14621-flight-model-validity-launch-spin-cap-tr.md)
- **Summary:** Flight model validity: launch spin cap, trajectory guard, pipeline metadata
- **Next step:** Shipped in PR #14641.

### DL-#14503 · NP-10 Offline Weight Inspector and Lexical CI Guardrail (#14503)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14503
- **Branch:** merged via #14817
- **PR:** #14817
- **Paths:** see #14817
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`c0739cf2`; collated from changes/14503-np10-weight-inspector.md)
- **Summary:** NP-10 offline weight inspector and lexical CI guardrail (#14503)
- **Next step:** Shipped in PR #14817.

### DL-#14405 · Bump PR Body Lint Caller Pin to the Post-RM#2697 Reusable Workflow (RM#2427)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14405
- **Branch:** merged via #14791
- **PR:** #14791
- **Paths:** see #14791
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`2c971605`; collated from changes/14405-bump-pr-body-lint-caller-pin-to-the-post.md)
- **Summary:** Bump PR Body Lint caller pin to the post-RM#2697 reusable workflow (RM#2427)
- **Next step:** Shipped in PR #14791.

### DL-#12583 · NSX-55: Document DV-12583 as Present on Main, Add Catalog Regression Test, Fix Publication_Gate Checklist

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12583
- **Branch:** merged via #14806
- **PR:** #14806
- **Paths:** see #14806
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`c32c76dc`; collated from changes/12583-nsx55-dv-catalog-gate-and-publication-checklis.md)
- **Summary:** NSX-55: document DV-12583 as present on main, add catalog regression test, fix publication_gate checklist
- **Next step:** Shipped in PR #14806.

### DL-#14771 · Ground-Support Segment-Scale Tie-Break, Receipt Canonical Compare (#14771)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14771
- **Branch:** merged via #14777
- **PR:** #14777
- **Paths:** see #14777
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`d0705fe6`; collated from changes/14771-ground-support-segment-scale-tie-break-r.md)
- **Summary:** Ground-support segment-scale tie-break, receipt canonical compare (#14771)
- **Next step:** Shipped in PR #14777.

### DL-#14757 · NSX-13 Spatial-Fixture Test Double Builds the R11 Shaft Springs Like NativePinocchioModel (Rigid Club), so Test_Snapshot_vs_Spatial_Fixture Is Green Again

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14757
- **Branch:** merged via #14759
- **PR:** #14759
- **Paths:** see #14759
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`a2cba942`; collated from changes/14757-nsx-13-spatial-fixture-test-double-build.md)
- **Summary:** NSX-13 spatial-fixture test double builds the R11 shaft springs like NativePinocchioModel (rigid club), so test_snapshot_vs_spatial_fixture is green again
- **Next step:** Shipped in PR #14759.

### DL-#14746 · SHOW-G1: Mirror the Right-Arm Joint Axes in Driver V1/V2 (R Scapula, Shoulder, Forearm, Wrist) so Positive L and R Coordinates Give Mirrored Motion

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14746
- **Branch:** merged via #14762
- **PR:** #14762
- **Paths:** see #14762
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`a4765a14`; collated from changes/14746-show-g1-mirror-the-right-arm-joint-axes.md)
- **Summary:** SHOW-G1: mirror the right-arm joint axes in driver v1/v2 (R scapula, shoulder, forearm, wrist) so positive L and R coordinates give mirrored motion
- **Next step:** Shipped in PR #14762.

### DL-#14741 · Add PR Size Gate Workflow and Git Numstat Checker (500-Line Cap, Exemptions).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14741
- **Branch:** merged via #14774
- **PR:** #14774, #14820
- **Paths:** see #14774
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`f46eba70`; collated from changes/14741-pr-size-advisory-bdb9.md)
- **Summary:** Add PR size gate workflow and git numstat checker (500-line cap, exemptions).
- **Next step:** Shipped in PR #14774.

### DL-#14478 · NP-03/04 Toy MLP Rank-1 Edit and LoRA Comparator (#14478 #14479)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14478
- **Branch:** merged via #14778
- **PR:** #14778
- **Paths:** see #14778
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`716ce0bc`; collated from changes/14478-np-03-04-toy-mlp-rank-1-edit-and-lora-co.md)
- **Summary:** NP-03/04 toy MLP rank-1 edit and LoRA comparator (#14478 #14479)
- **Next step:** Shipped in PR #14778.

### DL-#14422 · Add Pydantic Golf Metric-Definitions Registry Schema (GM-1a) With Filter Descriptor.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14422
- **Branch:** merged via #14776
- **PR:** #14776
- **Paths:** see #14776
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`78676865`; collated from changes/14422-add-pydantic-golf-metric-definitions-reg.md)
- **Summary:** Add Pydantic golf metric-definitions registry schema (GM-1a) with filter descriptor.
- **Next step:** Shipped in PR #14776.

### DL-#12724 · S6 Catalogue Reason Now Names the Missing Arm on the NSX-97 UCM Split

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12724
- **Branch:** merged via #14761
- **PR:** #14761, #15068
- **Paths:** see #14761
- **Started:** 2026-11-06
- **Last verified:** 2026-11-14 (`36f6dbbf`; collated from changes/12724-nsx-95-campaign-catalogue-json-strategy.md)
- **Summary:** S6 catalogue reason now names the missing arm on the NSX-97 UCM split
- **Next step:** Shipped in PR #14761.

### DL-#12721 · Xfail Placeholder for Anthro Noise-Aware Receipt (#12721)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12721
- **Branch:** merged via #14779
- **PR:** #14779
- **Paths:** see #14779
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`7bf79325`; collated from changes/12721-xfail-placeholder-for-anthro-noise-aware.md)
- **Summary:** xfail placeholder for anthro noise-aware receipt (#12721)
- **Next step:** Shipped in PR #14779.

### DL-#12719 · Event_Outcome_Rows Refuses a Near-Grazing Impact (Transversality Below GRAZING_TRANSVERSALITY) Instead of Returning Unreliable Linear Rows

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12719
- **Branch:** merged via #14782
- **PR:** #14782
- **Paths:** see #14782
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`05fb6000`; collated from changes/12719-event-outcome-rows-refuses-a-near-grazin.md)
- **Summary:** event_outcome_rows refuses a near-grazing impact (transversality below GRAZING_TRANSVERSALITY) instead of returning unreliable linear rows
- **Next step:** Shipped in PR #14782.

### DL-#14748 · Native_Viewer_Export CLI Explains a Foreign Shared Package Shadowing the Vendored Tools Tree

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14748
- **Branch:** merged via #14769
- **PR:** #14769, #14809
- **Paths:** see #14769
- **Started:** 2026-11-06
- **Last verified:** 2026-11-07 (`f111638e`; collated from changes/14748-native-viewer-export-bootstrap-imports-a.md)
- **Summary:** native_viewer_export CLI explains a foreign shared package shadowing the vendored Tools tree
- **Next step:** Shipped in PR #14769.

### DL-#14739 · Simscape Grip Docs State the SHOW-A7 Grip Zero (#14731); a MATLAB Test Checks the Gs3dx Topology Frames; the R2025b Address-Roll Check Is Cataloged as Deferred

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14739
- **Branch:** merged via #14772
- **PR:** #14772
- **Paths:** see #14772
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`90c8e841`; collated from changes/14739-simscape-grip-docs-state-the-show-a7-gri.md)
- **Summary:** Simscape grip docs state the SHOW-A7 grip zero (#14731); a MATLAB test checks the gs3dx topology frames; the R2025b address-roll check is cataloged as deferred
- **Next step:** Shipped in PR #14772.

### DL-#12737 · SPN-0B data_extraction.md: DOIs and Full Author Lists for Pearcy 1984, Fujii 2007, Wilke 1999 and Wu 2002 From Crossref; No Verified_From_Primary Flag Changed

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12737
- **Branch:** merged via #14760
- **PR:** #14760, #14924
- **Paths:** see #14760
- **Started:** 2026-11-06
- **Last verified:** 2026-11-10 (`ee2d7b2c`; collated from changes/12737-contract-test-bruno-b2-supplement-stays.md)
- **Summary:** SPN-0b data_extraction.md: DOIs and full author lists for Pearcy 1984, Fujii 2007, Wilke 1999 and Wu 2002 from Crossref; no verified_from_primary flag changed
- **Next step:** Shipped in PR #14760.

### DL-#12684 · Pin the R4 Capture-Mode Address Knee on the Canonical Receipts and Record Their Marker RMS Before/After (Acceptance 6)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12684
- **Branch:** merged via #14758
- **PR:** #14758
- **Paths:** see #14758
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`410d9f58`; collated from changes/12684-pin-the-r4-capture-mode-address-knee-on.md)
- **Summary:** Pin the R4 capture-mode address knee on the canonical receipts and record their marker RMS before/after (acceptance 6)
- **Next step:** Shipped in PR #14758.

### DL-#13044 · OSL-7 Predictive Deadlift Receipt Now Records the Tracked Comparison as Unavailable (Non-Converged Solve) via a No-Solve --Refresh-Comparison Mode; lift_models.tex Notes the Exploratory Status.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13044
- **Branch:** merged via #14766
- **PR:** #14766
- **Paths:** see #14766
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`8065fb0f`; collated from changes/13044-osl-7-predictive-deadlift-receipt-now-re.md)
- **Summary:** OSL-7 predictive deadlift receipt now records the tracked comparison as unavailable (non-converged solve) via a no-solve --refresh-comparison mode; lift_models.tex notes the exploratory status.
- **Next step:** Shipped in PR #14766.

### DL-#14749 · Fix Lod-Quality-Gate Regression: TableShotModel Asks DispersionTable.Club_Label Instead of Reaching Into Provenance (Law of Demeter)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14749
- **Branch:** merged via #14752
- **PR:** #14752
- **Paths:** see #14752
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`5a89fb4d`; collated from changes/14749-fix-lod-quality-gate-regression-tablesho.md)
- **Summary:** Fix lod-quality-gate regression: TableShotModel asks DispersionTable.club_label instead of reaching into provenance (Law of Demeter)
- **Next step:** Shipped in PR #14752.

### DL-#14703 · Putting Studio: Exact Imperial Ft Factor, a Parity Test for the Core Green Edge Clamp, and Locked Slope Sliders That Read the Core Green's Slope at the Ball

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14703
- **Branch:** merged via #14709
- **PR:** #14709
- **Paths:** see #14709
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`c179ec77`; collated from changes/14703-putting-studio-exact-imperial-ft-factor.md)
- **Summary:** Putting studio: exact imperial ft factor, a parity test for the core green edge clamp, and locked slope sliders that read the core green's slope at the ball
- **Next step:** Shipped in PR #14709.

### DL-#13779 · Knot_Gradient_Optimiser: Lbfgs_Minimise Records a Per-Evaluation Objective Versus Gradient Timing Split and the Projected-Gradient Norm (Per Evaluation and at SciPy's Final Point) so a Max_Iterations Stop Is Diagnosable; Results Are Pinned Bit-for-Bit

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13779
- **Branch:** merged via #14723
- **PR:** #14723, #14775
- **Paths:** see #14723
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`3d68d51d`; collated from changes/13779-mjx-l-bfgs-parameter-scale-scipy-ftol-gt.md)
- **Summary:** knot_gradient_optimiser: lbfgs_minimise records a per-evaluation objective versus gradient timing split and the projected-gradient norm (per evaluation and at SciPy's final point) so a max_iterations stop is diagnosable; results are pinned bit-for-bit
- **Next step:** Shipped in PR #14723.

### DL-#13057 · Lifting: The Four Strict Residual-Gate Xfails (Deadlift, Bench Press, Snatch, Clean and Jerk) Now Cite #13057, the Hip and Pelvis Rebuild That Owns the Residual Excess, Instead Of #13041

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13057
- **Branch:** merged via #14730
- **PR:** #14730, #14802
- **Paths:** see #14730
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`9569ef7e`; collated from changes/13057-osl-14-scoping-osl-scoping-osl14-fixture.md)
- **Summary:** lifting: the four strict residual-gate xfails (deadlift, bench press, snatch, clean and jerk) now cite #13057, the hip and pelvis rebuild that owns the residual excess, instead of #13041
- **Next step:** Shipped in PR #14730.

### DL-#14698 · Add the Lighthouse Mobile Baseline Receipt (Loopback Fixture Server and Egress Proxy, Report Validator, CI Steps on the Fleet Runner) as the Follow-Up to #14270; Scores Informational, No Threshold.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14698
- **Branch:** merged via #14717
- **PR:** #14717, #14813
- **Paths:** see #14717
- **Started:** 2026-11-05
- **Last verified:** 2026-11-06 (`a5fb78f4`; collated from changes/14698-ci-lighthouse-mobile-baseline-runs-in-it.md)
- **Summary:** Add the Lighthouse mobile baseline receipt (loopback fixture server and egress proxy, report validator, CI steps on the fleet runner) as the follow-up to #14270; scores informational, no threshold.
- **Next step:** Shipped in PR #14717.

### DL-#14610 · SHOW-B3b: Bounce/Roll Rest Positions (Sg_Runout_V1) on the B6 Slope Bins, CRN Lie Modifier for Dispersion Tables

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14610
- **Branch:** merged via #14708
- **PR:** #14708, #14726, #14642
- **Paths:** see #14708
- **Started:** 2026-11-05
- **Last verified:** 2026-11-06 (`ac859070`; collated from changes/14610-show-b3b-dispersion-ground-runout.md)
- **Summary:** SHOW-B3b: bounce/roll rest positions (sg_runout_v1) on the B6 slope bins, CRN lie modifier for dispersion tables
- **Next step:** Shipped in PR #14708.

### DL-#12769 · SPN-14a: Fixture_Peak_Spread_Table Serialises Law Drive Residuals and Lumbar Compression Spreads; the Invented-Fixture Table Is Committed as a CI Golden (Tests/Fixtures/Spine/Spn14a_Fixture_Peaks.Json)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12769
- **Branch:** merged via #14716
- **PR:** #14716
- **Paths:** see #14716
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`6bcba0c7`; collated from changes/12769-spn14a-fixture-golden.md)
- **Summary:** SPN-14a: fixture_peak_spread_table serialises law drive residuals and lumbar compression spreads; the invented-fixture table is committed as a CI golden (tests/fixtures/spine/spn14a_fixture_peaks.json)
- **Next step:** Shipped in PR #14716.

### DL-#14452 · Putting Parity Matrix: Units.React and Units.Tauri Move to Present, Citing the React Units Selector That #14458 Landed (PR #14489); No Putting Gaps Remain in the Matrix (Part Of #14452)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14452
- **Branch:** merged via #14712
- **PR:** #14712
- **Paths:** see #14712
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`5eee5589`; collated from changes/14452-units-matrix-cell.md)
- **Summary:** putting parity matrix: units.react and units.tauri move to present, citing the React units selector that #14458 landed (PR #14489); no putting gaps remain in the matrix (Part of #14452)
- **Next step:** Shipped in PR #14712.

### DL-#14702 · Putting_Dynamics: The Post-Impact Hop Reflects the Vertical Speed Relative to the Green, so a Downhill Putt Settles in Under 0.1 s Instead of Skipping for About 5 s; the Studio Long-Downhill Read Is Back in Core Parity

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14702
- **Branch:** merged via #14705
- **PR:** #14705
- **Paths:** see #14705
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`b76beaa7`; collated from changes/14702-downhill-hop.md)
- **Summary:** putting_dynamics: the post-impact hop reflects the vertical speed relative to the green, so a downhill putt settles in under 0.1 s instead of skipping for about 5 s; the studio long-downhill read is back in core parity
- **Next step:** Shipped in PR #14705.

### DL-#12691 · R11: Per-Engine Chain-Shaft Impact-Window Receipt and Overlay Now Cover Drake and Pinocchio, With an Engine-Free Gate

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12691
- **Branch:** merged via #14692
- **PR:** #14692, #14773, #15554
- **Paths:** see #14692
- **Started:** 2026-11-05
- **Last verified:** 2026-11-24 (`3c07a1c2`; collated from changes/12691-gs3dx-design-report-rigid-shaft.md)
- **Summary:** R11: per-engine chain-shaft impact-window receipt and overlay now cover drake and pinocchio, with an engine-free gate
- **Next step:** Shipped in PR #14692.

### DL-#12567 · NSX-24: Unify ContactForceAllocator and MultiEngineTorqueAllocator on Allocate_Contact_Frame Conic Path

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12567
- **Branch:** merged via #14699
- **PR:** #14699
- **Paths:** see #14699
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`ef08fbcb`; collated from changes/12567-nsx-24-unify-contactforceallocator-and-m.md)
- **Summary:** NSX-24: unify ContactForceAllocator and MultiEngineTorqueAllocator on allocate_contact_frame conic path
- **Next step:** Shipped in PR #14699.

### DL-#14677 · Sg_Optimizer: Common Random Numbers for an Injected ShotOutcomeModel so the Scalar Backup Converges

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14677
- **Branch:** merged via #14688
- **PR:** #14688
- **Paths:** see #14688
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`fe666fa5`; collated from changes/14677-sg-optimizer-common-random-numbers-for-a.md)
- **Summary:** sg_optimizer: common random numbers for an injected ShotOutcomeModel so the scalar backup converges
- **Next step:** Shipped in PR #14688.

### DL-#14650 · Putting Studio: Batch Strokes Gained, the E(d) Chart and the Optimal Pace Read the Core's Solved Player E(d) From Core_Sg.Js (Service_Sg.Sg_Study, Tour-Average Player, Side-Slope Grid), Replacing the Hand-Tuned Comeback Fit

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14650
- **Branch:** merged via #14689
- **PR:** #14689, #14695
- **Paths:** see #14689
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`37294a10`; collated from changes/14650-studio-pace.md)
- **Summary:** Putting studio: batch strokes gained, the E(d) chart and the optimal pace read the core's solved player E(d) from core_sg.js (service_sg.sg_study, tour-average player, side-slope grid), replacing the hand-tuned comeback fit
- **Next step:** Shipped in PR #14689.

### DL-#12553 · NSX-04 Slice 1: Verified Subtalar Inversion/Eversion and MTP Flexion Torque Capacities (da Fonseca 2025, Saeki 2021); Hip and MTP-Extension Slots Keep ASSUMED With Stated Reasons

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12553
- **Branch:** merged via #14684
- **PR:** #14684, #14690, #14696, #14704, #14713, #14725, #14787, #14925, #15064, #15174
- **Paths:** see #14684
- **Started:** 2026-11-05
- **Last verified:** 2026-11-24 (`316a2989`; collated from changes/12553-anatomical-torque-slots-acceptance.md)
- **Summary:** NSX-04 slice 1: verified subtalar inversion/eversion and MTP flexion torque capacities (da Fonseca 2025, Saeki 2021); hip and MTP-extension slots keep ASSUMED with stated reasons
- **Next step:** Shipped in PR #14684.

### DL-#14649 · Putting Studio: Auto-Solve Runs solveAimCore, a Port of the Core Solve_Aim (Pace Bisection and Heading Secant) Golden-Pinned to the Core Read on the Core Greens, and the Batch Strikes at the Solved Head Speed While the Read Is Current

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14649
- **Branch:** merged via #14683
- **PR:** #14683, #14678
- **Paths:** see #14683
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`032b2de4`; collated from changes/14649-studio-roll.md)
- **Summary:** Putting studio: Auto-Solve runs solveAimCore, a port of the core solve_aim (pace bisection and heading secant) golden-pinned to the core read on the core greens, and the batch strikes at the solved head speed while the read is current
- **Next step:** Shipped in PR #14683.

### DL-#14572 · SHOW-B8: Strategy CLI and HTML View With Tour vs Personal Strokes, Shot Options and Dispersion Ellipses

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14572
- **Branch:** merged via #14659
- **PR:** #14659
- **Paths:** see #14659
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`8b83fe3b`; collated from changes/14572-show-b8-strategy-cli-and-view.md)
- **Summary:** SHOW-B8: strategy CLI and HTML view with tour vs personal strokes, shot options and dispersion ellipses
- **Next step:** Shipped in PR #14659.

### DL-#14551 · Neutral Address Grip: Flagship Driver Grip Set to the HX-Canonical Grip (Alpha -10.73, Psi 40, Gamma 0) so Both Hands Read 15 Deg Face-On (Was 98/108 Deg); Address and Swing Fixtures Re-Solved. Deferred to A6 (#14555): At Address the RW Wrist Reads -41.8 Deg (Range -40..25) and the Driver Swing Leaves the Wrist Ranges (RWInputY 691 Frames to -130 Deg, RWInputX 66 Frames, LWInputY 216 Frames); Numbers in Neutral_Grip.Json. Capture-Matching Receipts Keep Chaining to Full_Body_Spec_Anthro_Driver_Capture.Json (The Pre-A2 Fitted Grip, Fitted to the Capture and Not a Grip Recommendation): On the Neutral Grip the Seeds Receipt Fails the Gates (IK Closure 3.9 -> 76.8 mm vs 5 mm, IK RMS 29.5 -> 55.3 mm, Backswing Min Weight Fraction 0.0 BW, Address-to-Impact 64 mm vs 34 mm), the A6 Problem Statement

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14551
- **Branch:** merged via #14644
- **PR:** #14644
- **Paths:** see #14644
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`1b514e12`; collated from changes/14551-neutral-address-grip-15-deg.md)
- **Summary:** Neutral address grip: flagship driver grip set to the HX-canonical grip (alpha -10.73, psi 40, gamma 0) so both hands read 15 deg face-on (was 98/108 deg); address and swing fixtures re-solved. Deferred to A6 (#14555): at address the RW wrist reads -41.8 deg (range -40..25) and the driver swing leaves the wrist ranges (RWInputY 691 frames to -130 deg, RWInputX 66 frames, LWInputY 216 frames); numbers in neutral_grip.json. Capture-matching receipts keep chaining to full_body_spec_anthro_driver_capture.json (the pre-A2 fitted grip, fitted to the capture and not a grip recommendation): on the neutral grip the seeds receipt fails the gates (IK closure 3.9 -> 76.8 mm vs 5 mm, IK RMS 29.5 -> 55.3 mm, backswing min weight fraction 0.0 BW, address-to-impact 64 mm vs 34 mm), the A6 problem statement
- **Next step:** Shipped in PR #14644.

### DL-#14648 · Putting Studio: Green Type Selector for Every Core Green Preset (Undulating Included), Height Grids Generated From GreenSpec Into Core_Greens.Js and Read With the Core Bilinear Lookup

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14648
- **Branch:** merged via #14676
- **PR:** #14676
- **Paths:** see #14676
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`42b69064`; collated from changes/14648-studio-greens.md)
- **Summary:** Putting studio: Green Type selector for every core green preset (undulating included), height grids generated from GreenSpec into core_greens.js and read with the core bilinear lookup
- **Next step:** Shipped in PR #14676.

### DL-#14660 · SG Optimizer: Common Random Numbers and Pin-Relative Aims so HoleMDP Value Iteration Converges (Cap Is a Safety Bound Only)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14660
- **Branch:** merged via #14663
- **PR:** #14663
- **Paths:** see #14663
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`6780ce16`; collated from changes/14660-sg-mdp-common-random-numbers.md)
- **Summary:** SG optimizer: common random numbers and pin-relative aims so HoleMDP value iteration converges (cap is a safety bound only)
- **Next step:** Shipped in PR #14663.

### DL-#14449 · Cross-Frontend Putting Golden: Shared-Core Fixture, React/Tauri Replay Parity Tests and a Generated Parity Report

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14449
- **Branch:** merged via #14666
- **PR:** #14666
- **Paths:** see #14666
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`300f194e`; collated from changes/14449-cross-frontend-putting-golden-shared-cor.md)
- **Summary:** Cross-frontend putting golden: shared-core fixture, React/Tauri replay parity tests and a generated parity report
- **Next step:** Shipped in PR #14666.

### DL-#14647 · Putting Studio: The Skid-to-Roll Transition Ends Where the Core Simulate_Strike Skid Ends for the Ball Speed, and the Telemetry Shows the Core Launch Angle

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14647
- **Branch:** merged via #14662
- **PR:** #14662, #14667
- **Paths:** see #14662
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`33a3f33f`; collated from changes/14647-studio-heel-toe-strike.md)
- **Summary:** Putting studio: the skid-to-roll transition ends where the core simulate_strike skid ends for the ball speed, and the telemetry shows the core launch angle
- **Next step:** Shipped in PR #14662.

### DL-#14600 · Head-CG Offset (Opt-In) and Toe_Down_m Shaft Metric for the C4 Shaft State (Refs #14600)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14600
- **Branch:** merged via #14658
- **PR:** #14658, #14674
- **Paths:** see #14658
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`7ecb8022`; collated from changes/14600-shaft-axis-from-chain.md)
- **Summary:** Head-CG offset (opt-in) and toe_down_m shaft metric for the C4 shaft state (Refs #14600)
- **Next step:** Shipped in PR #14658.

### DL-#14496 · Clarify the Fixed-State ZTCF Drift and ZVCF Control Component Definitions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14496
- **Branch:** merged via #14498
- **PR:** #14498
- **Paths:** see #14498
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`b9398531`; collated from changes/14496-clarify-the-fixed-state-ztcf-drift-and-z.md)
- **Summary:** Clarify the fixed-state ZTCF drift and ZVCF control component definitions
- **Next step:** Shipped in PR #14498.

### DL-#12688 · R9a: Commit Drake and Pinocchio MeshCat Appearance Stills and Evidence Test (#12688)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12688
- **Branch:** merged via #14395
- **PR:** #14395
- **Paths:** see #14395
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`ec12acb9`; collated from changes/12688-regenerate-matched-swing-ledger-after-r9.md)
- **Summary:** R9a: commit Drake and Pinocchio MeshCat appearance stills and evidence test (#12688)
- **Next step:** Shipped in PR #14395.

### DL-#14646 · Putting Studio: A Metric/Imperial Selector; Distance, Speed and Strike Labels Come From the Shared Units Table Generated Into Core_Units.Js

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14646
- **Branch:** merged via #14654
- **PR:** #14654, #14671
- **Paths:** see #14654
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`ea6d0a4d`; collated from changes/14646-units-api.md)
- **Summary:** Putting studio: a Metric/Imperial selector; distance, speed and strike labels come from the shared units table generated into core_units.js
- **Next step:** Shipped in PR #14654.

### DL-#14640 · Replace Wall-Clock Speed-Up Ratio in Test_Topology_Contact_Batched With a Deterministic Per-Pair Kernel Call Count

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14640
- **Branch:** merged via #14651
- **PR:** #14651
- **Paths:** see #14651
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`2c4bed1a`; collated from changes/14640-replace-wall-clock-speed-up-ratio-in-tes.md)
- **Summary:** Replace wall-clock speed-up ratio in test_topology_contact_batched with a deterministic per-pair kernel call count
- **Next step:** Shipped in PR #14651.

### DL-#14495 · Align MuJoCo Counterfactuals Zvcf() and the MATLAB ZVCF Comments With the Control-Component Contract

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14495
- **Branch:** merged via #14636
- **PR:** #14636
- **Paths:** see #14636
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`96bb31ae`; collated from changes/14495-align-mujoco-counterfactuals-zvcf-and-th.md)
- **Summary:** Align MuJoCo counterfactuals zvcf() and the MATLAB ZVCF comments with the control-component contract
- **Next step:** Shipped in PR #14636.

### DL-#14472 · React/Tauri Putting Page: Saved Scenarios and Comparison Over the Shared Core, Golden-Pinned (Part 1 Of #14472)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14472
- **Branch:** merged via #14637
- **PR:** #14637, #14656
- **Paths:** see #14637
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`b2c6da6c`; collated from changes/14472-react-tauri-putting-page-export-and-impo.md)
- **Summary:** React/Tauri putting page: saved scenarios and comparison over the shared core, golden-pinned (part 1 of #14472)
- **Next step:** Shipped in PR #14637.

### DL-#14630 · Putting: The Strokes Gained Chart Title Fits the Phone-Width Window and the Holed Check Box Reads on the Dark Controls Panel

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14630
- **Branch:** merged via #14645
- **PR:** #14645
- **Paths:** see #14645
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`43ee052d`; collated from changes/14630-sg-panel-phone-polish.md)
- **Summary:** Putting: the Strokes Gained chart title fits the phone-width window and the Holed check box reads on the dark controls panel
- **Next step:** Shipped in PR #14645.

### DL-#14570 · SHOW-B6: Slope/Stance Bins and Wind Bins in the Dispersion Tables (slope_wind_bins.py), Labelled Assumptions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14570
- **Branch:** merged via #14628
- **PR:** #14628
- **Paths:** see #14628
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`1558c825`; collated from changes/14570-show-b6-slope-stance-bins-and-wind-bins.md)
- **Summary:** SHOW-B6: slope/stance bins and wind bins in the dispersion tables (slope_wind_bins.py), labelled assumptions
- **Next step:** Shipped in PR #14628.

### DL-#14563 · SHOW-C7 Flight Report and Measured-Swing Flight Card (Part Of #14563)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14563
- **Branch:** merged via #14620
- **PR:** #14620, #14670
- **Paths:** see #14620
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`a3a1276e`; collated from changes/14563-c7-slice-2-address-pose-stills-capture-f.md)
- **Summary:** SHOW-C7 flight report and measured-swing flight card (Part of #14563)
- **Next step:** Shipped in PR #14620.

### DL-#14632 · Drake IK Now Implements the Club-Sole Ground Term (Base Was a No-Op), Fixing the 126.7 mm Address Hover

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14632
- **Branch:** merged via #14634
- **PR:** #14634
- **Paths:** see #14634
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`d58c3d44`; collated from changes/14632-drake-ik-now-implements-the-club-sole-gr.md)
- **Summary:** Drake IK now implements the club-sole ground term (base was a no-op), fixing the 126.7 mm address hover
- **Next step:** Shipped in PR #14634.

### DL-#14568 · SHOW-B4: Physics Backend on Dispersion Tables, EB Shrinkage, Personalised vs Tour Strategy

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14568
- **Branch:** merged via #14623
- **PR:** #14623
- **Paths:** see #14623
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`9b8b8625`; collated from changes/14568-show-b4-physics-backend-on-dispersion-ta.md)
- **Summary:** SHOW-B4: physics backend on dispersion tables, EB shrinkage, personalised vs tour strategy
- **Next step:** Shipped in PR #14623.

### DL-#14617 · Putting Simulator: The Counter-Torque Panel Bounds the Gust Input by the Core's GUST_RANGE_NM Instead of the Preload Range

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14617
- **Branch:** merged via #14626
- **PR:** #14626
- **Paths:** see #14626
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`1a513ea5`; collated from changes/14617-torque-gust-spin-range.md)
- **Summary:** Putting simulator: the counter-torque panel bounds the gust input by the core's GUST_RANGE_NM instead of the preload range
- **Next step:** Shipped in PR #14626.

### DL-#14451 · Add Npm Run Tauri:Artifact: A Local Tauri Installer Named With Version and Commit, Plus a Sha256 File, Never Published

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14451
- **Branch:** merged via #14624
- **PR:** #14624
- **Paths:** see #14624
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`84d828e2`; collated from changes/14451-add-npm-run-tauri-artifact-a-local-tauri.md)
- **Summary:** Add npm run tauri:artifact: a local Tauri installer named with version and commit, plus a sha256 file, never published
- **Next step:** Shipped in PR #14624.

### DL-#14562 · SHOW-C6 Shot-Shape Presets, Ball-Flight Curvature Sign Tests and Swing-Variant Artefact With Run Identity

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14562
- **Branch:** merged via #14618
- **PR:** #14618, #14686
- **Paths:** see #14618
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`46a51da5`; collated from changes/14562-name-shot-presets-by-face-to-path-d30-is.md)
- **Summary:** SHOW-C6 shot-shape presets, ball-flight curvature sign tests and swing-variant artefact with run identity
- **Next step:** Shipped in PR #14618.

### DL-#14471 · Putting Analytics: Versioned Setup Document (The Studio's Schema_Version 4.0.0) With Export/Import (Studio Exports Load Their Green and Distance), Named Scenarios With Their Dispersion Result and a Comparison Against the First, as a Shared Service and /Scenario, /Scenario-Compare, /Setup-Export and /Setup-Import Routes

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14471
- **Branch:** merged via #14615
- **PR:** #14615, #14629
- **Paths:** see #14615
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`392a57d0`; collated from changes/14471-putting-scenarios-desktop.md)
- **Summary:** Putting analytics: versioned setup document (the studio's schema_version 4.0.0) with export/import (studio exports load their green and distance), named scenarios with their dispersion result and a comparison against the first, as a shared service and /scenario, /scenario-compare, /setup-export and /setup-import routes
- **Next step:** Shipped in PR #14615.

### DL-#14571 · SHOW-B7: Fit a SkillFamily From Launch-Monitor Shot Data, Shrunk Toward a Preset

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14571
- **Branch:** merged via #14616
- **PR:** #14616
- **Paths:** see #14616
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`f2deca76`; collated from changes/14571-show-b7-fit-a-skillfamily-from-launch-mo.md)
- **Summary:** SHOW-B7: fit a SkillFamily from launch-monitor shot data, shrunk toward a preset
- **Next step:** Shipped in PR #14616.

### DL-#14569 · SHOW-B5: Parametric Lie/Rough Impact-Condition Modifier (Physics/lie_interaction.py) With Labelled Assumptions

- **State:** shipped
- **Owner:** fleet-orchestrator
- **Issue:** #14569
- **Branch:** desk/show-neck/14569-lie-interaction
- **PR:** #14613, #14852
- **Paths:** see #14613
- **Started:** 2026-11-05
- **Last verified:** 2026-11-07 (`684daf6b`; collated from changes/14569-b5-the-lie-spin-factor-now-reaches-the-b.md)
- **Summary:** SHOW-B5: parametric lie/rough impact-condition modifier (physics/lie_interaction.py) with labelled assumptions
- **Next step:** Shipped in PR #14613.

### DL-#14470 · React Putting Simulator Gains the Counter-Torque Stabilization Explorer (Sliders, Putter Preset, Hand-Couple Chart, Impact Readouts) Over the Core /Counter-Torque Route

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14470
- **Branch:** merged via #14611
- **PR:** #14611
- **Paths:** see #14611
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`162c2aef`; collated from changes/14470-react-putting-simulator-gains-the-counte.md)
- **Summary:** React putting simulator gains the counter-torque stabilization explorer (sliders, putter preset, hand-couple chart, impact readouts) over the core /counter-torque route
- **Next step:** Shipped in PR #14611.

### DL-#14567 · SHOW-B3: Cached, Versioned MC Dispersion Tables (CRN) + Pipeline Landing Adapter

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14567
- **Branch:** merged via #14608
- **PR:** #14608
- **Paths:** see #14608
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`25f3520a`; collated from changes/14567-show-b3-cached-versioned-mc-dispersion-t.md)
- **Summary:** SHOW-B3: cached, versioned MC dispersion tables (CRN) + pipeline landing adapter
- **Next step:** Shipped in PR #14608.

### DL-#14561 · SHOW-C5 Speed Continuation 100-125 Mph With Pareto Front and Binding Report

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14561
- **Branch:** merged via #14606
- **PR:** #14606, #14673, #14681, #14721, #14722
- **Paths:** see #14606
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`c64cb10c`; collated from changes/14561-show-c5-ladder-binding-report-classes-ev.md)
- **Summary:** SHOW-C5 speed continuation 100-125 mph with Pareto front and binding report
- **Next step:** Shipped in PR #14606.

### DL-#14550 · Add Golf-Language Face-On Address Grip Roll Metric (0 Deg Vertical, + Strong) With Flagship Measurement and Overlay Render

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14550
- **Branch:** merged via #14601
- **PR:** #14601
- **Paths:** see #14601
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`986b3224`; collated from changes/14550-add-golf-language-face-on-address-grip-r.md)
- **Summary:** Add golf-language face-on address grip roll metric (0 deg vertical, + strong) with flagship measurement and overlay render
- **Next step:** Shipped in PR #14601.

### DL-#14560 · SHOW-C4: Shaft_State_Series Reports MuJoCo Chain-Shaft Tip Deflection, Lead/Droop, Deflection Rate and Strain Energy Through Impact, With R11-Window Evidence on the Best-Fit Driver

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14560
- **Branch:** merged via #14598
- **PR:** #14598
- **Paths:** see #14598
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`1760fe35`; collated from changes/14560-show-c4-shaft-state-series-reports-mujoc.md)
- **Summary:** SHOW-C4: shaft_state_series reports MuJoCo chain-shaft tip deflection, lead/droop, deflection rate and strain energy through impact, with R11-window evidence on the best-fit driver
- **Next step:** Shipped in PR #14598.

### DL-#14554 · SHOW-A5: Trunk Observability Report (Marker Jacobian SVD), ADR-0059 and Coupled-by-Default Showpiece Spec

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14554
- **Branch:** merged via #14603
- **PR:** #14603
- **Paths:** see #14603
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`513127ec`; collated from changes/14554-show-a5-trunk-observability-report-marke.md)
- **Summary:** SHOW-A5: trunk observability report (marker Jacobian SVD), ADR-0059 and coupled-by-default showpiece spec
- **Next step:** Shipped in PR #14603.

### DL-#14553 · SHOW-A4: Cited Neck ROM (neck_rom.py, Single Source for Range_of_Motion) Enforced as MJCF Joint Limits on NeckInput X/Y/Z

- **State:** shipped
- **Owner:** fleet-orchestrator
- **Issue:** #14553
- **Branch:** desk/show-neck/14553-neck-segment
- **PR:** #14578
- **Paths:** see #14578
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`00e1e9ff`; collated from changes/14553-show-a4-neck-rom.md)
- **Summary:** SHOW-A4: cited neck ROM (neck_rom.py, single source for range_of_motion) enforced as MJCF joint limits on NeckInput X/Y/Z
- **Next step:** Shipped in PR #14578.

### DL-#14595 · CI Standard Sets PIP_RETRIES=10 and PIP_DEFAULT_TIMEOUT=60 Workflow-Wide and Passes Pip-Audit --Timeout 60, so Transient PyPI Read Timeouts No Longer Fail CI

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14595
- **Branch:** merged via #14597
- **PR:** #14597
- **Paths:** see #14597
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`807605a6`; collated from changes/14595-ci-standard-sets-pip-retries-10-and-pip.md)
- **Summary:** CI Standard sets PIP_RETRIES=10 and PIP_DEFAULT_TIMEOUT=60 workflow-wide and passes pip-audit --timeout 60, so transient PyPI read timeouts no longer fail CI
- **Next step:** Shipped in PR #14597.

### DL-#14566 · SHOW-B2: SkillFamily Schema (Impact-Condition Mean/Cov + Provenance), Assumption-Marked Presets, Registry, PlayerProfile Link

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14566
- **Branch:** merged via #14590
- **PR:** #14590
- **Paths:** see #14590
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`8e6506c8`; collated from changes/14566-show-b2-skillfamily-schema-impact-condit.md)
- **Summary:** SHOW-B2: SkillFamily schema (impact-condition mean/cov + provenance), assumption-marked presets, registry, PlayerProfile link
- **Next step:** Shipped in PR #14590.

### DL-#14530 · Putting Green Simulator: The Ud.Putt_Scene/1 Export Writes and Reads Impact_Toe_m (Optional on Read, Default 0.0, Range-Checked Against IMPACT_TOE_RANGE_M), so a Toe or Heel Strike Round-Trips

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14530
- **Branch:** merged via #14594
- **PR:** #14594
- **Paths:** see #14594
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`eaa7b7ac`; collated from changes/14530-scene-export-impact-toe.md)
- **Summary:** Putting Green Simulator: the ud.putt_scene/1 export writes and reads impact_toe_m (optional on read, default 0.0, range-checked against IMPACT_TOE_RANGE_M), so a toe or heel strike round-trips
- **Next step:** Shipped in PR #14594.

### DL-#14450 · React Web UI: One Command (Npm Run Local) Builds From the Lockfile and Serves It Through the API From a Clean Checkout; CI Build Check Runs the Same Clean-Build Check; README Run Steps; a Test Bars Paid-API Hosts and Keys

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14450
- **Branch:** merged via #14593
- **PR:** #14593
- **Paths:** see #14593
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`69ae6718`; collated from changes/14450-react-web-ui-one-command-npm-run-local-b.md)
- **Summary:** React web UI: one command (npm run local) builds from the lockfile and serves it through the API from a clean checkout; CI Build Check runs the same clean-build check; README run steps; a test bars paid-API hosts and keys
- **Next step:** Shipped in PR #14593.

### DL-#14559 · SHOW-C3a: SplineTorqueControl Piecewise-Linear Torque Parameterisation

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14559
- **Branch:** merged via #14589
- **PR:** #14589, #14596, #14625, #14638, #14653, #14655, #14680, #14685, #14693
- **Paths:** see #14589
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`d865f3c3`; collated from changes/14559-show-c3c-d32-assumption-limited-vs-proba.md)
- **Summary:** SHOW-C3a: SplineTorqueControl piecewise-linear torque parameterisation
- **Next step:** Shipped in PR #14589.

### DL-#14552 · SHOW-A3: Optional Lumbar:Thoracic Regional-Split Coupling (Trunk_Coupling, Off by Default) for the Full-Body-V2 Trunk, MuJoCo Joint Equalities, and Trunk Joint-Tree/Mass/ROM Contract Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14552
- **Branch:** merged via #14577
- **PR:** #14577
- **Paths:** see #14577
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`0cecc8f7`; collated from changes/14552-show-a3-optional-lumbar-thoracic-regiona.md)
- **Summary:** SHOW-A3: optional lumbar:thoracic regional-split coupling (trunk_coupling, off by default) for the full-body-v2 trunk, MuJoCo joint equalities, and trunk joint-tree/mass/ROM contract tests
- **Next step:** Shipped in PR #14577.

### DL-#14466 · Putting API: POST /Solve-Aim-3D Solves the Pace and Line That Hole a /Simulate-3D Putt on That Route's Own Surface and Head (Core Solve_Aim, Like the Desktop Auto-Solve); the 3-D Request Models Move to Src/Api/Models/putting_green_requests.py

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14466
- **Branch:** merged via #14588
- **PR:** #14588, #14604
- **Paths:** see #14588
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`416c462c`; collated from changes/14466-react-putting-simulator-gains-the-deskto.md)
- **Summary:** Putting API: POST /solve-aim-3d solves the pace and line that hole a /simulate-3d putt on that route's own surface and head (core solve_aim, like the desktop Auto-Solve); the 3-D request models move to src/api/models/putting_green_requests.py
- **Next step:** Shipped in PR #14588.

### DL-#14441 · WSCG Update Manuscript Draft on the Best-Match MuJoCo ZTCF/ZVCF Run, Citing the Pinned Run_Id

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14441
- **Branch:** merged via #14584
- **PR:** #14584
- **Paths:** see #14584
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`c82e3803`; collated from changes/14441-wscg-update-manuscript-draft-on-the-best.md)
- **Summary:** WSCG update manuscript draft on the best-match MuJoCo ZTCF/ZVCF run, citing the pinned run_id
- **Next step:** Shipped in PR #14584.

### DL-#14565 · SHOW-B1: ShotOutcomeModel Protocol + Backend Registry; Tour Baseline Is the Default Quick_Sg Backend

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14565
- **Branch:** merged via #14581
- **PR:** #14581
- **Paths:** see #14581
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`950bedd8`; collated from changes/14565-show-b1-shotoutcomemodel-protocol-backen.md)
- **Summary:** SHOW-B1: ShotOutcomeModel protocol + backend registry; tour baseline is the default quick_sg backend
- **Next step:** Shipped in PR #14581.

### DL-#14558 · SHOW-C2: ImpactConditions + Evaluate_Rollout Facade for Optimiser Loops (Face, Path, AoA, Speed, Strike) Over Impact_Parameters

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14558
- **Branch:** merged via #14579
- **PR:** #14579
- **Paths:** see #14579
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`ddf355e4`; collated from changes/14558-show-c2-impactconditions-evaluate-rollou.md)
- **Summary:** SHOW-C2: ImpactConditions + evaluate_rollout facade for optimiser loops (face, path, AoA, speed, strike) over impact_parameters
- **Next step:** Shipped in PR #14579.

### DL-#14557 · SHOW-C1: ImpactTarget, TrustRegion and PhysiologicalEnvelope Contracts (Impact_Optimization Package; Face:Path 10:1; NSX-04 Torque Limits; Cited Velocity YAML)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14557
- **Branch:** merged via #14580
- **PR:** #14580, #14707
- **Paths:** see #14580
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`3e6b8f9b`; collated from changes/14557-source-skilled-male-peak-angular-velocit.md)
- **Summary:** SHOW-C1: ImpactTarget, TrustRegion and PhysiologicalEnvelope contracts (impact_optimization package; face:path 10:1; NSX-04 torque limits; cited velocity YAML)
- **Next step:** Shipped in PR #14580.

### DL-#14464 · Putting API: /Simulate-3D Accepts the Core Green Shape (Green: GreenShapeRequest, a Preset or Slope, Tier and Bumps) and Rolls on Its GreenSpec Height Field, Reporting That Spec's Grade and Aspect, and Strikes Along Aim_Heading_Deg (Was Always +X), so the React Page Can Send the Core Presets and an Aimed Putt

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14464
- **Branch:** merged via #14583
- **PR:** #14583, #14586
- **Paths:** see #14583
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`f2db11cc`; collated from changes/14464-react-putting-page-green-type-selector-o.md)
- **Summary:** Putting API: /simulate-3d accepts the core green shape (green: GreenShapeRequest, a preset or slope, tier and bumps) and rolls on its GreenSpec height field, reporting that spec's grade and aspect, and strikes along aim_heading_deg (was always +x), so the React page can send the core presets and an aimed putt
- **Next step:** Shipped in PR #14583.

### DL-#14469 · Putting Dynamics: Counter-Torque Stabilization Model (Two-Hand Couples, Off-Plane Preload, Noise, Gust, Lie, Putter) in the Shared Core, Reading the Shared Putter Presets, Drift-Gated Against the Studio

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14469
- **Branch:** merged via #14548
- **PR:** #14548, #14602, #14609
- **Paths:** see #14548
- **Started:** 2026-11-04
- **Last verified:** 2026-11-05 (`c39d2c75`; collated from changes/14469-putting-counter-torque-desktop.md)
- **Summary:** Putting dynamics: counter-torque stabilization model (two-hand couples, off-plane preload, noise, gust, lie, putter) in the shared core, reading the shared putter presets, drift-gated against the studio
- **Next step:** Shipped in PR #14548.

### DL-#13560 · MuJoCo Grip Samples Read Zero-Slip Pairs Frictionless, Matching the Shared Law's Zero Friction at Rest; the Ten_Finger Hold Net_Moment_Com Xfail Is Removed

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13560
- **Branch:** merged via #14185
- **PR:** #14185, #14351, #14354, #14711, #14902
- **Paths:** see #14185
- **Started:** 2026-11-01
- **Last verified:** 2026-11-08 (`79dc26d9`; collated from changes/13560-grip-contact-preload-parity.md)
- **Summary:** MuJoCo grip samples read zero-slip pairs frictionless, matching the shared law's zero friction at rest; the ten_finger hold net_moment_com xfail is removed
- **Next step:** Shipped in PR #14185.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.

## Field Reference

| Field           | Required                   | Notes                                                          |
| --------------- | -------------------------- | -------------------------------------------------------------- |
| `State`         | Always                     | One of the six states above                                    |
| `Owner`         | Always                     | Agent id from the fleet roster, or `unassigned`                |
| `Issue`         | While live                 | Governing GitHub issue; enforces the entry/issue join          |
| `Branch`        | `in_progress`, `in_review` | Enforces the entry/branch join                                 |
| `PR`            | Always                     | Number and state, or `not created`                             |
| `Paths`         | Always                     | Globs; drives silent-entry detection                           |
| `Started`       | Always                     | Drives cycle time                                              |
| `Last verified` | Always                     | Date plus SHA — the liveness signal                            |
| `Summary`       | Always                     | One or two sentences                                           |
| `Next step`     | While live                 | Exactly one action; if it needs two sentences, split the entry |
| `Parked`        | When `parked`              | Date plus reason                                               |

Never place credentials, tokens, or customer data in a development log.
No material development-log change — Bolt `np.linalg.norm` → `einsum` consolidation (#11073, #11074, #11076) is a behaviour-preserving micro-optimisation with no feature entry.
No material development-log change — Bolt `np.linalg.norm` → `sqrt(einsum)`/`math.sqrt(np.vdot)` consolidation (#11112, #11128, #11129) is a behaviour-preserving micro-optimisation with no feature entry.
No material development-log change — PyJWT floor/lock bump to 2.14.0 for OSV GHSA-w6j9-cwv2-h6wq (#11153) is a dependency-only change with no feature entry.
No material development-log change — urllib3 (2.8.0, CVE-2026-97687) and PyJWT (2.15.0, CVE-2026-101918) floor/lock bumps (#11191) are dependency-only changes with no feature entry.
No material development-log change — Bolt np.linalg.norm → sqrt(einsum) in rigidity.py (#11189) is a behaviour-preserving micro-optimisation with no feature entry.

## DL-#11357 — Pure Raw Shaft Fragment Derivative Diagnostics

- **Change:** Canonical immutable options/check/result contracts and public
  raw fragment analytic/central FD diagnostics; canonical camera/marker reuse.
- **Validation:** Repository fail-first scaffold regression followed by focused
  native-free synthetic tests, curated exports and fingerprint inclusion.
- **Limits:** No historical native/Library/render/fit operation, no objective
  or evidence-role change, and no clinical/scientific qualification.
- **Next Step:** Review typed core, then separately design authenticated
  workspace consumers using existing job/store/renderer boundaries.

## Authored Replay Impact Application Checkpoint — #11464

The normal React journey now previews impact from a verified authored replay,
with exact geometry/sample/flight declarations, bounded native execution,
cancellation, persisted run-ID recall and authenticated four-file download.
Saved results display their own assumptions and metrics. Canonical receipt and
trajectory owners remain unchanged; source/runtime/parent and artifact checks
fail closed. Research acceptance remains rejected, with authored seconds and
false physical/scientific qualification.

Root final validation: 125 focused Python cases pass with one Windows symlink
capability skip; one additional opt-in native case passes with actual MuJoCo,
clean process transport, impact solver and Rust flight (no substitutions).
The React cohort passes 104 cases; configured six-owner mypy, scoped lint and
format checks pass. Earlier failures and immutable agent freezes are retained
in the application review. Four-page LaTeX Application Supplement 1 is compiled
without final layout warnings and all pages visually reviewed; editable source,
PDF and the explicitly synthetic native bundle are saved on the local Desktop.
Earlier Methods V8 and the five-page extraction report remain unchanged.

See [Application Procedure](necromatcher-replay-impact.md) and its linked review/report. Tiger stays
within original frames 0–190 ([0,191)); Hogan stays [0,750). No historical
Library mutation, qualified match or new fleet-native-host delivery is established.
Remaining goal work includes improved historical matching, authored dynamics,
simulation golf integration and desktop parity. The full goal remains active.

## Native Desktop Replay Impact Checkpoint — #11487

The native Necromatcher tile now imports registered authored replays, runs the
shared bounded replay-impact session off the GUI thread, cancels owned jobs,
recalls retained run IDs after restart and exports an authenticated four-file
ZIP exclusively. Saved assumptions remain separate from editable declarations.
The public Shot Tracer handoff retains exact detached positions and provenance.

Root validation passes 146 focused Qt/viewer/parity cases without skips and
configured mypy on three production owners. One additional opt-in actual
MuJoCo/HDF5/clean-worker/impact/Rust native Qt case passes with Python exit 0.
The three final synthetic screenshots are readable and visually reviewed;
retained positions are verified, while OpenGL pixel accuracy is unverified.
Test-only font bootstrap uses existing Windows Segoe UI when the offscreen
font database is empty. The 120-second whole-test cap does not change the
60-second native budget; earlier timeout evidence is preserved.

The three-page Native Desktop Supplement 1 and editable LaTeX, screenshots and
intact synthetic bundle are saved on the local Desktop. See [Procedure](necromatcher-replay-impact.md)
and [Frozen Review](historical_capture/native-desktop-replay-impact-review.json). Tiger remains [0,191), Hogan [0,750).
Historical fits, scientific qualification, simulation golf and final overall
desktop parity acceptance remain open. No new fleet-native-host delivery or remote
CI-green claim is made. The full goal remains active.

## Local Research Admission Race Follow-Up — #11493

A concurrent request regression reproduced reused shot-ID admission after another
request prepared and cancelled during authentication. The API now applies the
same identity guard before and immediately after authentication. All 10 API
cases pass; a fresh additional actual MuJoCo/clean-worker/Rust API and Qt
acceptance passes in 24.56 s, with seven exact retained samples and no
re-simulation on recall. Configured mypy passes for the changed API owner.
The earlier 209-case cohort and 106 React cases remain separate checkpoint
evidence. The follow-up preserves unverified qualification and all historical
limits; the full goal remains active.

See [Follow-Up Review](historical_capture/local-research-golf-admission-race-review.json). Source inventory covers 6,697 src/test
files, with equal before/after maps; this is not a whole-repository baseline.

## User-Requested Handoff and Stop Checkpoint

The user requested a committed checkpoint and instructed this agent to stop
pursuing the goal after turnover. See [Next-Agent Turnover](necromatcher-next-agent-turnover.md) for the
exact implementation state at f0e14944, scoped work packets, public owners,
validation commands, historical evidence boundaries and remaining acceptance.
No proposed optimized video export or new fit was started. This documentation
checkpoint follows the published local research integration; the full goal is
unfinished and is to be paused after normal commit/push and handoff publication.

## Capture Registry Parent Integration - #11162 / #11172

Accepted-main synchronization preserves capture-registry, export and swing-comparison source changes. Record conflicts retain distinct entries from both branches. All 133 ordered matched-swing ledger rows and non-timestamp metadata are identical; retain accepted-main timestamp. Regenerate the monolith register from merged source. Scientific matching/dynamics acceptance remains separate.

Record validation: all five monolith-register tests pass; no duplicate primary headings or conflict markers; all 133 ledger rows and non-timestamp metadata preserved exactly; registry/export/comparison source unchanged. Broader capture tests are pending pinned Tools initialization; first collection attempt stopped on the missing submodule. Protected parent CI and full scoped rerun remain required before merge.

## Capture Comparison DRY Gate Remediation - #11162 / #11172

CI identified duplicated club endpoint selection in angular-speed and wrist metrics. A shared private helper preserves explicit endpoint precedence, measured-marker aliases, array identity and missing-endpoint/shaft-axis behavior. AGY Gemini Flash supplied the bounded extraction; root reviewed the exact diff. Existing focused regression suite passes 190 tests with 15 unavailable-private-workbook skips, zero errors/failures, natural exit 0. Ruff format/check and full DRY duplication gate pass without changing baselines. Regenerate the monolith register for the revised file length. Current-head protected CI remains required.

## Capture Leaderboard Suite Classification - #11162 / #11172

Protected CI verified the DRY extraction, then identified 14 net-new unmarked tests in TestPerCaptureLeaderboard. Mark this pure-Python class as unit; all 14 execute and pass under unit selection with no skips/errors. Full suite-marker ratchet passes without baseline changes. Broader focused capture/comparison qualification remains 190 passed and 15 unavailable-private-workbook skips. Protected current-head CI remains required before merge.

## Capture Comparison Inventory Integration - #11162 / #11172

CI at bd1ee9a8e passes repository structure and 20,546 unit cases; its only unit failure identifies five new swing_comparison paths missing from the shared-tools divergence inventory. Regenerate JSON and Markdown with the authoritative generator against the pinned Tools tree. All five paths classify as UD-only; existing overlap totals remain unchanged. All ten inventory tests pass with no errors/skips, including the committed-tree freshness test, and generator --check succeeds. Protected current-head CI remains required before merging.

### Selected Club Continuity Update (2026-10-03)

PR #11256 merged at `7db86ff173653a235ea63bb4af347af0a8a5ab83`. The eight verified Human ellipsoid MP4s remain on both Desktops in `Simscape_Matches_20261003_Club_Refined_1080p`; these are IK reconstructions. The source reference now presents this current selection before historical experiments.

The selected club fits were tested at all anchors and midpoints using independent native position, velocity and fresh 91-state checks. Sequential seeding accepted 1,306/1,307 tour states and 732/733 owner states. A targeted diagnostic recovered the owner's rejected original anchor from its saved closed-arm seed; the previous accepted pose reproduced an alternate branch. A complete anchor-seeded rerun exited naturally at 20:31:23 UTC and accepted all 733 owner states (367 anchors and 366 midpoints), while the tour still rejected midpoint 998 at 1.3847222 s. All original anchors matched within 1.73e-15 physical residual. The peak accepted dependent angular rates remain approximately 1,639 and 4,118 deg/s; sampled-state acceptance does not qualify between-sample branch continuity, dependent acceleration, anatomy, contact or forward dynamics.

The private production integration candidate passed 17 native pre-model contract cases (parent RED, candidate GREEN, 51 recorded checks) at 20:20:58 UTC. Numerical whole-output parity and positive-fit validation remain separate pending gates. Original failed runner receipts are preserved. A custom vector measurement block built successfully, but the eight-axis instrumented simulation hit the Home license's 1,000-nonvirtual-block limit. No all-35 measured torque or moving feedback-off replay claim follows from these tests. Original physical model files were not saved.

See the separate editable research reference and the extended aggregate `tangent_c2_checkpoint_20261003.json` for scope and provenance. The built-in LaTeX compiler still fails with `Unable to find standard directories for platform`; PDF compilation/page review are unverified. Canonical calculation inventory and governed manual release remain blocked; this update grants neither a release exemption nor scientific approval.

## F05d Native Manifold BoxFDDP Candidate (#11932; 2026-10-09)

A test-first MuJoCo 3.8/Crocoddyl 3.2.1 child now solves the same
contact-free floating-root/two-hinge physical state with a 16-dimensional
configuration/velocity tangent, exact native Euler action derivative,
bounded direct motor input and nonlinear native fallback admission. The
paired source-hashed receipt measures BoxFDDP against SciPy native shooting
under identical reference, truth, input box, horizon, iteration and
cooperative wall budget, alternates solver order and charges a common
Tools T01 bootstrap to both. Every applied input independently replays from
complete native integration state. This is synthetic controller evidence,
not a hard deadline, measured capture fit or full-body qualification;
the F05 parent and canonical manual publication remain blocked. Read
chapter 26 and `F05D_NATIVE_MANIFOLD_BOX_FDDP_TURNOVER.md`.

## Bounded Nonuniform Moco Refinement Plan (#12244; 2026-10-10)

The 20-cell synthetic Moco parent inputs are retained byte-for-byte and the
offline activation report reproduces from those committed source, request,
solution, control and replay-audit artifacts. A pure source-bound planner marks
the observed full Hermite-Simpson cells, caps one child pass at four marked
cells and 24 intervals, and rejects evidence/hash or grid mismatches. The worker
uses the normalized custom mesh and parent state seed, then linearly samples
the persisted controls at every child endpoint and midpoint without control
trajectory resampling. Pure planner/interpolation regressions pass. No native
child solve or replay was run; solver API/time-grid acceptance and all physical
qualification remain open. The context catalog could not be refreshed here:
its `vendor/ud-tools` junction escapes this checkout to the frozen
`feedback-drake-native-11838` donor, and the context CLI correctly rejects that
path. See the #12244 change fragment, chapter 13 and F07 ladder turnover.

The follow-up recomputes and compares the full plan before solver setup and
builds the post-solve mesh audit from the seeded grid in the child request.
Pure JSON-reload and tampering regressions pass; there is still no native child
solve or replay evidence.

The next correction uses the OpenSim 4.6-compatible states-table insertion
signature and restores exact declared initial values after interpolation; a
strict fake-SWIG test covers the behavior without running the native solver.

An installed OpenSim 4.6 API smoke also passed for the committed parent STOs:
the states table inserted into a native 43-row `MocoTrajectory`, controls were
PWL-seeded, and cold initial values matched exactly. It ran under the 30-second
owned-tree supervisor with 140 MB sampled peak, and called no model init, solve,
or replay. A first malformed empty-trajectory smoke crashed in a SWIG setter;
the corrected data-bearing constructor passed. Actual solver acceptance remains
unverified.
