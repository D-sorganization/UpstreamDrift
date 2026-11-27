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

### DL-#15836 · Deliver Five Historical Golfer Model Bundles and Catalog-Driven Batch Previews

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15836
- **Branch:** feat/15836-legends-delivery
- **PR:** #15841
- **Paths:** scripts/export_player_models.py,assets/player_models/legends,tests/unit/model_appearance/test_player_export_selection.py
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`b1084002`; collated from changes/15836-deliver-five-historical-golfer-model-bun.md)
- **Summary:** Deliver five historical golfer model bundles and catalog-driven batch previews
- **Next step:** Merge the verified delivery through required checks and the merge queue.

### DL-#15835 · Acquire and Document Private Reference Clips for Five Historical Golf Legends

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15835
- **Branch:** docs/15835-legends-reference-videos
- **PR:** #15839
- **Paths:** docs/development/historical_capture/legend-reference-acquisition-2026-11-26.json,docs/development/historical_capture/legend-reference-acquisition-commands-2026-11-26.md,docs/development/historical-capture-procedure.md,AGENT_HANDOFF.md
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`2863ed86`; collated from changes/15835-legends-reference-footage-acquisition.md)
- **Summary:** Acquire and document private reference clips for five historical golf legends
- **Next step:** Review source identity, time, and rights limitations before any motion analysis.

### DL-#15821 · Calibrate Native Player Stature With Bounded Torso Scaling

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15821
- **Branch:** feat/15821-player-stature
- **PR:** #15823
- **Paths:** see #15823
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`1c6d2171`; collated from changes/15821-calibrate-native-player-stature-with-bou.md)
- **Summary:** Calibrate native player stature with bounded torso scaling
- **Next step:** Integrate calibrated construction into player bundles under #15816

### DL-#15819 · Deliver Portable Player Models and Hash-Bound Native Review Views

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15819
- **Branch:** feat/15819-player-delivery
- **PR:** #15831
- **Paths:** see #15831
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`1892e0bd`; collated from changes/15819-deliver-portable-player-models-and-hash.md)
- **Summary:** Deliver portable player models and hash-bound native review views
- **Next step:** Import candidates into Necromatcher and refit historical motion separately

### DL-#15816 · Build Source-Attributed Tiger Woods and Ben Hogan Skins and Current-Model Candidates

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15816
- **Branch:** feat/15816-player-model-bundles
- **PR:** #15828
- **Paths:** see #15828
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`91f99d7b`; collated from changes/15816-build-source-attributed-tiger-woods-and.md)
- **Summary:** Build source-attributed Tiger Woods and Ben Hogan skins and current-model candidates
- **Next step:** Review native player previews and reference media under #15814

### DL-#15815 · Add Bounded Visual Head Morphology Without Changing Golfer Physics

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15815
- **Branch:** feat/necromatcher-player-skins
- **PR:** #15818
- **Paths:** see #15818
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`306f7e57`; collated from changes/15815-add-bounded-visual-head-morphology-witho.md)
- **Summary:** Add bounded visual head morphology without changing golfer physics
- **Next step:** Review player-specific presets and native renders under #15814

### DL-#15767 · Fail Closed on Simulation WebSocket Auth, the LocalUser ADMIN Bypass, and Legacy REST Prefixes

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15767
- **Branch:** cursor/simulation-ws-auth-15767-8cd0
- **PR:** #15807
- **Paths:** see #15807
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`1524457d`; collated from changes/15767-fail-closed-on-simulation-websocket-auth.md)
- **Summary:** Fail closed on simulation WebSocket auth, the LocalUser ADMIN bypass, and legacy REST prefixes
- **Next step:** Merge the ready PR after CI is green on this head.

### DL-#15765 · CI Standard Runs Workflow Pushes, Code-Quality Failures Fail the Job, and OSV Crashes Are No Longer Swallowed

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15765
- **Branch:** cursor/ci-failure-propagation-15765-6aac
- **PR:** #15806
- **Paths:** see #15806
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`32852d70`; collated from changes/15765-ci-standard-runs-workflow-pushes-code-qu.md)
- **Summary:** CI Standard runs workflow pushes, code-quality failures fail the job, and OSV crashes are no longer swallowed
- **Next step:** Merge after CI is green on the head. Do not arm.

### DL-#15764 · Keep Advertised Engines Out of Unsupported_Engines When a Qualification Receipt Is Missing

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15764
- **Branch:** cursor/fix-engine-capability-overlap-ab90
- **PR:** #15804
- **Paths:** see #15804
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`11aca7ba`; collated from changes/15764-keep-advertised-engines-out-of-unsupport.md)
- **Summary:** Keep advertised engines out of unsupported_engines when a qualification receipt is missing
- **Next step:** CI green on the head; do not arm this lane.

### DL-#15762 · Install.Sh Runs Only After a Full Parse, and Tagged Releases Publish Its SHA-256 Checksum Instead of an Unpinned Curl-to-Bash From Main

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15762
- **Branch:** cursor/verified-install-sh-15762-d56f
- **PR:** #15803, #15782
- **Paths:** see #15803
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`fb6e2c9a`; collated from changes/15762-verified-rustup-init.md)
- **Summary:** install.sh runs only after a full parse, and tagged releases publish its SHA-256 checksum instead of an unpinned curl-to-bash from main
- **Next step:** Merge the checksum-verified install.sh pull request.

### DL-#15732 · Fail-Closed Manifest.Json Verifier, Resume Key and Atomic Publish for Swing Report Packages (SRP-6 Slice 1)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15732
- **Branch:** feat/srp6-manifest-verify-15732
- **PR:** #15772
- **Paths:** see #15772
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`8ea41a76`; collated from changes/15732-fail-closed-manifest-json-verifier-resum.md)
- **Summary:** Fail-closed manifest.json verifier, resume key and atomic publish for swing report packages (SRP-6 slice 1)
- **Next step:** Independent review, then arm squash auto-merge.

### DL-#15731 · Add the Engine-Agnostic SimulationOutput Contract and the Analytic Hand-Load Null-Space Solvers

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15731
- **Branch:** cursor/hand-load-nullspace-srp0-f288
- **PR:** #15742
- **Paths:** see #15742
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`75d7da66`; collated from changes/15731-add-the-engine-agnostic-simulationoutput.md)
- **Summary:** Add the engine-agnostic SimulationOutput contract and the analytic hand-load null-space solvers
- **Next step:** Independent reviewer checks the exact head, then arms merge.

### DL-#15725 · Add the Face-On 2D Tip-to-Tail Still for Midhands ZTCF Plus ZVCF Equals Total

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15725
- **Branch:** cursor/hand-load-face-on-stills-c3ae
- **PR:** #15760, #15761
- **Paths:** see #15760
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`bb81bfa6`; collated from changes/15725-encode-the-hand-load-comparison-grid-to.md)
- **Summary:** Add the face-on 2D tip-to-tail still for midhands ZTCF plus ZVCF equals Total
- **Next step:** Review the MP4 writer. Freeze-frame sheets stay with the swing report package.

### DL-#15703 · GUI E2E Gate Crash Under Windows Xdist

- **State:** in_review
- **Owner:** codex-ct-1
- **Issue:** #15703
- **Branch:** codex-ct-1/15703-gui-e2e
- **PR:** #15713
- **Paths:** src/shared/python/simulation_backends/capabilities.py; tests/unit/simulation_backends; tests/ui/tools/simulation_backends
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`1cf4108f`; collated from changes/15703-make-mujoco-capability-probes-safe-under.md)
- **Summary:** Make MuJoCo capability probes safe under headless Windows xdist and use CPU-only MuJoCo mode for GUI tests
- **Next step:** Review and merge the ready PR after exact-head checks pass

### DL-#15539 · Wait for Async Results Load Without Spinning the Qt Event Loop (#15539)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15539
- **Branch:** codex/15539-results-wait
- **PR:** #15541
- **Paths:** see #15541
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`5a02b0c1`; collated from changes/15539-wait-for-async-results-load-without-spin.md)
- **Summary:** Wait for async results load without spinning the Qt event loop (#15539)
- **Next step:** Review the PR; operator handles merge authorization.

### DL-#15538 · Project Playback Ball Overlay Using the View Look-At (#15538)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15538
- **Branch:** codex/15538-overlay-lookat
- **PR:** #15542
- **Paths:** see #15542
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`ccbeee4e`; collated from changes/15538-project-playback-ball-overlay-using-the.md)
- **Summary:** Project playback ball overlay using the view look-at (#15538)
- **Next step:** Independent acceptance at the new head; production wiring remains tracked by open #15538.

### DL-#15524 · Preserve the Simulation Time Origin When Exporting Contact-Interval Playback Frames.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15524
- **Branch:** codex/15524-impact-playback-timebase
- **PR:** #15581, #15598, #15615, #15572, #15548, #15626, #15589
- **Paths:** see #15581
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`cc53d0ba`; collated from changes/15524-separation-gate.md)
- **Summary:** Preserve the simulation time origin when exporting contact-interval playback frames.
- **Next step:** Independent review; PR intentionally unarmed. Wiring the signal to ImpactZonePlayback (#15615) and the window source (#15598) is a follow-up once those land.

### DL-#15497 · Expose Digest-Covered Bunker Flight and Rollout Traces in Workbench API Records

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15497
- **Branch:** feat/15497-api-flight-trace
- **PR:** #15555, #15512, #15559, #15619, #15590, #15604, #15597, #15629, #15623, #15586
- **Paths:** see #15555
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`02d306f2`; collated from changes/15497-web-flight-view.md)
- **Summary:** Expose digest-covered bunker flight and rollout traces in workbench API records
- **Next step:** Review this unarmed web slice; API payload remains in PR 15555

### DL-#15494 · Add Fixture-Driven 3D Tip-to-Tail Stills for Midhands ZTCF Plus ZVCF Equals Total

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15494
- **Branch:** cursor/hand-load-stills-8f17
- **PR:** #15750, #15611, #15751, #15741, #15569, #15562
- **Paths:** see #15750
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`4b720f2b`; collated from changes/15494-strategy-torque-margins.md)
- **Summary:** Add fixture-driven 3D tip-to-tail stills for midhands ZTCF plus ZVCF equals Total
- **Next step:** Merge the PR; wire the manifest into the video exporter.

### DL-#15399 · dE/Dt Diverging Segment Shading, Interface Transfer Glyphs, Swing-Derived Limits

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15399
- **Branch:** cursor/segment-net-power-shading-15399-4895
- **PR:** #15431
- **Paths:** see #15431
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`0c874082`; collated from changes/15399-segment-net-power-shading.md)
- **Summary:** dE/dt diverging segment shading, interface transfer glyphs, swing-derived limits
- **Next step:** CI quality-gate on PR.

### DL-#15390 · Robertson–Winter Interface Power (F·V + M·Ω), Per-Segment Balance Residual, MuJoCo Adapter

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15390
- **Branch:** cursor/rw-interface-power-15390-a314
- **PR:** #15393
- **Paths:** see #15393
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`73ec5d73`; collated from changes/15390-robertson-winter-interface-power.md)
- **Summary:** Robertson–Winter interface power (F·v + M·ω), per-segment balance residual, MuJoCo adapter
- **Next step:** CI on PR for #15390.

### DL-#15389 · Segment Kinetic-Energy Globes, Per-Frame KE Time Series API, and Translational/Rotational Split

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15389
- **Branch:** cursor/segment-ke-globe-view-f14a
- **PR:** #15391
- **Paths:** see #15391
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`889deecd`; collated from changes/15389-segment-ke-globe-overlay.md)
- **Summary:** Segment kinetic-energy globes, per-frame KE time series API, and translational/rotational split
- **Next step:** CI on PR #15391.

### DL-#15383 · Measure Native Skinned Body in Showpiece Lighting Checks

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15383
- **Branch:** codex-oglaptop-2/15383-showpiece-lighting
- **PR:** #15392
- **Paths:** see #15392
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`636f6e1a`; collated from changes/15383-measure-native-skinned-body-in-showpiece.md)
- **Summary:** Measure native skinned body in showpiece lighting checks
- **Next step:** Review the metric/test change, then qualify showpiece lighting (ragged shadow and clipping) against the acceptance bounds and remove the strict xfail.

### DL-#15244 · Derive Matched Swing Qualification Ladder From AcceptanceGates + Ledger Receipts

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15244
- **Branch:** cursor/matched-swing-ladder-15244-2d97
- **PR:** #15570
- **Paths:** see #15570
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`4c9f89a3`; collated from changes/15244-matched-swing-qualification-ladder.md)
- **Summary:** Derive matched swing qualification ladder from AcceptanceGates + ledger receipts
- **Next step:** Merge after CI green; regenerate README with generate_matched_swing_status --write when ledger changes.

### DL-#15238 · Split Positive and Negative Work and Impulse at the Zero Crossing Instead of Clipping the Trapezoid

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15238
- **Branch:** claude-desk/t4-work-zero-crossing
- **PR:** #15239
- **Paths:** see #15239
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`b0e4880c`; collated from changes/15238-power-work-zero-crossing-split.md)
- **Summary:** Split positive and negative work and impulse at the zero crossing instead of clipping the trapezoid
- **Next step:** Review the PR; operator handles merge authorization.

### DL-#15233 · Unify Spatial_Algebra Jcalc on Featherstone Passive X_J for Revolute and Prismatic Joints

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15233
- **Branch:** cursor/jcalc-featherstone-15233-2d97
- **PR:** #15560
- **Paths:** see #15560
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`1493bc1a`; collated from changes/15233-jcalc-featherstone-passive-convention.md)
- **Summary:** Unify spatial_algebra jcalc on Featherstone passive X_J for revolute and prismatic joints
- **Next step:** Merge after CI green; dynamics callers inherit the convention via shared jcalc.

### DL-#15232 · Motion Pipeline Resample Keeps the 1/Target_Fps Step and Rejects Invalid Target_Fps

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15232
- **Branch:** claude-desk/t4-resample-grid
- **PR:** #15234
- **Paths:** see #15234
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f9877dfd`; collated from changes/15232-resample-grid-keeps-target-fps-step.md)
- **Summary:** Motion Pipeline Resample Keeps the 1/target_fps Step and Rejects Invalid target_fps
- **Next step:** Review the PR; operator handles merge authorization.

### DL-#15231 · Fix Replay Impact_to_Send_Latency_ms (Replay Quantization Plus Detect-to-Send Overhead)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15231
- **Branch:** cursor/impact-send-latency-15231-2d97
- **PR:** #15567
- **Paths:** see #15567
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`e93d70af`; collated from changes/15231-impact-to-send-latency-metric.md)
- **Summary:** Fix replay impact_to_send_latency_ms (replay quantization plus detect-to-send overhead)
- **Next step:** Merge after CI green; benchmark now measures presentation-to-send budget.

### DL-#15221 · Handle Overflow Through Safe Numeric Conversion Fallback and Strict Contracts

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15221
- **Branch:** codex/burn1116-15221-numeric-overflow
- **PR:** #15222
- **Paths:** src/shared/python/core/type_utils.py,tests/unit/core/test_type_conversion_overflow.py
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`e0535f8b`; collated from changes/15221-safe-conversion-overflow.md)
- **Summary:** Handle overflow through safe numeric conversion fallback and strict contracts
- **Next step:** CI and automated review

### DL-#15218 · Preserve Fresh Caller Defaults After JSON and YAML Load Failures

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15218
- **Branch:** codex/burn1116-15218-config-default
- **PR:** #15220
- **Paths:** src/shared/python/config/config_utils.py,tests/unit/config/test_config_load_failure_defaults.py
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`517d71b7`; collated from changes/15218-config-load-defaults.md)
- **Summary:** Preserve fresh caller defaults after JSON and YAML load failures
- **Next step:** CI and automated review

### DL-#15216 · Floor Grid Indices so Points Just Below the Raster Origin Are Off-Grid

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15216
- **Branch:** claude-desk/t4-puttsg
- **PR:** #15217
- **Paths:** see #15217
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`0c3ea407`; collated from changes/15216-floor-grid-indices-so-points-just-below.md)
- **Summary:** Floor grid indices so points just below the raster origin are off-grid
- **Next step:** Review the PR.

### DL-#15196 · Validate JSON and YAML Serialization Before Changing Destination Files

- **State:** in_review
- **Owner:** codex
- **Issue:** #15196
- **Branch:** fix/json-serialization-preserve
- **PR:** #15198
- **Paths:** see #15198
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`ff1575d1`; collated from changes/15196-validate-json-and-yaml-serialization-bef.md)
- **Summary:** Validate JSON and YAML serialization before changing destination files
- **Next step:** Review serialization-failure preservation and merge through the supervised pipeline

### DL-#15193 · Resolve Dotted Import Names and Version Checks on the Requested Module

- **State:** in_review
- **Owner:** codex
- **Issue:** #15193
- **Branch:** fix/dotted-module-imports
- **PR:** #15195
- **Paths:** see #15195
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f424f4e3`; collated from changes/15193-resolve-dotted-import-names-and-version.md)
- **Summary:** Resolve dotted import names and version checks on the requested module
- **Next step:** Review dotted-module regression fix through the supervised pipeline

### DL-#15191 · Reject Invalid Seed Types Before Mutating Global RNG State

- **State:** in_review
- **Owner:** codex
- **Issue:** #15191
- **Branch:** fix/seed-validation-atomic
- **PR:** #15194
- **Paths:** see #15194
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`75c19da6`; collated from changes/15191-reject-invalid-seed-types-before-mutatin.md)
- **Summary:** Reject invalid seed types before mutating global RNG state
- **Next step:** Review seed precondition and regression evidence through the supervised pipeline

### DL-#15189 · Report Filesystem Errors Through Async and Background Save Callbacks

- **State:** in_review
- **Owner:** codex
- **Issue:** #15189
- **Branch:** fix/async-save-io-errors
- **PR:** #15190
- **Paths:** src/shared/python/data_io/_async_io.py; tests/output/wave5_output/test_async_io.py
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`45e57a32`; collated from changes/15189-report-filesystem-errors-through-async-a.md)
- **Summary:** Report filesystem errors through async and background save callbacks
- **Next step:** Review regression fix through the supervised pipeline

### DL-#15187 · Refresh the Full-Source Mypy Diagnostic Baseline and Mark Its Regression Test

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15187
- **Branch:** cursor/fix-full-src-mypy-baseline-4b58
- **PR:** #15188
- **Paths:** see #15188
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`ca9d8887`; collated from changes/15187-refresh-the-full-source-mypy-diagnostic.md)
- **Summary:** Refresh the Full-Source Mypy Diagnostic Baseline and Mark Its Regression Test
- **Next step:** Review PR #15188; operator handles merge authorization.

### DL-#15148 · Skinned Human Hand Visual Backend: MakeHuman CC0 Rig and Weights, Hierarchical LBS, Shared Curl-to-Contact Posing, EGL Evidence Renders.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15148
- **Branch:** cursor/skinned-human-hand-backend-751f
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f637f68b`; collated from changes/15148-skinned-human-hand-visual-backend-makehu.md)
- **Summary:** Skinned human hand visual backend: MakeHuman CC0 rig and weights, hierarchical LBS, shared curl-to-contact posing, EGL evidence renders.
- **Next step:** Cup the palm and solve the trail hand against the lead hand; see PR #15149 score table.

### DL-#15144 · Calibrate MyoHand Anatomical Contact Solver, Bilateral Mirroring, Skeleton JSON Export, and 74% Z-Buffer Framing

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15144
- **Branch:** cursor/myohand-visual-backend-dcca
- **PR:** #15147
- **Paths:** see #15147
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`4c18f5cc`; collated from changes/15144-myohand-skeleton-visual-calibration.md)
- **Summary:** Calibrate MyoHand anatomical contact solver, bilateral mirroring, skeleton JSON export, and 74% Z-buffer framing
- **Next step:** Integrate shared posing.py (#15164) and render.py (#15154) when landed while keeping research_only=True until #13763 owner sign-off

### DL-#15138 · HRG-GB1b: Per-Hand Shaft Roll on Grip Attachment via Apply_Shaft_Roll and Cross-Engine Parity.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15138
- **Branch:** cursor/hrg-gb1b-shaft-roll-2882
- **PR:** #15139
- **Paths:** see #15139
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`4f30c6dc`; collated from changes/15138-hrg-gb1b-per-hand-shaft-roll-on-grip-att.md)
- **Summary:** HRG-GB1b: per-hand shaft roll on grip attachment via apply_shaft_roll and cross-engine parity.
- **Next step:** Merge after GB-1 #15133; GB-2 wires roll into the procedural builder.

### DL-#15128 · HRG-GB5b: Offline Grip-Loss Evaluator With Configurable Hysteresis and Committed-Series Prediction Reports.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15128
- **Branch:** cursor/hrg-gb5b-grip-loss-evaluator-4e08
- **PR:** #15136
- **Paths:** see #15136
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`6db6bf8b`; collated from changes/15128-hrg-gb5b-grip-loss-evaluator.md)
- **Summary:** HRG-GB5b: offline grip-loss evaluator with configurable hysteresis and committed-series prediction reports.
- **Next step:** Stack merges after GB-1; GB-6 MuJoCo clamp consumes the same threshold config pattern.

### DL-#15125 · HRG-GB3 Palm Wrap, Thumb Press, Trail Overlap/Interlock/Ten-Finger Linking, Golden Checksums

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15125
- **Branch:** cursor/hrg-gb3-grip-builder-fd55
- **PR:** #15137
- **Paths:** see #15137
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`5d7af863`; collated from changes/15125-hrg-gb3-palm-thumb-trail-goldens.md)
- **Summary:** HRG-GB3 palm wrap, thumb press, trail overlap/interlock/ten-finger linking, golden checksums
- **Next step:** Review stacked on GB-2 (#15134); merge queue after parent chain.

### DL-#15124 · HRG-GB2 Engine-Free Flat-Hand Curl Builder and Build Receipt Stacked on GB-1 GripSpec

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15124
- **Branch:** cursor/hrg-gb2-grip-builder-0932
- **PR:** #15137
- **Paths:** see #15137
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`5d7af863`; collated from changes/15124-hrg-gb2-engine-free-flat-hand-curl-build.md)
- **Summary:** HRG-GB2 engine-free flat-hand curl builder and build receipt stacked on GB-1 GripSpec
- **Next step:** Merge GB-1 (#15133) then GB-2 (#15134).

### DL-#15123 · HRG-GB1: Versioned Engine-Free GripSpec, GripPose Projection, Force/Contact Providers, ADR-0057 Proposed Addendum.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15123
- **Branch:** cursor/hrg-gb1-grip-spec-96ca
- **PR:** #15133
- **Paths:** see #15133
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`e652d497`; collated from changes/15123-hrg-gb1-grip-spec.md)
- **Summary:** HRG-GB1: versioned engine-free GripSpec, GripPose projection, force/contact providers, ADR-0057 Proposed addendum.
- **Next step:** Review and merge; GB-2 may consume GripSpec for the procedural builder.

### DL-#15086 · Restore Per-Workflow Docker CI Concurrency Groups and Per-Run Image Tags in Docker-Smoke (Fix-Forward).

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15086
- **Branch:** cursor/docker-ci-concurrency-fix-f3f5
- **PR:** #15167
- **Paths:** see #15167
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`a0409003`; collated from changes/15086-docker-ci-concurrency-fix-forward.md)
- **Summary:** Restore per-workflow Docker CI concurrency groups and per-run image tags in docker-smoke (fix-forward).
- **Next step:** Merge after CI green, including Docker Size and Health Check.

### DL-#15022 · Close #15022 — Conformance Template Registry Test After #15073 Landed FastMCP

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15022
- **Branch:** cursor/close-agent-api-15022-4eef
- **PR:** #15163
- **Paths:** see #15163
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`26334a68`; collated from changes/15022-agent-api-issue-closure.md)
- **Summary:** Close #15022 — conformance template registry test after #15073 landed FastMCP
- **Next step:** Merge for issue-closure evidence; implementation already on main via #15073 (808174d49).

### DL-#15833 · Add Source-Attributed Legend Appearance Candidates

- **State:** in_review
- **Owner:** Codex
- **Issue:** #15833
- **Branch:** pr/15833-legends-profiles
- **PR:** #15840
- **Paths:** player appearance catalog, profile accessor and contract tests; morphology reference
- **Started:** 2026-11-26
- **Last verified:** 2026-11-27 (`d4c21a84`; collated from changes/15833-source-attributed-legend-profile-candidates.md)
- **Summary:** Add five source-attributed, stylized golfer profiles; preserve measurement conflicts and visual limits.
- **Next step:** Merge the PR; final delivery #15836 adds Snead's straw_hat after #15838 lands.

### DL-#15115 · Local API App Applies the Shared Allowed-Host Policy

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15115
- **Branch:** claude-og/local-server-host-check
- **PR:** #15116
- **Paths:** see #15116
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`795caaad`; collated from changes/15115-local-api-app-applies-the-shared-allowed.md)
- **Summary:** Local API app applies the shared allowed-host policy
- **Next step:** Merge the PR.

### DL-#15112 · Fail Closed on Per-Crate Maturin Build Errors in Dockerfile and Enforce in check_dockerfile_contracts.py.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15112
- **Branch:** gemini-og/15112-dockerfile-maturin-fail-closed
- **PR:** #15114
- **Paths:** see #15114
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`7699de0c`; collated from changes/15112-dockerfile-maturin-fail-closed.md)
- **Summary:** Fail closed on per-crate maturin build errors in Dockerfile and enforce in check_dockerfile_contracts.py.
- **Next step:** Merge PR for #15112.

### DL-#15107 · Fix Broken Repo Paths in Curated Testing, Architecture, and Manual Docs and Enforce in check_spec_paths.py.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15107
- **Branch:** gemini-og/15107-docs-broken-repo-paths
- **PR:** #15113
- **Paths:** see #15113
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`6b454ac0`; collated from changes/15107-docs-broken-repo-paths.md)
- **Summary:** Fix broken repo paths in curated testing, architecture, and manual docs and enforce in check_spec_paths.py.
- **Next step:** Merge PR #15113.

### DL-#13888 · Add HILL_MTU Drive Mode to 3DOF FTLE Study With Matched Nominal Excitation and Four-Mode Receipt.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13888
- **Branch:** cursor/hill-mtu-drive-mode-3dof-ftle-f296
- **PR:** #15076
- **Paths:** see #15076
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`1b4ce670`; collated from changes/13888-hill-mtu-drive-mode-3dof-ftle-study.md)
- **Summary:** Add HILL_MTU drive mode to 3DOF FTLE study with matched nominal excitation and four-mode receipt.
- **Next step:** Merge after CI; full t_end=0.2 Hill receipt rows remain NOT_QUALIFIED until MTU equilibrium converges on full swing.

### DL-#13782 · ML-WS-5 MuJoCo Corpus Run Receipt, Manifest-Verified Loader, and CI Evidence

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13782
- **Branch:** cursor/ml-ws5-corpus-receipt-loader-791f
- **PR:** #15075
- **Paths:** see #15075
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`0adc2797`; collated from changes/13782-ml-ws-5-mujoco-corpus-run-receipt-manife.md)
- **Summary:** ML-WS-5 MuJoCo corpus run receipt, manifest-verified loader, and CI evidence
- **Next step:** Barb review / CI

### DL-#13761 · Bump Dnspython to 2.9.2 and Tornado to 6.6.1; Remove Stale Tornado Pip-Audit Waivers.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13761
- **Branch:** cursor/pip-audit-dnspython-tornado-54d2
- **PR:** #15086
- **Paths:** see #15086
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`b6abbfbf`; collated from changes/13761-bump-dnspython-to-2-9-2-and-tornado-to-6.md)
- **Summary:** Bump dnspython to 2.9.2 and tornado to 6.6.1; remove stale tornado pip-audit waivers.
- **Next step:** Merge after CI green.

### DL-#13427 · Regression-Test Address Reserve Share; Wire Study Allow_Implausible and OS-9 Asset Gate

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13427
- **Branch:** cursor/spn-d9-address-reserve-share-ab3e
- **PR:** #15077
- **Paths:** see #15077
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`d56bdccf`; collated from changes/13427-regression-test-address-reserve-share-wi.md)
- **Summary:** Regression-test address reserve share; wire study allow_implausible and OS-9 asset gate
- **Next step:** Owner decides address trunk bound; merge qualified OS-9 native receipt when measured

### DL-#15014 · Fix Dempster/Zatsiorsky Segment Mass Expansion and C3D Export Placeholder (Part A)

- **State:** shipped
- **Shipped:** 2026-11-14
- **Owner:** unassigned
- **Issue:** #15014
- **Branch:** cursor/segment-mass-c3d-fix-e135
- **PR:** #15036, #15035, #15159, #15088
- **Paths:** see #15036
- **Started:** 2026-11-14
- **Last verified:** 2026-11-27 (`504086c0`; collated from changes/15014-mocap-conventions-and-de-leva-guard.md)
- **Summary:** Fix Dempster/Zatsiorsky segment mass expansion and C3D export placeholder (part a)

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
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`e63b3a62`; collated from changes/2725-remove-the-retired-vendored-gaai-framewo.md)
- **Summary:** Remove the retired vendored GAAI framework tree, its launcher and slash commands; .claude/ is now the canonical agent configuration root
- **Next step:** Shipped in PR #15119.

### DL-#15834 · Add a Visual-Only Straw Hat to the Shared Golfer Head

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15834
- **Branch:** merged via #15838
- **PR:** #15838
- **Paths:** see #15838
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f059bfb9`; collated from changes/15834-add-a-visual-only-straw-hat-to-the-share.md)
- **Summary:** Add a visual-only straw hat to the shared golfer head
- **Next step:** Shipped in PR #15838.

### DL-#15825 · Bind Explicitly Styled Native Definitions to Their Exact MuJoCo XML

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15825
- **Branch:** merged via #15827
- **PR:** #15827
- **Paths:** see #15827
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`01062bbf`; collated from changes/15825-bind-explicitly-styled-native-definition.md)
- **Summary:** Bind explicitly styled native definitions to their exact MuJoCo XML
- **Next step:** Shipped in PR #15827.

### DL-#15822 · Add a Visual Flat-Cap Appearance Style

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15822
- **Branch:** merged via #15826
- **PR:** #15826
- **Paths:** see #15826
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`abae5d6a`; collated from changes/15822-add-a-visual-flat-cap-appearance-style.md)
- **Summary:** Add a visual flat-cap appearance style
- **Next step:** Shipped in PR #15826.

### DL-#15817 · Record Historical Player Reference Video Source Provenance

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15817
- **Branch:** merged via #15820
- **PR:** #15820
- **Paths:** see #15820
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`5e818003`; collated from changes/15817-record-historical-player-reference-video.md)
- **Summary:** Record historical player reference video source provenance
- **Next step:** Shipped in PR #15820.

### DL-#15808 · Canonical-V2 Solution Bundles Can Record a Preregistered Admission Block: Controller, Interval, Per-Frame Validity, and Separate Torque Channels.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15808
- **Branch:** merged via #15810
- **PR:** #15810, #15812
- **Paths:** see #15810
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`a972f67c`; collated from changes/15808-root-wrench-residual-audit.md)
- **Summary:** Canonical-v2 solution bundles can record a preregistered admission block: controller, interval, per-frame validity, and separate torque channels.
- **Next step:** Shipped in PR #15810.

### DL-#15794 · Implement Immutable Serializable SimulationOutput (#15731) and DynamicsQuery Interface (WP B2, Panel Th_Bd39271e0dc6)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15794
- **Branch:** merged via #15796
- **PR:** #15796
- **Paths:** see #15796
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`4700d593`; collated from changes/15794-b2-simulationoutput-dynamicsquery.md)
- **Summary:** Implement immutable serializable SimulationOutput (#15731) and DynamicsQuery interface (WP B2, Panel th_bd39271e0dc6)
- **Next step:** Shipped in PR #15796.

### DL-#15776 · Extend SimulationOutput With GRF/CoP/CoG Contract Fields and Add Deterministic Frame-Level Replay Harness (WP-1)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15776
- **Branch:** merged via #15791
- **PR:** #15791
- **Paths:** see #15791
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`3525f8ee`; collated from changes/15776-wp1-simulationoutput-replay-harness.md)
- **Summary:** Extend SimulationOutput with GRF/CoP/CoG contract fields and add deterministic frame-level replay harness (WP-1)
- **Next step:** Shipped in PR #15791.

### DL-#15768 · Remove the Duplicate Certifi Floor From Pyproject Runtime Dependencies and Add a Duplicate-Dependency Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15768
- **Branch:** merged via #15778
- **PR:** #15778
- **Paths:** see #15778
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`c1cab363`; collated from changes/15768-dedupe-certifi.md)
- **Summary:** Remove the duplicate certifi floor from pyproject runtime dependencies and add a duplicate-dependency test
- **Next step:** Shipped in PR #15778.

### DL-#15763 · Compose: Container-Native PATH, Drop Obsolete Version Key, Move Dev --Reload to docker-compose.override.yml, Add Hygiene Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15763
- **Branch:** merged via #15781
- **PR:** #15781
- **Paths:** see #15781
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`a3b029ed`; collated from changes/15763-compose-hygiene.md)
- **Summary:** Compose: container-native PATH, drop obsolete version key, move dev --reload to docker-compose.override.yml, add hygiene test
- **Next step:** Shipped in PR #15781.

### DL-#15724 · Swing Report Package Command Line: Python3 -m Src.Tools.Swing_Report_Package Lists the Catalog, Dry-Runs or Runs a Default, Named or Custom Package With Key Phases From the Bundle (SRP-5 Slice 2)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15724
- **Branch:** merged via #15771
- **PR:** #15771, #15755, #15775, #15773, #15774
- **Paths:** see #15771
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`402eb6b1`; collated from changes/15724-swing-report-package-window-model-ticks.md)
- **Summary:** Swing report package command line: python3 -m src.tools.swing_report_package lists the catalog, dry-runs or runs a default, named or custom package with key phases from the bundle (SRP-5 slice 2)
- **Next step:** Shipped in PR #15771.

### DL-#15723 · Swing Report Package: Catalog of Package Items (Videos, Case Grids, Phase Sheets) With a Default Package and Request Files That Include, Exclude and Override

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15723
- **Branch:** merged via #15727
- **PR:** #15727
- **Paths:** see #15727
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`7d28f4ea`; collated from changes/15723-swing-report-package-catalog-of-package.md)
- **Summary:** Swing report package: catalog of package items (videos, case grids, phase sheets) with a default package and request files that include, exclude and override
- **Next step:** Shipped in PR #15727.

### DL-#15722 · Exported Clips Write a Frame-Times Sidecar; HUD Explains Withheld Shading at Ball Contact; Colour Key Rows Get Solid Swatches

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15722
- **Branch:** merged via #15736
- **PR:** #15736, #15747
- **Paths:** see #15736
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`34fd3e33`; collated from changes/15722-swing-report-package-key-phase-grid-and.md)
- **Summary:** Exported clips write a frame-times sidecar; HUD explains withheld shading at ball contact; colour key rows get solid swatches
- **Next step:** Shipped in PR #15736.

### DL-#15721 · Swing Report Package: Key Swing Phases (Address, Shaft Parallel, Top, Lead Arm Parallel, Impact, Follow-Through) Detected From Body Positions, Rules in Settings

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15721
- **Branch:** merged via #15726
- **PR:** #15726
- **Paths:** see #15726
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`300420c7`; collated from changes/15721-swing-report-package-key-swing-phases-ad.md)
- **Summary:** Swing report package: key swing phases (address, shaft parallel, top, lead arm parallel, impact, follow-through) detected from body positions, rules in settings
- **Next step:** Shipped in PR #15726.

### DL-#15720 · Native Viewer Export: --Ztcf-Zvcf-Hands Midhands Draws Both Hands' Load on the Club as One Equivalent Force Plus Couple at the Hand Midpoint, per ZTCF/ZVCF Case

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15720
- **Branch:** merged via #15759
- **PR:** #15759
- **Paths:** see #15759
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`e1d52239`; collated from changes/15720-native-viewer-export-ztcf-zvcf-hands-mid.md)
- **Summary:** Native viewer export: --ztcf-zvcf-hands midhands draws both hands' load on the club as one equivalent force plus couple at the hand midpoint, per ZTCF/ZVCF case
- **Next step:** Shipped in PR #15759.

### DL-#15711 · Dashboard and Web ZVCF Now Plot Applied-Control Acceleration With Pointwise ZTCF Plus ZVCF Closure.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15711
- **Branch:** merged via #15715
- **PR:** #15715
- **Paths:** see #15715
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`92c34596`; collated from changes/15711-dashboard-and-web-zvcf-now-plot-applied.md)
- **Summary:** Dashboard and web ZVCF now plot applied-control acceleration with pointwise ZTCF plus ZVCF closure.
- **Next step:** Shipped in PR #15715.

### DL-#15685 · Kinetic-Match Bundles Declare Replay_Mode Inverse_Dynamics, so Export Clips Label Them Honestly Instead of Unknown Replay Mode

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15685
- **Branch:** merged via #15707
- **PR:** #15707, #15701, #15695, #15687, #15700, #15706, #15699
- **Paths:** see #15707
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f69495c1`; collated from changes/15685-procedure-doc-forces-and-torques-for-a-n.md)
- **Summary:** Kinetic-match bundles declare replay_mode inverse_dynamics, so export clips label them honestly instead of Unknown Replay Mode
- **Next step:** Shipped in PR #15707.

### DL-#15684 · Add Club_Fitting Flex Sweep and CLI (Same Hand Motion, Every Flex)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15684
- **Branch:** merged via #15702
- **PR:** #15702, #15696, #15698, #15704, #15686
- **Paths:** see #15702
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`bfe4f0bb`; collated from changes/15684-the-current-golfer-model-full-body-skinn.md)
- **Summary:** Add club_fitting flex sweep and CLI (same hand motion, every flex)
- **Next step:** Shipped in PR #15702.

### DL-#15682 · Mirror_Spec_to_Left_Handed Reflects a Right-Handed Full-Body Spec Through the Sagittal X-Z Plane; the Right Hand Carries the Club

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15682
- **Branch:** merged via #15683
- **PR:** #15683
- **Paths:** see #15683
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`017899c8`; collated from changes/15682-mirrored-spec.md)
- **Summary:** mirror_spec_to_left_handed reflects a right-handed full-body spec through the sagittal x-z plane; the right hand carries the club
- **Next step:** Shipped in PR #15683.

### DL-#15670 · The Model Explorer Demo Swing (Trunk_Visuals/Driver_V2_Bundle.Npz) No Longer Falls Over About 0.8 s in. Its V2 Spec Now Carries the Run's Hip, Ankle, Shank and Ground Calibration (Carry_Calibration), so the Pelvis Stays Within 2.3 cm (Height) of the Run's Target. the Evidence Generator's Repository Root Is Fixed as Well.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15670
- **Branch:** merged via #15680
- **PR:** #15680
- **Paths:** see #15680
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`60bf6549`; collated from changes/15670-driver-v2-bundle-calibrated.md)
- **Summary:** The Model Explorer demo swing (trunk_visuals/driver_v2_bundle.npz) no longer falls over about 0.8 s in. Its v2 spec now carries the run's hip, ankle, shank and ground calibration (carry_calibration), so the pelvis stays within 2.3 cm (height) of the run's target. The evidence generator's repository root is fixed as well.
- **Next step:** Shipped in PR #15680.

### DL-#15659 · Canonical Golfer Gate Refuses Capsule, Loft and Unrevised Specs Before Simulation; Solution Bundles Carry a Plant/Controller/Solver Model Lock and Unlocked Bundles Are Replay-Only

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15659
- **Branch:** merged via #15668
- **PR:** #15668
- **Paths:** see #15668
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`36aec7db`; collated from changes/15659-canonical-golfer-gate.md)
- **Summary:** Canonical golfer gate refuses capsule, loft and unrevised specs before simulation; Solution Bundles carry a plant/controller/solver model lock and unlocked bundles are replay-only
- **Next step:** Shipped in PR #15668.

### DL-#15654 · The Spec Builder Moves Specs With Physical Grip Picks to the Current Golfer Model, Posing the Skinned Hands From Subject.Grip_Config; Only Left-Handed Specs Keep the Deprecated Fallback.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15654
- **Branch:** merged via #15677
- **PR:** #15677, #15656
- **Paths:** see #15677
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`7ee0859b`; collated from changes/15654-skinned-hands-grip-config.md)
- **Summary:** The spec builder moves specs with physical grip picks to the current golfer model, posing the skinned hands from subject.grip_config; only left-handed specs keep the deprecated fallback.
- **Next step:** Shipped in PR #15677.

### DL-#15652 · Ratchet Test: Committed Specs and Bundles on a Deprecated Golfer Model Part May Only Leave Src/Config/Golfer_Model_Legacy_Assets.Json, Never Join It.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15652
- **Branch:** merged via #15658
- **PR:** #15658, #15655, #15705
- **Paths:** see #15658
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`02e4a7d7`; collated from changes/15652-spec-builder-writes-left-handed-golfers.md)
- **Summary:** Ratchet test: committed specs and bundles on a deprecated golfer model part may only leave src/config/golfer_model_legacy_assets.json, never join it.
- **Next step:** Shipped in PR #15658.

### DL-#15650 · Resolve Four Mypy Arg-Type and Operator Errors in cross_engine_replay.py That Blocked Pre-Push Hooks

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15650
- **Branch:** merged via #15653
- **PR:** #15653
- **Paths:** see #15653
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`4fb75a6b`; collated from changes/15650-fix-mypy-cross-engine-replay.md)
- **Summary:** Resolve four mypy arg-type and operator errors in cross_engine_replay.py that blocked pre-push hooks
- **Next step:** Shipped in PR #15653.

### DL-#15648 · Build_Anthropometric_Spec Writes Right-Handed Specs on the Current Golfer Model (Skinned Hand Wrists, Skinned Body); Left-Handed Specs Log the Deprecated Fallback Until Left-Handed Skinned Hands Land.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15648
- **Branch:** merged via #15657
- **PR:** #15657, #15651, #15678
- **Paths:** see #15657
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`937979e1`; collated from changes/15648-golfer-reference-specs.md)
- **Summary:** build_anthropometric_spec writes right-handed specs on the current golfer model (skinned hand wrists, skinned body); left-handed specs log the deprecated fallback until left-handed skinned hands land.
- **Next step:** Shipped in PR #15657.

### DL-#15523 · Add Iron Divot Model With Fat-Shot Pre-Ball Turf Contact (Physics/divot_model.py)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15523
- **Branch:** merged via #15545
- **PR:** #15545, #15617, #15583, #15606, #15596
- **Paths:** see #15545
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`1bcb6da4`; collated from changes/15523-iron-ground-contact.md)
- **Summary:** Add iron divot model with fat-shot pre-ball turf contact (physics/divot_model.py)
- **Next step:** Shipped in PR #15545.

### DL-#15522 · Add Headless Qt End-to-End Tests Driving the Ball Flight GUI Simulate Click Through Real Physics

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15522
- **Branch:** merged via #15543
- **PR:** #15543, #15631, #15577, #15574, #15568, #15647, #15633, #15634, #15622, #15641, #15630, #15593, #15628, #15640, #15614, #15649, #15618, #15621, #15637, #15561, #15636, #15612, #15563, #15643, #15645, #15601, #15599, #15579, #15605, #15627, #15638, #15588, #15639, #15591, #15607, #15644, #15608
- **Paths:** see #15543
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`c220a014`; collated from changes/15522-unified-dashboard-qtest-e2e.md)
- **Summary:** Add headless Qt end-to-end tests driving the Ball Flight GUI Simulate click through real physics
- **Next step:** Shipped in PR #15543.

### DL-#15493 · Add Versioned Solution Bundle Schema and Strict JSON+NPZ Codec for Replayable Optimized Swings (Slice 2).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15493
- **Branch:** merged via #15564
- **PR:** #15564, #15595, #15624, #15625, #15573, #15565, #15552, #15600
- **Paths:** see #15564
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`49723453`; collated from changes/15493-solution-bundle-time-domain.md)
- **Summary:** Add versioned Solution Bundle schema and strict JSON+NPZ codec for replayable optimized swings (Slice 2).
- **Next step:** Shipped in PR #15564.

### DL-#15485 · Regression Test and Citation Tightening for Wind Consumers in the Flight Shortcomings Register (#15482 Review)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15485
- **Branch:** merged via #15550
- **PR:** #15550
- **Paths:** see #15550
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`d1a76787`; collated from changes/15485-qualify-wind-consumer-register.md)
- **Summary:** Regression test and citation tightening for wind consumers in the flight shortcomings register (#15482 review)
- **Next step:** Shipped in PR #15550.

### DL-#15471 · Codex #15464 Review Follow-Up: Tip-Slope Regression and Narrow Acceptance Reference

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15471
- **Branch:** merged via #15534
- **PR:** #15534
- **Paths:** see #15534
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`86f92af8`; collated from changes/15471-codex-15464-review-follow-up-tip-slope-r.md)
- **Summary:** Codex #15464 review follow-up: tip-slope regression and narrow acceptance reference
- **Next step:** Shipped in PR #15534.

### DL-#15444 · Normalize MuJoCo Geom_Rgba Ndarrays Before MeshCat Material Binding (#15444).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15444
- **Branch:** merged via #15525
- **PR:** #15525
- **Paths:** see #15525
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`23d638b5`; collated from changes/15444-normalize-mujoco-geom-rgba-ndarrays-befo.md)
- **Summary:** Normalize MuJoCo geom_rgba ndarrays before MeshCat material binding (#15444).
- **Next step:** Shipped in PR #15525.

### DL-#15443 · Add Reference-Shot Regression for the Ball-Flight Model: Published Driver/Iron/Wedge Launches With Carry, Apex and Descent Bands; Rows the Model Misses Are Strict Xfails With Measured Errors

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15443
- **Branch:** merged via #15450
- **PR:** #15450, #15498, #15482, #15770, #15500, #15501, #15489, #15488, #15460, #15473, #15744, #15474, #15492, #15499, #15748, #15490, #15757, #15466, #15468, #15454, #15506, #15502, #15646, #15486, #15475, #15496, #15749
- **Paths:** see #15450
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`15c4b794`; collated from changes/15443-spin-aware-runout-bounce.md)
- **Summary:** Add reference-shot regression for the ball-flight model: published driver/iron/wedge launches with carry, apex and descent bands; rows the model misses are strict xfails with measured errors
- **Next step:** Shipped in PR #15450.

### DL-#15435 · S4 Angmom Independent Oracle; Ball Excluded From Whole-Body H; Root_Body_Id Deprecated

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15435
- **Branch:** merged via #15544
- **PR:** #15544
- **Paths:** see #15544
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`5795bbc0`; collated from changes/15435-s4-angmom-oracle-ball-com-consistency.md)
- **Summary:** S4 angmom independent oracle; ball excluded from whole-body H; root_body_id deprecated
- **Next step:** Shipped in PR #15544.

### DL-#15429 · Reject Non-Finite Flight Inset Positives; Pin Headless Native Verification Clip

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15429
- **Branch:** merged via #15527
- **PR:** #15527
- **Paths:** see #15527
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`0173406c`; collated from changes/15429-reject-non-finite-flight-inset-positives.md)
- **Summary:** Reject non-finite flight inset positives; pin headless native verification clip
- **Next step:** Shipped in PR #15527.

### DL-#15398 · S4: Straight Angular-Momentum Arrows (Whole-Body + Groups, Fixed Scale) for MuJoCo/MeshCat

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15398
- **Branch:** merged via #15409
- **PR:** #15409
- **Paths:** see #15409
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`d26d6e2a`; collated from changes/15398-angular-momentum-vector-overlay.md)
- **Summary:** S4: straight angular-momentum arrows (whole-body + groups, fixed scale) for MuJoCo/MeshCat
- **Next step:** Shipped in PR #15409.

### DL-#15397 · Engine-Agnostic Segment Angular-Momentum Frame and MuJoCo Provider (About Whole-Body COM, World Frame)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15397
- **Branch:** merged via #15400
- **PR:** #15400
- **Paths:** see #15400
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`a439780d`; collated from changes/15397-segment-angular-momentum-provider-mujoco.md)
- **Summary:** Engine-agnostic segment angular-momentum frame and MuJoCo provider (about whole-body COM, world frame)
- **Next step:** Shipped in PR #15400.

### DL-#15396 · Segment Kinetic-Energy Surface Shading (Cividis, Fixed Limits); Demote KE Globes to KE Proxy Debug Overlay

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15396
- **Branch:** merged via #15427
- **PR:** #15427
- **Paths:** see #15427
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`510beb20`; collated from changes/15396-segment-ke-surface-shading.md)
- **Summary:** Segment kinetic-energy surface shading (cividis, fixed limits); demote KE globes to KE proxy debug overlay
- **Next step:** Shipped in PR #15427.

### DL-#15395 · Generalize Segment Tension/Compression Shading Into Quantity-Agnostic Scalar Series and Colour Scale

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15395
- **Branch:** merged via #15403
- **PR:** #15403
- **Paths:** see #15403
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`0ad23f4f`; collated from changes/15395-segment-scalar-shading-core.md)
- **Summary:** Generalize segment tension/compression shading into quantity-agnostic scalar series and colour scale
- **Next step:** Shipped in PR #15403.

### DL-#15345 · Bundle Adjustment FC-011 Guards, Feasible Tangent Projection, Bounded FD, and Positive Prior Sigmas

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15345
- **Branch:** merged via #15537
- **PR:** #15537
- **Paths:** see #15537
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`38425e26`; collated from changes/15345-bundle-adjustment-fc-011-guards-inward-f.md)
- **Summary:** Bundle adjustment FC-011 guards, feasible tangent projection, bounded FD, and positive prior sigmas
- **Next step:** Shipped in PR #15537.

### DL-#15344 · Guard Degenerate GP Radius Window in Compute_Correlation_Dimension

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15344
- **Branch:** merged via #15530
- **PR:** #15530
- **Paths:** see #15530
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`8994c0e2`; collated from changes/15344-guard-degenerate-gp-radius-window-in-com.md)
- **Summary:** Guard degenerate GP radius window in compute_correlation_dimension
- **Next step:** Shipped in PR #15530.

### DL-#15343 · Guard Hysteresis, Bound Power Durations, Defensive Work-Metrics Cache Copies

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15343
- **Branch:** merged via #15531
- **PR:** #15531
- **Paths:** see #15531
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`c981f575`; collated from changes/15343-guard-hysteresis-bound-power-durations-d.md)
- **Summary:** Guard hysteresis, bound power durations, defensive work-metrics cache copies
- **Next step:** Shipped in PR #15531.

### DL-#15342 · Require a List for Subject.Dropped_Markers in Marker_Selection; Dict, Int, Bool, Tuple and Stringified Values Now Raise ValueError

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15342
- **Branch:** merged via #15513
- **PR:** #15513
- **Paths:** see #15513
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`fa228f3a`; collated from changes/15342-require-a-list-for-subject-dropped-marke.md)
- **Summary:** Require a list for subject.dropped_markers in marker_selection; dict, int, bool, tuple and stringified values now raise ValueError
- **Next step:** Shipped in PR #15513.

### DL-#15333 · C3D Viewer Embed Loader Registers the Tool Again When the Registry Was Reset After the First Import

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15333
- **Branch:** merged via #15334
- **PR:** #15334
- **Paths:** see #15334
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`224fae7e`; collated from changes/15333-c3d-viewer-embed-loader-registers-the-to.md)
- **Summary:** C3D viewer embed loader registers the tool again when the registry was reset after the first import
- **Next step:** Shipped in PR #15334.

### DL-#15329 · A Hand May Be Given a Thumb Target of Its Own (Thumb.Toward_Index_Target_of, None Set): The Trail Thumb Can Close a V on Its Forefinger While the Lead Thumb Keeps Its Azimuth. A Hand's Thumb May Be Held Flexed, a Short Thumb (Thumb.Held_Flexion_Deg_of, None Set): Tried for the Lead Thumb Under the Squarer Trail Seats and Not Set, Because the Raised Knuckle Lifts the Trail Palm Off the Thumb (Crown Gap 12.6 to 15.3-17.9 mm); Drawn Hands Unchanged (Appearance Only)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15329
- **Branch:** merged via #15372
- **PR:** #15372, #15364, #15368, #15377, #15362
- **Paths:** see #15372
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`0f125125`; collated from changes/15329-set-metacarpal-arch-for-the-skinned-hand.md)
- **Summary:** A hand may be given a thumb target of its own (thumb.toward_index_target_of, none set): the trail thumb can close a V on its forefinger while the lead thumb keeps its azimuth. A hand's thumb may be held flexed, a short thumb (thumb.held_flexion_deg_of, none set): tried for the lead thumb under the squarer trail seats and not set, because the raised knuckle lifts the trail palm off the thumb (crown gap 12.6 to 15.3-17.9 mm); drawn hands unchanged (appearance only)
- **Next step:** Shipped in PR #15372.

### DL-#15322 · Pelvis Skin Map Scales Lateral Hip Spacing to Match Subject Hip_Half_Spacing

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15322
- **Branch:** merged via #15556
- **PR:** #15556
- **Paths:** see #15556
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`45491e1d`; collated from changes/15322-pelvis-map-hip-half-spacing.md)
- **Summary:** Pelvis skin map scales lateral hip spacing to match subject hip_half_spacing
- **Next step:** Shipped in PR #15556.

### DL-#15299 · Add a Chest Depth Scale and a Thigh-Front Shift to the CC0 Body Skin Fit as Named Settings, Neutral by Default and Set by the Showpiece Profile

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15299
- **Branch:** merged via #15325
- **PR:** #15325, #15324, #15317, #15321, #15335, #15314, #15384, #15331, #15316, #15367, #15366, #15303, #15310, #15309, #15328, #15338, #15313, #15336, #15370, #15308, #15386, #15307, #15365, #15371, #15375, #15356, #15376, #15385, #15360, #15353, #15369, #15379, #15347, #15341, #15339, #15349, #15363, #15305, #15326, #15311
- **Paths:** see #15325
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`1918c7f4`; collated from changes/15299-wrist-ball-elbow-and-one-tone-skin.md)
- **Summary:** Add a chest depth scale and a thigh-front shift to the CC0 body skin fit as named settings, neutral by default and set by the showpiece profile
- **Next step:** Shipped in PR #15325.

### DL-#15296 · Add a Selectable Club-Only Driver Club Marker Registration (Club_Only); the Default Driver.Json and the Committed Documents Are Unchanged

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15296
- **Branch:** merged via #15302
- **PR:** #15302, #15374, #15304, #15320, #15361, #15351, #15340, #15346, #15337, #15332, #15354, #15348, #15318, #15330, #15315, #15358, #15312
- **Paths:** see #15302
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`7684bb42`; collated from changes/15296-thigh-shank-and-hip-spacing-as-optional.md)
- **Summary:** Add a selectable club-only driver club marker registration (club_only); the default driver.json and the committed documents are unchanged
- **Next step:** Shipped in PR #15302.

### DL-#15292 · Slanted Crew-Neck Collar and Shoulder Caps Lowered Below the Neck Point (Trunk, Yoke and Upper-Arm Geometry)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15292
- **Branch:** merged via #15298
- **PR:** #15298, #15300, #15294
- **Paths:** see #15298
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`74469014`; collated from changes/15292-visual-neck-solid-for-the-cc0-head-thora.md)
- **Summary:** Slanted crew-neck collar and shoulder caps lowered below the neck point (trunk, yoke and upper-arm geometry)
- **Next step:** Shipped in PR #15298.

### DL-#15280 · Hide Contact Spheres by Default, Light Golf Shoes With Dark Sole, Tapered Legs

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15280
- **Branch:** merged via #15281
- **PR:** #15281
- **Paths:** see #15281
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`2b02c618`; collated from changes/15280-hide-contact-spheres-by-default-light-go.md)
- **Summary:** Hide contact spheres by default, light golf shoes with dark sole, tapered legs
- **Next step:** Shipped in PR #15281.

### DL-#15279 · Shoulder Caps Sit Lower and Smaller so the Shoulders Read as Shoulders, Not Pads

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15279
- **Branch:** merged via #15290
- **PR:** #15290, #15282, #15283
- **Paths:** see #15290
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`5033cda6`; collated from changes/15279-upper-body-builder-can-take-the-clavicle.md)
- **Summary:** Shoulder caps sit lower and smaller so the shoulders read as shoulders, not pads
- **Next step:** Shipped in PR #15290.

### DL-#15278 · Knuckle Sideways Range as a Setting (Off by Default): A Bent Knuckle's Sideways Swing Is Held to a Range That Shrinks With Flexion

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15278
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f637f68b`; collated from changes/15278-trail-thumb-set-40-degrees-to-the-target.md)
- **Summary:** Knuckle sideways range as a setting (off by default): a bent knuckle's sideways swing is held to a range that shrinks with flexion
- **Next step:** Shipped in PR #15288.

### DL-#15277 · Arm Refit Takes Posture Targets; Address Posture Is a Neutral Lumbar Joint With a Slight Thoracic Bend

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15277
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f637f68b`; collated from changes/15277-arm-refit-takes-posture-targets-address.md)
- **Summary:** Arm refit takes posture targets; address posture is a neutral lumbar joint with a slight thoracic bend
- **Next step:** Shipped in PR #15288.

### DL-#15276 · Golfer Head: Dirty Blonde Short-Crop Hair With a Natural Taper, Eyes, Brows and Mouth

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15276
- **Branch:** merged via #15287
- **PR:** #15287
- **Paths:** see #15287
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`117df07e`; collated from changes/15276-golfer-head-dirty-blonde-short-crop-hair.md)
- **Summary:** Golfer head: dirty blonde short-crop hair with a natural taper, eyes, brows and mouth
- **Next step:** Shipped in PR #15287.

### DL-#15275 · Add Per-Frame Swing Wrist and Forearm Readout (Flexion/Extension, Radial/Ulnar Deviation, Pronation/Supination) and 3-Row Plot Script

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15275
- **Branch:** merged via #15284
- **PR:** #15284
- **Paths:** see #15284
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`4237aa18`; collated from changes/15275-add-per-frame-swing-wrist-and-forearm-re.md)
- **Summary:** Add per-frame swing wrist and forearm readout (flexion/extension, radial/ulnar deviation, pronation/supination) and 3-row plot script
- **Next step:** Shipped in PR #15284.

### DL-#15272 · Skinned Hand: Typed Landmarks and Joint Frames Export for the Other Hand Tracks

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15272
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f637f68b`; collated from changes/15272-skinned-hand-typed-landmarks-and-joint-f.md)
- **Summary:** Skinned hand: typed landmarks and joint frames export for the other hand tracks
- **Next step:** Shipped in PR #15288.

### DL-#15270 · Skinned Hand: Palm Fold Joint so the Trail Palm Closes on the Lead Thumb

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15270
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f637f68b`; collated from changes/15270-palm-fold-joint-so-the-trail-palm-closes.md)
- **Summary:** Skinned hand: palm fold joint so the trail palm closes on the lead thumb
- **Next step:** Shipped in PR #15288.

### DL-#15265 · Rest the Trail Palm on the Lead Thumb: Owner Trail Cross Angle, Thumb Contact Check

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15265
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f637f68b`; collated from changes/15265-rest-the-trail-palm-on-the-lead-thumb-ow.md)
- **Summary:** Rest the trail palm on the lead thumb: owner trail cross angle, thumb contact check
- **Next step:** Shipped in PR #15288.

### DL-#15264 · Size the Forearm's Wrist End From the Skinned Hand's Wrist When the Skinned Hands Are Drawn

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15264
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f637f68b`; collated from changes/15264-size-the-forearm-s-wrist-end-from-the-sk.md)
- **Summary:** Size the forearm's wrist end from the skinned hand's wrist when the skinned hands are drawn
- **Next step:** Shipped in PR #15288.

### DL-#15262 · Draw the Forearm as One Continuous Tapered Shape Across Its Rotation Joint, so the Golfer No Longer Appears to Have Two Elbows

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15262
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f637f68b`; collated from changes/15262-draw-the-forearm-as-one-continuous-taper.md)
- **Summary:** Draw the forearm as one continuous tapered shape across its rotation joint, so the golfer no longer appears to have two elbows
- **Next step:** Shipped in PR #15288.

### DL-#15258 · Arm Scale Estimated From Capture Marker Distances With Stated Skin-to-Joint Offsets

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15258
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f637f68b`; collated from changes/15258-arm-scale-estimated-from-capture-marker.md)
- **Summary:** Arm scale estimated from capture marker distances with stated skin-to-joint offsets
- **Next step:** Shipped in PR #15288.

### DL-#15255 · Arm Refit That Returns the Club to Its Pose After the Wrist Joints Move to the Skinned Hands, With Optional Trunk Coordinates

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15255
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f637f68b`; collated from changes/15255-arm-refit-that-returns-the-club-to-its-p.md)
- **Summary:** Arm refit that returns the club to its pose after the wrist joints move to the skinned hands, with optional trunk coordinates
- **Next step:** Shipped in PR #15288.

### DL-#15253 · Spec Revision That Moves the Golfer's Wrist Joints to the Skinned Hands' Wrists; Skinned Club Hands Cut and Capped at the Wrist

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15253
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f637f68b`; collated from changes/15253-spec-revision-that-moves-the-golfer-s-wr.md)
- **Summary:** Spec revision that moves the golfer's wrist joints to the skinned hands' wrists; skinned club hands cut and capped at the wrist
- **Next step:** Shipped in PR #15288.

### DL-#15251 · Skinned Human Hands Selectable on the Golfer's Club, Squared to the Clubface (Subject.Hand_Visual_Backend)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15251
- **Branch:** merged via #15288
- **PR:** #15288
- **Paths:** see #15288
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f637f68b`; collated from changes/15251-skinned-human-hands-selectable-on-the-go.md)
- **Summary:** Skinned human hands selectable on the golfer's club, squared to the clubface (subject.hand_visual_backend)
- **Next step:** Shipped in PR #15288.

### DL-#15248 · Remove Unused Numpy Import From Motion_Pipeline Resample Preprocessing (Ruff F401)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15248
- **Branch:** barb/remove-unused-numpy-import
- **PR:** #15248
- **Paths:** see #15248
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`53dd87f7`; collated from changes/15248-remove-unused-numpy-import-from-motion-p.md)
- **Summary:** Remove unused numpy import from motion_pipeline resample preprocessing (ruff F401)
- **Next step:** Shipped in PR #15248.

### DL-#15246 · Conflict Disarm Starts Again: Pass Only the Secrets the Pinned Provider Declares

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15246
- **Branch:** merged via #15247
- **PR:** #15247
- **Paths:** see #15247
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`caeb99c2`; collated from changes/15246-conflict-disarm-starts-again-pass-only-t.md)
- **Summary:** Conflict Disarm starts again: pass only the secrets the pinned provider declares
- **Next step:** Shipped in PR #15247.

### DL-#15241 · Preserve JSON Recording Files on Serialization or Publication Failure by Reusing the Atomic Text Writer.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15241
- **Branch:** merged via #15243
- **PR:** #15243
- **Paths:** see #15243
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`93797bb7`; collated from changes/15241-preserve-json-recording-files-on-seriali.md)
- **Summary:** Preserve JSON recording files on serialization or publication failure by reusing the atomic text writer.
- **Next step:** Shipped in PR #15243.

### DL-#15240 · Honor Package Anchors in Safe_Import Relative Module Resolution

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15240
- **Branch:** merged via #15242
- **PR:** #15242
- **Paths:** see #15242
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`a401215a`; collated from changes/15240-safe-relative-imports.md)
- **Summary:** Honor package anchors in safe_import relative module resolution
- **Next step:** Shipped in PR #15242.

### DL-#15236 · Reject Unsupported Analysis Report Formats Before Creating Directories or Capturing Provenance, Preserving JSON and HTML Exports.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15236
- **Branch:** merged via #15237
- **PR:** #15237
- **Paths:** see #15237
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`c38937eb`; collated from changes/15236-reject-unsupported-analysis-report-forma.md)
- **Summary:** Reject unsupported analysis report formats before creating directories or capturing provenance, preserving JSON and HTML exports.
- **Next step:** Shipped in PR #15237.

### DL-#15227 · RQA Determinism Counts Diagonal Lines in Both Triangles and Excludes the Main Diagonal by Its Content

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15227
- **Branch:** merged via #15230
- **PR:** #15230
- **Paths:** see #15230
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`55dbf8d1`; collated from changes/15227-rqa-determinism-counts-diagonal-lines-in.md)
- **Summary:** RQA determinism counts diagonal lines in both triangles and excludes the main diagonal by its content
- **Next step:** Shipped in PR #15230.

### DL-#15226 · Compare_Scalars No Longer Reports 0% for Opposite-Sign or NaN Values

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15226
- **Branch:** merged via #15229
- **PR:** #15229
- **Paths:** see #15229
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`af7c1593`; collated from changes/15226-compare-scalars-no-longer-reports-0-for.md)
- **Summary:** compare_scalars no longer reports 0% for opposite-sign or NaN values
- **Next step:** Shipped in PR #15229.

### DL-#15225 · Pair Tools Cache Count-Pressure Correction While Preserving Integrity and Minimum-Age Checks

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15225
- **Branch:** merged via #15228
- **PR:** #15228
- **Paths:** see #15228
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`4a46a5ed`; collated from changes/15225-pair-tools-cache-count-pressure-correcti.md)
- **Summary:** Pair Tools cache count-pressure correction while preserving integrity and minimum-age checks
- **Next step:** Shipped in PR #15228.

### DL-#15223 · SwingComparator No Longer Scores 100 When the Reference Peak Speed, Club Head Speed or Tempo Ratio Is Zero.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15223
- **Branch:** merged via #15224
- **PR:** #15224
- **Paths:** see #15224
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`b1d00398`; collated from changes/15223-swing-comparison-zero-reference-score.md)
- **Summary:** SwingComparator no longer scores 100 when the reference peak speed, club head speed or tempo ratio is zero.
- **Next step:** Shipped in PR #15224.

### DL-#15214 · Adopt Tools Cumulative Integral Initial Offset via Coherent Provider Pin and Consumer Regressions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15214
- **Branch:** merged via #15575
- **PR:** #15575
- **Paths:** see #15575
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`54b5f908`; collated from changes/15214-adopt-paired-cumulative-integral-offset.md)
- **Summary:** Adopt Tools cumulative integral initial offset via coherent provider pin and consumer regressions
- **Next step:** Shipped in PR #15575.

### DL-#15213 · Compute_Screw_Axis Refuses Non-Finite Twists, Bad Thresholds and Bad Endpoint Lengths Instead of Returning NaN Screw Axes.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15213
- **Branch:** merged via #15215
- **PR:** #15215
- **Paths:** see #15215
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`b208d7aa`; collated from changes/15213-screw-axis-refuses-non-finite-twists.md)
- **Summary:** compute_screw_axis refuses non-finite twists, bad thresholds and bad endpoint lengths instead of returning NaN screw axes.
- **Next step:** Shipped in PR #15215.

### DL-#15209 · Reject NaN Environment Floats When Numeric Bounds Are Supplied; Pair Tools#6083

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15209
- **Branch:** merged via #15211
- **PR:** #15211
- **Paths:** see #15211
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`3e0db00b`; collated from changes/15209-reject-nan-environment-floats-when-numer.md)
- **Summary:** Reject NaN environment floats when numeric bounds are supplied; pair Tools#6083
- **Next step:** Shipped in PR #15211.

### DL-#15206 · Spectral Arc Length Now Uses the Adaptive Cut-Off From Balasubramanian 2015, so It No Longer Depends on Fc

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15206
- **Branch:** merged via #15210
- **PR:** #15210
- **Paths:** see #15210
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`2de7b64d`; collated from changes/15206-spectral-arc-length-adaptive-cutoff.md)
- **Summary:** Spectral arc length now uses the adaptive cut-off from Balasubramanian 2015, so it no longer depends on fc
- **Next step:** Shipped in PR #15210.

### DL-#15205 · Capture an Independent Deep Snapshot of Provenance Parameters so Caller Mutations Cannot Rewrite Run Metadata or Later Export Headers.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15205
- **Branch:** merged via #15235
- **PR:** #15235
- **Paths:** see #15235
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`09b84de8`; collated from changes/15205-capture-an-independent-deep-snapshot-of.md)
- **Summary:** Capture an independent deep snapshot of provenance parameters so caller mutations cannot rewrite run metadata or later export headers.
- **Next step:** Shipped in PR #15235.

### DL-#15204 · Use Transverse Spin in the Python Magnus/Lift Force Laws (BallFlightSimulator, LiftModel/MagnusModel, ODE Flight Models) as #15042 Did for Rust.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15204
- **Branch:** merged via #15208
- **PR:** #15208
- **Paths:** see #15208
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`8a715591`; collated from changes/15204-use-transverse-spin-in-python-magnus-lift.md)
- **Summary:** Use transverse spin in the Python Magnus/lift force laws (BallFlightSimulator, LiftModel/MagnusModel, ODE flight models) as #15042 did for Rust.
- **Next step:** Shipped in PR #15208.

### DL-#15202 · Reject Zero-Row CSV Frames and Trajectory Arrays Before Reporting Successful Loads

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15202
- **Branch:** merged via #15207
- **PR:** #15207
- **Paths:** see #15207
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`a565e63f`; collated from changes/15202-reject-zero-row-csv-frames-and-trajector.md)
- **Summary:** Reject zero-row CSV frames and trajectory arrays before reporting successful loads
- **Next step:** Shipped in PR #15207.

### DL-#15200 · Preserve Exact NumPy Integer Values and JSON Integer Types in Output Exports

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15200
- **Branch:** merged via #15203
- **PR:** #15203
- **Paths:** see #15203
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`c2c88b51`; collated from changes/15200-preserve-exact-numpy-integer-values-and.md)
- **Summary:** Preserve exact NumPy integer values and JSON integer types in output exports
- **Next step:** Shipped in PR #15203.

### DL-#15199 · Stage Legacy Launcher Configuration Copies so Partial Failures Remain Retryable

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15199
- **Branch:** merged via #15201
- **PR:** #15201
- **Paths:** see #15201
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`a75eb07c`; collated from changes/15199-stage-legacy-launcher-configuration-copi.md)
- **Summary:** Stage legacy launcher configuration copies so partial failures remain retryable
- **Next step:** Shipped in PR #15201.

### DL-#15150 · Add a Dispersion-on-the-Hole Still (--Hole, --Hole-Preset) to the Shot Dispersion CLI

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15150
- **Branch:** merged via #15507
- **PR:** #15507, #15441, #15192, #15476, #15758, #15423, #15738, #15472, #15797, #15783, #15789, #15458, #15674, #15394, #15479, #15381, #15802, #15798, #15681, #15510, #15417, #15432, #15413, #15407, #15465, #15515, #15481, #15746, #15754, #15753, #15740, #15478, #15487, #15795, #15714, #15428, #15437, #15374, #15449, #15457, #15456, #15436, #15716, #15739, #15412, #15415, #15434, #15464, #15784, #15728, #15719, #15422, #15533, #15520, #15528, #15526, #15519, #15532, #15470, #15408, #15387, #15455, #15411, #15402, #15433, #15799, #15421, #15420, #15426, #15410, #15424, #15790, #15469, #15484, #15453, #15401, #15779, #15483, #15509, #15418, #15718, #15769, #15517, #15467, #15461, #15756, #15504, #15438, #15792, #15414, #15406, #15405, #15430, #15440, #15508, #15745, #15785, #15752, #15730, #15516, #15642, #15549, #15540, #15566, #15547, #15557, #15529, #15553, #15536, #15480, #15511, #15514, #15477, #15378, #15712, #15425, #15380, #15675, #15463, #15780, #15679, #15777, #15801, #15737, #15197, #15459, #15451, #15419, #15442, #15382, #15388, #15404, #15729, #15301, #15505, #15717, #15416, #15350, #15355, #15359, #15708, #15503, #15709, #15743
- **Paths:** see #15507
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`ca73cff4`; collated from changes/15150-ztcf-zvcf-hand-arrows-draw-each-hand-in.md)
- **Summary:** Add a dispersion-on-the-hole still (--hole, --hole-preset) to the shot dispersion CLI
- **Next step:** Shipped in PR #15507.

### DL-#15130 · Local Desktop API App Requires the Launcher Capability Token on API Requests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15130
- **Branch:** merged via #15132
- **PR:** #15132
- **Paths:** see #15132
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`4a1c5f43`; collated from changes/15130-local-app-requires-the-launcher-capability.md)
- **Summary:** Local desktop API app requires the launcher capability token on API requests
- **Next step:** Shipped in PR #15132.

### DL-#15122 · Fleet Critic 2026-11-20: Grip Epic Critique (Builder Capsule Clearance Uses Max Not Min; Grip-Loss Shear Estimate Drops Axial Pull; Near-Circular Null Result).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15122
- **Branch:** merged via #15352
- **PR:** #15352, #15357
- **Paths:** see #15352
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`cc80419a`; collated from changes/15122-pragmatic-programmer-review-2026-11-20-o.md)
- **Summary:** Fleet Critic 2026-11-20: grip epic critique (builder capsule clearance uses max not min; grip-loss shear estimate drops axial pull; near-circular null result).
- **Next step:** Shipped in PR #15352.

### DL-#15119 · Record the Reviewed GAAI Policy-Test Retirement for Merge-Group CI

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15119
- **Branch:** merged via #15119
- **PR:** #15119
- **Paths:** see #15119
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`e63b3a62`; collated from changes/15119-reviewed-test-deletion.md)
- **Summary:** Record the reviewed GAAI policy-test retirement for merge-group CI
- **Next step:** Shipped in PR #15119.

### DL-#15106 · Prune Dead Testpaths/LEGACY_SRC_TEST_DIRS, Enable Green Tests/* Suites in Unit-Test-Gate, and Enforce Unrun Suite Exclusions (#15106)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15106
- **Branch:** merged via #15110
- **PR:** #15110
- **Paths:** see #15110
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f25a6b67`; collated from changes/15106-testpaths-and-unrun-suite-exclusions.md)
- **Summary:** Prune dead testpaths/LEGACY_SRC_TEST_DIRS, enable green tests/* suites in unit-test-gate, and enforce unrun suite exclusions (#15106)
- **Next step:** Shipped in PR #15110.

### DL-#15105 · Guard Drain_Process_Streams Against Infinite MagicMock Readline Loops in Launcher Pipe Drains

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15105
- **Branch:** merged via #15157
- **PR:** #15157
- **Paths:** see #15157
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`e2cecf0c`; collated from changes/15105-drain-magicmock-guard.md)
- **Summary:** Guard drain_process_streams against infinite MagicMock readline loops in launcher pipe drains
- **Next step:** Shipped in PR #15157.

### DL-#15103 · Guard Pytest Makereport and Teardown Against Leaked Os.Name Patches

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15103
- **Branch:** merged via #15168
- **PR:** #15168
- **Paths:** see #15168
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`b8159ea3`; collated from changes/15103-guard-pytest-makereport-and-teardown-aga.md)
- **Summary:** Guard pytest makereport and teardown against leaked os.name patches
- **Next step:** Shipped in PR #15168.

### DL-#15029 · Wave-0 Budget Expiry Cohort Closure Tests; Confirm Gates Past 2026-11-30.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15029
- **Branch:** cursor/budget-wave0-reconcile-88ba
- **PR:** #15162, #15174
- **Paths:** see #15162
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`316a2989`; collated from changes/15029-wave0-architecture-issue-field.md)
- **Summary:** Wave-0 budget expiry cohort closure tests; confirm gates past 2026-11-30.
- **Next step:** Shipped in PR #15162.

### DL-#15015 · Align GATES.md With AcceptanceGates, Disambiguate A..G Order, and Add --Pelvis-Ilc-Passes

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15015
- **Branch:** merged via #15078
- **PR:** #15078, #15173, #15160, #15172, #15178, #15171, #15170
- **Paths:** see #15078
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`3a36793e`; collated from changes/15015-two-window-pelvis-yaw-nan-00d4.md)
- **Summary:** Align GATES.md with AcceptanceGates, disambiguate A..G order, and add --pelvis-ilc-passes
- **Next step:** Shipped in PR #15078.

### DL-#14756 · Cross-Engine Replay Parity Reports Use Aligned Marker Trajectories, Not Aggregate RMSE Deltas.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14756
- **Branch:** merged via #15165
- **PR:** #15165, #15373, #15587, #15613, #15594, #15602, #15578, #15576, #15580
- **Paths:** see #15165
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`a7326c81`; collated from changes/14756-trunk-bundle-source.md)
- **Summary:** Cross-engine replay parity reports use aligned marker trajectories, not aggregate RMSE deltas.
- **Next step:** Independent review; keep PR unarmed.

### DL-#13575 · Retire Jules Push/PR Noise Workflows and Use FLEET_GENERAL_AGENT App Credentials for Fleet Policy Gates.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13575
- **Branch:** merged via #14999
- **PR:** #14999
- **Paths:** see #14999
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`310c1327`; collated from changes/13575-retire-jules-workflows.md)
- **Summary:** Retire Jules push/PR noise workflows and use FLEET_GENERAL_AGENT App credentials for fleet policy gates.
- **Next step:** Shipped in PR #14999.

### DL-#13314 · Unit-Test Gate Runs in Three Shards; Leg Two of Two Was Cancelled at Its 35 Minute Limit

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13314
- **Branch:** merged via #15319
- **PR:** #15319
- **Paths:** see #15319
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`f7827f0a`; collated from changes/13314-unit-test-gate-runs-in-three-shards-leg.md)
- **Summary:** Unit-test gate runs in three shards; leg two of two was cancelled at its 35 minute limit
- **Next step:** Shipped in PR #15319.

### DL-#12691 · Correct the GS3DX Design Report: The Committed Simscape Golfer Ships a Rigid Shaft, and a Test Pins the Claim to the Model.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12691
- **Branch:** merged via #15554
- **PR:** #15554
- **Paths:** see #15554
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`3c07a1c2`; collated from changes/12691-gs3dx-design-report-rigid-shaft.md)
- **Summary:** Correct the GS3DX design report: the committed Simscape golfer ships a rigid shaft, and a test pins the claim to the model.
- **Next step:** Shipped in PR #15554.

### DL-#12678 · Preserve Smooth CC0 Hair Shading in Shared Native-Viewer OBJ Exports, Matching MuJoCo's Hair Selection

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12678
- **Branch:** merged via #15571
- **PR:** #15571
- **Paths:** see #15571
- **Started:** 2026-11-27
- **Last verified:** 2026-11-27 (`af612ed8`; collated from changes/12678-native-hair-normals.md)
- **Summary:** Preserve smooth CC0 hair shading in shared native-viewer OBJ exports, matching MuJoCo's hair selection
- **Next step:** Shipped in PR #15571.

### DL-#15117 · Desktop UI Sends the Launcher Capability Token on Backend API Requests Through One Shared Fetch Helper

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15117
- **Branch:** merged via #15118
- **PR:** #15118
- **Paths:** see #15118
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`800ac553`; collated from changes/15117-desktop-ui-sends-the-launcher-capability.md)
- **Summary:** Desktop UI sends the launcher capability token on backend API requests through one shared fetch helper
- **Next step:** Shipped in PR #15118.

### DL-#15108 · Scope Tests/Unit/conftest.py Engine MagicMocks to Tests/Unit and Guard OpenSim Dynamics Parity (#15108)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15108
- **Branch:** merged via #15109
- **PR:** #15109
- **Paths:** see #15109
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`daf6ce2a`; collated from changes/15108-scope-unit-conftest-engine-mocks.md)
- **Summary:** Scope tests/unit/conftest.py engine MagicMocks to tests/unit and guard OpenSim dynamics parity (#15108)
- **Next step:** Shipped in PR #15109.

### DL-#15101 · Join or Retain Retiring BinderThread and CameraWorker Instances Until Qt Finishes Thread Teardown so Capture Rig Preview Never Aborts in ~QThread.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15101
- **Branch:** merged via #15102
- **PR:** #15102
- **Paths:** see #15102
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`8123908e`; collated from changes/15101-preview-qthread-lifetime.md)
- **Summary:** Join or retain retiring BinderThread and CameraWorker instances until Qt finishes thread teardown so Capture Rig preview never aborts in ~QThread.
- **Next step:** Shipped in PR #15102.

### DL-#15097 · Isolate Sidekick Sys.Modules and Sys.Meta_Path State Across Overlay Tests (#15097)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15097
- **Branch:** merged via #15099
- **PR:** #15099
- **Paths:** see #15099
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`eb2d9b7e`; collated from changes/15097-sidekick-overlay-module-isolation.md)
- **Summary:** Isolate sidekick sys.modules and sys.meta_path state across overlay tests (#15097)
- **Next step:** Shipped in PR #15099.

### DL-#15091 · Drain Child Output Streams Before Transitioning ManagedProcess Exit State and Exposing Exit_Code

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15091
- **Branch:** merged via #15092
- **PR:** #15092
- **Paths:** see #15092
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`257654e0`; collated from changes/15091-stderr-drain-before-exit.md)
- **Summary:** Drain child output streams before transitioning ManagedProcess exit state and exposing exit_code
- **Next step:** Shipped in PR #15092.

### DL-#15013 · Restore UdAttachmentManifestMixin so Rejected File_Upload Clears Dedup SHA and Retry Sends WebSocket Again; Re-Add Reject-Then-Retry Regression Tests.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15013
- **Branch:** merged via #15072
- **PR:** #15072, #15060, #15156
- **Paths:** see #15072
- **Started:** 2026-11-15
- **Last verified:** 2026-11-27 (`d7bdbc56`; collated from changes/15013-sg-analytics-documentation-close.md)
- **Summary:** Restore UdAttachmentManifestMixin so rejected file_upload clears dedup SHA and retry sends WebSocket again; re-add reject-then-retry regression tests.
- **Next step:** Shipped in PR #15072.

### DL-#14850 · Redact Host/Home Literals in Batch 5 Docs and Add Shell Placeholder Regression Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14850
- **Branch:** merged via #14850
- **PR:** #14850
- **Paths:** see #14850
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`9a0dbab7`; collated from changes/14850-redact-host-home-literals-in-batch-5-doc.md)
- **Summary:** Redact host/home literals in batch 5 docs and add shell placeholder regression test
- **Next step:** Shipped in PR #14850.

### DL-#14849 · Parameterize Remote Evidence Runners and Finish Batch 3 Host Redaction

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14849
- **Branch:** merged via #14849
- **PR:** #14849
- **Paths:** see #14849
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`c60c445c`; collated from changes/14849-parameterize-remote-evidence-runners-and.md)
- **Summary:** Parameterize remote evidence runners and finish batch 3 host redaction
- **Next step:** Shipped in PR #14849.

### DL-#14310 · Headless Qt/Matplotlib Teardown After Each Test and MuJoCo Offscreen Renderer Release (#14310)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14310
- **Branch:** merged via #15098
- **PR:** #15098
- **Paths:** see #15098
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`180394fd`; collated from changes/14310-headless-gui-teardown.md)
- **Summary:** Headless Qt/matplotlib teardown after each test and MuJoCo offscreen renderer release (#14310)
- **Next step:** Shipped in PR #15098.

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
- **Last verified:** 2026-11-15 (`d54c0c57`; collated from changes/15016-spec-input-map-manifest-honesty.md)
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
- **Last verified:** 2026-11-15 (`339be5a5`; collated from changes/15012-friction-cone-rl-honesty.md)
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
- **Last verified:** 2026-11-27 (`6996475c`; collated from changes/15010-flight-aero-impact-honesty.md)
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
- **Last verified:** 2026-11-15 (`318242c0`; collated from changes/14935-evidence-recompute-and-ratchets.md)
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
- **Last verified:** 2026-11-15 (`30f937bf`; collated from changes/14502-np09-lexical-fragment-guard.md)
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
- **Last verified:** 2026-11-15 (`e47a24aa`; collated from changes/12046-realtime-adr-split-ruling.md)
- **Summary:** Interim 30-day realtime shadow ledger sunset (#12046) so hygiene tests pass pending Tools migration
- **Next step:** Shipped in PR #14929.

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

### DL-#14227 · Document CI Optional Stack Slice 1 (Pinocchio Gravity) and Add Runbook Contract Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14227
- **Branch:** merged via #14923
- **PR:** #14923, #15089, #15158, #15176
- **Paths:** see #14923
- **Started:** 2026-11-10
- **Last verified:** 2026-11-27 (`cc10309b`; collated from changes/14227-optional-stack-scheduled-green.md)
- **Summary:** Document CI Optional Stack slice 1 (Pinocchio gravity) and add runbook contract tests
- **Next step:** Shipped in PR #14923.

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

### DL-#12553 · NSX-04 Slice 1: Verified Subtalar Inversion/Eversion and MTP Flexion Torque Capacities (da Fonseca 2025, Saeki 2021); Hip and MTP-Extension Slots Keep ASSUMED With Stated Reasons

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12553
- **Branch:** merged via #14684
- **PR:** #14684, #14690, #14696, #14704, #14713, #14725, #14787, #14925, #15064, #15174
- **Paths:** see #14684
- **Started:** 2026-11-05
- **Last verified:** 2026-11-27 (`316a2989`; collated from changes/12553-anatomical-torque-slots-acceptance.md)
- **Summary:** NSX-04 slice 1: verified subtalar inversion/eversion and MTP flexion torque capacities (da Fonseca 2025, Saeki 2021); hip and MTP-extension slots keep ASSUMED with stated reasons
- **Next step:** Shipped in PR #14684.

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
