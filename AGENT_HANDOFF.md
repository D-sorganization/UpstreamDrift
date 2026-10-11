# Agent Handoff: Current-State Index

## Active Study and Current Status

- **User-Requested Follow-Up**: PR [#12278](https://github.com/D-sorganization/UpstreamDrift/pull/12278) is open, ready for review, and blocked. The separate validation follow-up uses `codex/shot-pattern-validation-followup`. Local repairs and 168 focused passes are recorded in [Follow-Up Evidence](docs/research/shot_pattern_analysis/phase2/validation/pr12278_followup_v1/README.md); completed scoped pre-PR exited 1 (14 research failures and fleet-policy debt). No push or remote-head integration. Start with [Resume Handoff](docs/research/shot_pattern_analysis/phase2/RESUME_HANDOFF.md).
- **Matched Delivery Study — Phase Two**: All 10.44 million attempted simulation rows are complete. Do not rerun simulations. Read [Scientific Report](docs/research/shot_pattern_analysis/phase2/REPORT.md) and [Design](docs/research/shot_pattern_analysis/phase2/DESIGN.md).
- **Scientific Acceptance**: Astra accepted the report after 406 numeric/count checks and 52 source-hash checks. These are geometric, model-conditional results; no measured player benefit or universal best shot shape is established.
- **Report PDF**: The illustrated LaTeX research report is accepted for document delivery: 33 pages, 13 source-bound figures, zero Tectonic warnings. PDF SHA256 `ca030b3c97fb759b384aab58b3da6be68687a194f353976754d43e7ffa69beda`; review receipt `docs/research/shot_pattern_analysis/phase2/validation/illustrated_article_v1/review.json`. Local PDF: STUDY `docs/research/shot_pattern_analysis/phase2/validation/illustrated_article_v1/final_build/shot_pattern_analysis.pdf`; user-facing name: “Shot Pattern Analysis - Illustrated Research Report.pdf”. Source archive: `/tmp/shot-pattern-delivery-resume/illustrated-report-source.zip` (SHA256 `6567a9dcee60d6e18e4b1fe39ca34b3631eac09a578ce3eacd64914c93c7c1d2`).
- **Graphics**: All assigned baseline PNG reviews are complete. The 49 density-layout replacements (#12243) have root/Sol/Astra PNG acceptance and native PDF/SVG review. Coverage reconciliation accepts 2,304 pair PNGs plus 104 unchanged cell PNGs. All three archive bundles passed fresh local restoration.
- **Native Formats**: All 2,024 PDF and 2,024 SVG baseline renders completed. The 1,122 PDF bounding-box warnings were checked against painted pixels: 10,842 probes, minimum 6.4952 pt visible-ink clearance, no clipped edge ink. The renderer's final summary failed serialization; a separate successful finalizer preserved that distinction.
- **Integration and Publication**: The research prerelease is [published](https://github.com/D-sorganization/UpstreamDrift/releases/tag/shot-pattern-phase2-research-2026-10-10), bound to delivery checkpoint `cb5eff6dcd9df7d878f10578078b6fea69c8b681`. Its original payload is 21 inventory assets; two separate verification receipts were subsequently uploaded (23 release files total). Fresh download verification passed for the 21 payload assets, all three bundle member inventories, the 15-member illustrated source archive with 13 figures, and 98/98 report links. Receipt SHA256 `94f1fdea97207cb0eb8bf59db981d09cffd60d74d02f3ddd8957a3990e728843`: `docs/research/shot_pattern_analysis/phase2/validation/artifact_delivery_v3/remote-restoration-v3.json.gz`. The first restore attempt failed on archive-name mapping after asset hashes passed; Sol reviewed the corrected size/SHA bijection and the clean rerun exited 0 without asset changes. Attempt details: `docs/research/shot_pattern_analysis/phase2/validation/artifact_delivery_v3/remote_verification_summary.json`.
- **LaTeX Editor Blocker**: The local Tectonic 0.15.0 CLI compiled and visually reviewed the illustrated report successfully; the separate built-in Codex editor/compiler still fails before parsing because it cannot fetch its Tectonic bundle (#12120). Local CLI success does not clear the editor issue.

## In-Flight Branch Stacking and Dependencies

- **Phase 1 PR #11941**: Previously armed on `codex/shot-pattern-analysis`; do not touch its state.
- **Tools Pin**: Delivery uses `86d0f28b1cc5acf61185e07e320c816c2d005512`; source-projection and runtime checks cover the integration boundary.
- **Frozen Producer**: Presentation commit `4023111d2aaf8b20449a36f8691a295f4a94ceff` retains historical Tools pin `3678409`. Do not install, refresh receipts, or rerun it.
- **Artifact Routing**: Package `supplement_report_v2_layout_repair` but restore it as `supplement_report_v1` in the reader publication tree, preserving accepted report links and keeping the original study baseline intact.

## Must-Read Architecture Pointers

- [Architecture Map](docs/architecture/C4.md) and [Shared Infrastructure](docs/agents/shared-infrastructure.md).
- [Resume Handoff](docs/research/shot_pattern_analysis/phase2/RESUME_HANDOFF.md), [Turnover](docs/research/shot_pattern_analysis/phase2/TURNOVER.md), and [Artifact Delivery Plan](docs/research/shot_pattern_analysis/phase2/review_followups/ARTIFACT_DELIVERY_PLAN.md).
- [CLAUDE.md](CLAUDE.md) and [AGENTS.md](AGENTS.md) govern integration, issue closure, and PR delivery.

## Verification Gates and Commands

- `python scripts/check_document_title_case.py AGENT_HANDOFF.md`
- `python -m src.tools.check_markdown_links AGENT_HANDOFF.md`
- Root's 282 provider/scoring/GUI/report/contact tests and Astra's 71 launcher-boundary tests passed; exact receipts live under `phase2/validation/`.
- Canonical pre-PR ran at `dac6618c58`: Ruff, mypy, consistency, fragments, and workflow checks pass. Affected tests report 5,861 passes, 34 failures, and seven errors in unchanged outside-study test files; this is scope evidence, not runtime-baseline proof. Fleet development-log policy also fails. The subsequent XML parser fix passed 41 focused vector tests, Ruff, mypy, and the central import-policy check; Semgrep was unavailable and advisory-skipped. A later corrected-PYTHONPATH full pre-PR attempt at `cb5eff6dcd` terminated with exit 143 during tests before a final summary; it is incomplete and does not replace the failed complete gate.
- Focused baseline audit: all 41 original failed/error nodes span 18 unchanged paths. At the base commit, 31 failed and seven errored; three passed and also pass in isolated current-branch tests. The full-suite cause of those three remains unresolved. See `phase2/validation/baseline_gate_audit_v1/`.
- Verify archive manifests and every restored member before publication. Upload success alone is insufficient.

## Do-Not List and Next Steps

- Preserve frozen science, baseline graphics, peer records, and armed Phase 1 PR state.
- Do not claim player calibration, isolated causal curvature benefit, per-round strokes gained, or successful compilation in the built-in LaTeX editor.
- The 21-asset v3 inventory, published research prerelease, and fresh remote restoration are verified. PR #12278 is open and blocked with explicit unresolved repository gates; do not claim merge readiness or waive protection.
- Keep empirical validation #11984 deferred and compiler blocker #12120 open; close implementation issues only with qualifying merged evidence.

## Imported Peer Handoff Records Index

Imported peer record content is preserved in [docs/agents/handoffs/shot-pattern-integration-peer-records.md](docs/agents/handoffs/shot-pattern-integration-peer-records.md) (twelve relative links adjusted; original bytes retained in [the source archive](docs/agents/handoffs/shot-pattern-integration-peer-original.zip), original payload SHA256 `c1b9011feb789e7e949ff03adcace4f3eced7b79dfb97b0040c9f69e2079a845`). These records retain their reported status and source provenance from parent `2953c52b37001a8a3893e07943fb3cf2b6c38b45` and incoming `fa7ee92f5cddd753099d7418fc974b142b7ebb54`, without claiming independent validation or completion:

- [Active: MSK Retarget Turn Targets - #12042 Slice 4](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-msk-retarget-turn-targets---12042-slice-4)
- [Active: Drake IK Coordinate Order — #12042 Slice 5](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-drake-ik-coordinate-order--12042-slice-5)
- [Active: Shoulder, Trunk and Pelvis Turn Metrics - #12042 Slices 1-2](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-shoulder-trunk-and-pelvis-turn-metrics---12042-slices-1-2)
- [Active: Two-Hand Club in Tour-Matching Variants and Static Optimisation, OSV-9 Slice 2 #11756](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-two-hand-club-in-tour-matching-variants-and-static-optimisation-osv-9-slice-2-11756)
- [Active: Two-Hand Grip Pose - OSV-2 #11728 (Slice 1)](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-two-hand-grip-pose---osv-2-11728-slice-1)
- [Active: Gaze Weight Selection, #11729](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-gaze-weight-selection-11729)
- [Active: Hip Rewrite Uses the Spec's Own Pelvis Alignment - #12109](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-hip-rewrite-uses-the-specs-own-pelvis-alignment---12109)
- [Active: Golf Ball in Engine Scenes - GCV-13 #11719, Epic #11706](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-golf-ball-in-engine-scenes---gcv-13-11719-epic-11706)
- [Active: MyoSuite Hip Retarget - #12052 (Stacked On #12051)](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-myosuite-hip-retarget---12052-stacked-on-12051)
- [Active: Reference ZMP Split at the Ball Impact - #12117 (GCV-20 #11767)](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-reference-zmp-split-at-the-ball-impact---12117-gcv-20-11767)
- [Active: Clubhead Speed Timing — GCV-20 #11767, Epic #11706](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-clubhead-speed-timing--gcv-20-11767-epic-11706)
- [Active: Right Knee Flexes Negative Like the Left - #12057](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-right-knee-flexes-negative-like-the-left---12057)
- [Active: Grip Wrench Gaps - GCV-8 #11714](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-grip-wrench-gaps---gcv-8-11714)
- [Active: MuJoCo Native Export Backend - GCV-14 #11720](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-mujoco-native-export-backend---gcv-14-11720)
- [Active: OpenSim Contact Grip, OSV-7 Phase 4 #11739](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-opensim-contact-grip-osv-7-phase-4-11739)
- [Active: Quarter-Speed Export Variants for Every Engine - GCV-14 #11720](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-quarter-speed-export-variants-for-every-engine---gcv-14-11720)
- [Active: Same-Input Parity Design-Manual Chapter - SIP P-9 #11614](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-same-input-parity-design-manual-chapter---sip-p-9-11614)
- [Active: Contact Grip and MyoSuite Bushing, OSV-7 Phase 3 #11739](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-contact-grip-and-myosuite-bushing-osv-7-phase-3-11739)
- [Active: Lead-Foot Address Corrections - OSV-6 #11737 (Stacked On #12031)](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-lead-foot-address-corrections---osv-6-11737-stacked-on-12031)
- [Active: Mirrored Left Hip Axis - OSV-6 #11737](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-mirrored-left-hip-axis---osv-6-11737)
- [Active: Lead-Foot Address Corrections - OSV-6 #11737 (Stacked On #12031)](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-lead-foot-address-corrections---osv-6-11737-stacked-on-12031-1)
- [Active: Mirrored Left Hip Axis - OSV-6 #11737](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-mirrored-left-hip-axis---osv-6-11737-1)
- [Active: Native Export Impact Time From Ball Passage - GCV-14 #11720](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-native-export-impact-time-from-ball-passage---gcv-14-11720)
- [Active: Shared Club in the Musculoskeletal OpenSim Models, OSV-9 #11756](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-shared-club-in-the-musculoskeletal-opensim-models-osv-9-11756)
- [Active: Native Platform Integration — F10i #12060](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-native-platform-integration--f10i-12060)
- [Prior: Prediction Integrity Consolidation — F09p #12055](docs/agents/handoffs/shot-pattern-integration-peer-records.md#prior-prediction-integrity-consolidation--f09p-12055)
- [Prior: Native Tools Integration — F09o #12044](docs/agents/handoffs/shot-pattern-integration-peer-records.md#prior-native-tools-integration--f09o-12044)
- [Prior Native Command Solve — F09n #12036](docs/agents/handoffs/shot-pattern-integration-peer-records.md#prior-native-command-solve--f09n-12036)
- [Prior Native Tracking Objective — F09m #12023](docs/agents/handoffs/shot-pattern-integration-peer-records.md#prior-native-tracking-objective--f09m-12023)
- [Active: Persistent Project MyoSuite Native Search — F09l #12014](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-persistent-project-myosuite-native-search--f09l-12014)
- [Active: Project MyoSuite Feedback Recording — F09j #12003](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-project-myosuite-feedback-recording--f09j-12003)
- [Active: Project MyoSuite Task and Independent Replay — F09i #11983](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-project-myosuite-task-and-independent-replay--f09i-11983)
- [Active: Same-Input Bushing Grip Parity, OSV-7 Phase 2 #11739](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-same-input-bushing-grip-parity-osv-7-phase-2-11739)
- [Active: Simscape URDF Exchange — #11569 Task 3](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-simscape-urdf-exchange--11569-task-3)
- [Active: GRF Engine Wiring and CoP Consolidation - GCV-2 #11708, GCV-6 #11712](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-grf-engine-wiring-and-cop-consolidation---gcv-2-11708-gcv-6-11712)
- [Active: Native OpenSim Geometry — #11903](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-native-opensim-geometry--11903)
- [Active: MeshCat Camera Framing - NV-9 #11697](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-meshcat-camera-framing---nv-9-11697)
- [Active: Run-102 Simscape Playback MP4 — #11569 Task 4](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-run-102-simscape-playback-mp4--11569-task-4)
- [Active: Muscle Qualification Evidence Guard — F07 #11791](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-muscle-qualification-evidence-guard--f07-11791)
- [Active: Native Muscle-Model Asset Admission — #11819](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-native-muscle-model-asset-admission--11819)
- [Active: Feedback Controls Planning — #11784](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-feedback-controls-planning--11784)
- [Active: Impact-Phase Bushing Grip, OSV-7 #11739](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-impact-phase-bushing-grip-osv-7-11739)
- [Active: Ground Reaction Design Manual Slice - GCV-18 #11724](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-ground-reaction-design-manual-slice---gcv-18-11724)
- [Active: Club Force and Torque Overlays - GCV-10 #11716](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-club-force-and-torque-overlays---gcv-10-11716)
- [Active: Lift Pack Parity Baseline - LIFT-1 #11741, Epic #11740](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-lift-pack-parity-baseline---lift-1-11741-epic-11740)
- [Active: Shared Ground Reaction Core — GCV-1, Epic #11706](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-shared-ground-reaction-core--gcv-1-epic-11706)
- [Active: Force Arrow Scaling — GCV-4, Epic #11706](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-force-arrow-scaling--gcv-4-epic-11706)
- [Active: Club-Face Orientation Residual, OSV-10 #11759](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-club-face-orientation-residual-osv-10-11759)
- [Active: Clubface Roll at Address, OSV-8 #11755](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-clubface-roll-at-address-osv-8-11755)
- [Active: Visible Head and Neck (GCV-12) - #11718](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-visible-head-and-neck-gcv-12---11718)
- [Active: Head Gaze Stabilisation - #11729](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-head-gaze-stabilisation---11729)
- [Active: Compliant Bushing Grip Model - #11739](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-compliant-bushing-grip-model---11739)
- [Active: Impact Parameters Panel - GCV-17 (#11723)](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-impact-parameters-panel---gcv-17-11723)
- [Active: ClubheadSeries Engine Adapters - GCV-16 (#11722)](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-clubheadseries-engine-adapters---gcv-16-11722)
- [Active: Centroidal Feasibility Filter V2 — #11669](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-centroidal-feasibility-filter-v2--11669)
- [Active: High-FPS Video Frame Schedule — GCV-14, Epic #11706](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-high-fps-video-frame-schedule--gcv-14-epic-11706)
- [Active: Shared Grip Wrench Core — #11713](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-shared-grip-wrench-core--11713)
- [Active: Finish Feasibility Balance - #11667](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-finish-feasibility-balance---11667)
- [Active: Address Foot Progression — #11730 (OSV-4, Epic #11726)](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-address-foot-progression--11730-osv-4-epic-11726)
- [Active: MyoFullBody Muscle-Driven Swing, Epic #11642](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-myofullbody-muscle-driven-swing-epic-11642)
- [Active: MOSAIC Model-Aware Matching — #11532](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-mosaic-model-aware-matching--11532)
- [Active: OpenCap to OpenSim Integration — #11400](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-opencap-to-opensim-integration--11400)
- [Active: Necromatcher Native Fit Delivery — #11240](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-necromatcher-native-fit-delivery--11240)
- [Necromatcher Workspace Handoff — #11239 (Merged)](docs/agents/handoffs/shot-pattern-integration-peer-records.md#necromatcher-workspace-handoff--11239-merged)
- [Historical Player Capture Handoff](docs/agents/handoffs/shot-pattern-integration-peer-records.md#historical-player-capture-handoff)
- [Motion-Matching Handoff](docs/agents/handoffs/shot-pattern-integration-peer-records.md#motion-matching-handoff)
- [Source-Preserving BUET–Hamner Native Candidate - #12157](docs/agents/handoffs/shot-pattern-integration-peer-records.md#source-preserving-buethamner-native-candidate---12157)
- [Active: Assistance-Explicit OpenSim Replay, F07 #12147](docs/agents/handoffs/shot-pattern-integration-peer-records.md#active-assistance-explicit-opensim-replay-f07-12147)
