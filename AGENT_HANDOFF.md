# Agent Handoff: Current-State Index

## Paused Study and Current Status

- **User-Requested Stop**: Goal pursuit stops for credit conservation. Start with [Resume Handoff](docs/research/shot_pattern_analysis/phase2/RESUME_HANDOFF.md). The existing local Python export is left running; no agent monitoring is scheduled.
- **Matched Delivery Study — Phase Two**: Full truthful current study handoff lives at [docs/research/shot_pattern_analysis/phase2/TURNOVER.md](docs/research/shot_pattern_analysis/phase2/TURNOVER.md) in the separate `shot-pattern-analysis` worktree on branch `codex/shot-pattern-matched-delivery`; the integration includes user-requested handoff checkpoint `dd2c235a0a`; later publication updates still need to be brought forward. Do not invent final acceptance.
- **Simulation and Exporter State**: All simulation data is complete; final publication remains pending. The frozen v6 supplementary graphic exporter is still live (PID 51475) in separate presentation checkout `4023111d2a`.
- **Graphics Qualification**: Original primary/control graphics are accepted; supplemental graphic review, report composition, remote asset delivery, and ready PR remain pending.
- **Integration Branch Status**: Branch `codex/shot-pattern-main-integration` contains reviewed local merge `7bdb71016c` of exact `main` `fa7ee92f5cddd753099d7418fc974b142b7ebb54` into `2953c52b37001a8a3893e07943fb3cf2b6c38b45`; model registries are preserved. Final publication and pre-PR gates remain pending.
- **Validation Evidence**: Root has independently executed 282 provider/scoring/GUI/report/contact tests (all pass), and Astra 71 launcher boundary tests pass (review recorded after fingerprint verification; context render/check/evaluate pass). Do not rely on terminal merge worker report.
- **External Blockers**: LaTeX editor compiler pre-parse bundle download failure (#12120) remains an external unresolved blocker.

## In-Flight Branch Stacking and Integration Dependencies

- **Phase 1 PR #11941**: Armed on branch `codex/shot-pattern-analysis`; do not touch it.
- **Integration Dependencies**: The resolved Tools pin is `86d0f28b1cc5acf61185e07e320c816c2d005512`; separate source-projection and runtime checks cover the study integration boundary.
- **Historical Science / Presentation Pin**: Historical science and frozen presentation retain the old `3678409` pin; no installs, receipt refreshes, or reruns.
- **Precondition Guard**: Guard `c7a2f41fc8` remains isolated on branch `codex/shot-pattern-protocol-preconditions` until the frozen final report is completed.

## Must-Read Architecture Pointers

- [docs/architecture/C4.md](docs/architecture/C4.md) — System context, container definitions, and architecture change log.
- [docs/agents/shared-infrastructure.md](docs/agents/shared-infrastructure.md) — Shared physics/FK modules, theme tokens, and embeddable tool contracts.
- [docs/research/shot_pattern_analysis/phase2/TURNOVER.md](docs/research/shot_pattern_analysis/phase2/TURNOVER.md) — Canonical Phase Two study turnover, execution receipts, and provenance logs.
- [CLAUDE.md](CLAUDE.md) and [AGENTS.md](AGENTS.md) — Repository governance, title-case rules, and the strict 150-line root handoff policy.

## Verification Gates and Commands

- `python scripts/check_document_title_case.py AGENT_HANDOFF.md docs/agents/handoffs/shot-pattern-integration-peer-records.md`
- `python -m src.tools.check_markdown_links AGENT_HANDOFF.md docs/agents/handoffs/shot-pattern-integration-peer-records.md`
- Use the exact scoped 282-test commands in `phase2/validation/main_integration_independent_runtime_v1/` and the 71-test boundary receipt in `phase2/validation/launcher_boundary_independent_review_v1/` (under the study research directory; included in this integration).
- Before delivery, run `python -m scripts.pre_pr` with Repository_Management first on `PYTHONPATH`; preserve all reported blockers.

## Do-Not List

- Do not touch armed Phase 1 PR #11941 on branch `codex/shot-pattern-analysis`.
- Do not terminate, signal, or modify live PID 51475 frozen v6 supplementary exporter in `4023111d2a`.
- Do not integrate guard `c7a2f41fc8` until the frozen final report is fully completed.
- Do not run installs, receipt refreshes, or reruns against historical science/frozen presentation pinned at `3678409`.
- Do not infer peer work completion, discard peer records, or claim independent validation for imported peer records.
- Do not invent final acceptance for Phase Two.

## Short-Term Roadmap and Next Steps

- Monitor completion of live exporter PID 51475 in frozen presentation checkout `4023111d2a`.
- Complete supplemental graphic review, report composition, and remote asset delivery.
- Resume only when authorized; bring forward final publication updates after study checkpoint `dd2c235a0a`.
- Retain completed root/Astra integration evidence; complete publication and final gates before a ready PR.

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
