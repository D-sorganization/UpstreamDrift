# Shot-Pattern Integration Peer Records

> **Provenance Notice**: Complete upstream peer records from `fa7ee92f5cddd753099d7418fc974b142b7ebb54:AGENT_HANDOFF.md`, imported during integration with study parent `2953c52b37001a8a3893e07943fb3cf2b6c38b45`. Reported peer status is preserved without independent validation. The displayed text changes only twelve relative Markdown links for relocation. The [original byte-identical payload](shot-pattern-integration-peer-original.zip) has SHA-256 `c1b9011feb789e7e949ff03adcace4f3eced7b79dfb97b0040c9f69e2079a845`. The study section is maintained in the [current root index](../../../AGENT_HANDOFF.md).

---

# Active: MSK Retarget Turn Targets - #12042 Slice 4

- Branch `claude/msk-turn-targets` (on `claude/turn-metrics-core`, #12068). `msk_club_tracking.track_swing(..., turn_targets=, feet=)` adds pelvis/upper-trunk yaw residuals (0.02 rad, relative to the calibrated address heading) from the shared marker turn lines, and feet planted from the capture's foot markers (`msk_turn_targets.py`). Opt-in; the default tracking and renders are unchanged.
- Capture A with targets + feet: pelvis 40.6/-49.7 (markers 40.9/-49.8), upper trunk 95.8/-19.1 (92.2/-19.2). Costs: trail-hand gap 15 -> 50 mm (early downswing), lead grip 1.3 -> 4.3 mm. Shoulder girdle is 7-10 deg short (no Rajagopal scapula). `scripts/msk_turn_audit.py` reproduces; capture B needs an uncommitted iron7 grip calibration.
- Next: wire targets into `render_msk_club.py` (needs the private capture path), slice 7 (source trunk over-twist) to remove the trail-gap trade-off.

# Active: Drake IK Coordinate Order — #12042 Slice 5

- Branch `claude/drake-ik-twist`. Root cause of the Drake thorax under-twist: Drake's plant position order is not the spec `coordinate_order` (spine/torso/arms/neck permuted). `DrakeFullBodyIK._set` and the model's array paths skipped the permutation, so the twist was solved under the `SpineInputX` ±35° bound. Fixed with `to_plant_positions`/`named_columns` on `FullBodyDrakeModel`; tests in `tests/unit/motion_matching/test_drake_coordinate_order.py`.
- Capture-A receipt `evidence/ground_support/anthro_driver_drake` regenerated: IK 47.2 → 34.6 mm, FD 382 → 60.1 mm; the twist now matches MuJoCo (inherits MuJoCo's +16°/+20° top over-twist, slice 7). Calc record: `simscape_matching_reference.tex`, section "Drake Full-Body IK Coordinate Order".
- Open: shared spec FK misplaces the `Clubhead` triad by about 1 m versus Drake/MuJoCo (do not use it for club markers).

# Active: Shoulder, Trunk and Pelvis Turn Metrics - #12042 Slices 1-2

- Branch `claude/turn-metrics-core`; refs #11726. `swing_comparison/turn.py` splits the old "thorax yaw" into `shoulder_girdle` (ShoulderBack, scapular) and `upper_trunk` (BackLeft/BackRight) lines, keeps `pelvis`, and reports X-factor as upper trunk minus pelvis (plus a shoulder-girdle variant). Turn is relative to address, + = backswing, unwrapped. Gaps over 0.10 s stay NaN with a reason; unavailable is never zero. `SegmentRotationMetrics.thorax_yaw*` is a deprecated alias (shoulder girdle).
- Model side: `spec_model_points` (shared FK; hip centres, `LS`/`RS`, thorax attachments) or model marker sites. Receipt block: `motion_matching/turn_receipt.py` (`turn_block/v1`), wired into the MuJoCo pipeline (`pipeline/cli.py`) and Pinocchio (`full_body_fit.py`); schema check in `receipt_schema.validate_receipt`. No thresholds (slice 8 gates).
- Follow-ups: Drake, OpenSim, MyoSuite, MuJoCo replay, Simscape writers (`FOLLOW_UP_WRITERS`, with reasons). Next: slice 3 (OSV-6 #11737). Definitions: `simscape_matching_reference.tex`, section on turn definitions.

# Active: Two-Hand Club in Tour-Matching Variants and Static Optimisation, OSV-9 Slice 2 #11756

- Branch `claude/osv-9b-msk-club-variants`. `tour_matching/model_variants.equipment_from_model` reads the `Club` the model carries (shared-club hash, spec mass, weld/bushing topology, both hands, all meshes) and fails closed; both variant factories use it.
- `run_musculoskeletal_swing.py --club driver --club-control`: SO with the club plus a no-club control. The IK does not close the two-hand loop (with the trail weld, 14 arm coordinates failed at every frame tried), so SO releases the trail weld (`release_trail_weld`; `--enforce-trail-weld` keeps it). Per-frame convergence is recorded from the native SO log (violation <= 1e-3): 234/262 with the club, 237/262 without.
- Club load on 225 jointly converged frames: lead wrist deviation +32.9 N m RMS, lead shoulder flexion +45.0, trail arm exactly 0. Receipt `evidence/musculoskeletal/receipt.json`; the ledger is regenerated. No MyoFullBody OpenSim export exists.

# Active: Two-Hand Grip Pose - OSV-2 #11728 (Slice 1)

- Branch `claude/osv-2-grip-pose` (PR #12032), merged with main after OSV-9 (#11795). `model_appearance/grip_pose.py` is the club-agnostic grip definition; its lead (3.2 cm below the butt) and trail (10.82 cm) positions mirror the generated-spec `GripInterface`, the single source, and `tests/opensim/test_grip_closure_both_hands.py` fails on drift. `club_geometry`, `address.py` and the MyoSuite scene sites import it. The Rajagopal models keep the OSV-9 topology (lead `WeldJoint hand_l_to_club`, trail `WeldConstraint hand_r_to_club`); `msk_club_grip_calibration.json` is untouched.
- Canned-swing closure residual (`python3 -m scripts.grip_closure_report --output r.json`): MuJoCo, Drake (ControlTower), Pinocchio, OpenSim full-body weld <= 1.8e-7 m; MyoSuite unavailable. Open: owner review of a 1.5 cm lead position (needs `msk_club_calibration` rerun for both models and clubs), finger meshes, stills and clips (slice 2), no `golf_humanoid` swing residual.

# Active: Gaze Weight Selection, #11729

- Branch `claude/osv-3b-gaze-default`. `motion_matching/gaze_sweep.py` (feasible set, Pareto knee, cross-capture `select_default`, `REPORTING_GAZE_WEIGHT = 0.1`), `scripts/sweep_gaze_weight.py` (IK-stage run per capture and weight), `scripts/summarize_gaze_sweep.py` (evidence JSON + Pareto plot), `scripts/render_head_gaze_clips.py` (MuJoCo side by side, 1080p60, 1x/0.5x/impact 0.25x), `scripts/render_head_gaze_engines.py` (Drake/Pinocchio/OpenSim/MyoSuite native clips + pairing).
- Result: knee 0.1 in both captures (common feasible {0, 0.1, 0.2}); theta_gaze RMS 20.2 to 8.5 deg (driver), 18.6 to 6.1 deg (iron), marker RMS +0.6/+1.3 mm. `--gaze-weight` still defaults to 0 for qualified receipts. Evidence `docs/development/full_body_models/evidence/head_gaze/gaze_weight_sweep.{json,png}`; method in `HEAD_GAZE_REFERENCE.md` (Gaze Weight Selection) and the `.tex`.
- Open: neck PD tracking in forward dynamics, published tour head ranges.

# Active: Hip Rewrite Uses the Spec's Own Pelvis Alignment - #12109

- Branch `claude/trail-hip-12109`, stacked on #12113. `cli._calibrate_and_scale` took `hip_from_opensim_pelvis` from `build_receipt_v2.json` for every spec, so `apply_hip_calibration` rotated the anthro hips (femur 24-66 deg off the pelvis at zero coordinates) and pinned `hip_rotation` at +-40 deg. `hip_calibration.pelvis_alignment_from_spec` now recovers the alignment from the spec's own hips (reproduces the v1/v2 receipts exactly) and fails closed when the hips disagree.
- Address (MuJoCo, foot progression on): toe-out trail/lead -0.11/0.22 deg (driver) and -0.05/0.19 deg (7-iron); `hip_rotation` -17/+10 and -24/+18 deg. Evidence `evidence/foot_progression/pelvis_alignment_12109/`; diagnostic `scripts/diagnose_address_leg_yaw.py`. Reference: `docs/research/hip_axis_mirroring/` (Hip Rewrite Pelvis Alignment section).
- Every calibrated anthro run changes: the canonical ground-support receipts are regenerated in this branch. Likely also the lead-hip limit in #12042 slice 3 / #11737.

# Active: Golf Ball in Engine Scenes - GCV-13 #11719, Epic #11706

- Branch `claude/gcv-13-ball-scenes`. `src/shared/python/model_appearance/ball.py` (landed in #11772) already had the regulation-radius address-geometry function; this slice adds `resolve_ball_visual` (enabled/override/computed, "unavailable" with a reason rather than a guess) and a `ball` block (`enabled`, `position_m`, `source` in `BALL_SOURCES = (address_geometry, measured, model_estimate)`) on `AppearanceDocument`/`appearance_v1.schema.json`.
- MuJoCo: `visual_layer.attach_visual_layer` now attaches a massless, non-colliding `visual_ball` sphere geom fixed to the world body whenever the spec has a club (`_attach_decorative_ball`/`_club_face_world`). An explicit `appearance.ball.position_m` (e.g. a mocap-matched swing's measured ball) is trusted verbatim; the computed address-geometry fallback is only used when the exporter's static reference-pose face centre is plausibly grounded (within 0.5 m) — it does **not** drive the model to a true biomechanical address qpos (that needs a per-swing pose this static exporter does not have), so for the real `full_body_spec_anthro_*.json` fixtures (reference pose is a rest stance, not address) the ball reports `{"enabled": False, "reason": "..."}` until a caller supplies `position_m` from a resolved address pose.
- Open (next slices, in order the issue prefers): (b) wire a resolved address/measured ball position through the native-export/overlay frame path so every engine's rendered video gets the ball; (c) Drake/Pinocchio MeshCat, OpenSim, MyoSuite visual layers; (d) web `GolferModel.tsx`. Also open: GCV-15 impact detection to drive the post-impact hide/launch behaviour (ball currently always shown pre-impact, by construction never moves). Feature-parity entry `render.club_head_and_ball` (#11717) notes updated; still `gap` until web + the other engines land.

# Active: MyoSuite Hip Retarget - #12052 (Stacked On #12051)

- Branch `claude/osv-6e-myosuite-hip-retarget`. `myosuite/python/hip_retarget.py` maps each femur's orientation relative to the pelvis from the fitted (hip-calibrated) spec into the `myolegs` Z-X-Y hip hinges; `retarget_frame/retarget_trajectory(..., hip_spec=spec)` and `ReplayConfig.hip_spec` use it. The map's `knee_angle_l` sign is now -1.
- MyoSuite address toe-out error, lead/trail: driver +0.12/-0.37 deg, 7-iron -0.04/-1.46 deg (was -45/+33, -40/+36). Evidence: `docs/development/full_body_models/evidence/foot_progression/osv6_myosuite/`. Open: #12057, where the spec's right knee flexes positive but its range caps it at +10 deg (canonical IK sits on the bound).

# Active: Reference ZMP Split at the Ball Impact - #12117 (GCV-20 #11767)

- Branch `claude/zmp-impact-split-12117`, stacked on #11960 (GCV-20). `reference_zmp(..., split_time_s=)` uses `impact_force.reference_rates`; the pipeline calls `pipeline.dynamics.lane_reference_zmp` (passes `lane.impact_time_s`). Without it the ZMP of frames impact-1..+2 was 4-29 m outside the feet and the cart-table shift dragged the 7-iron reference off its IK over the downswing.
- 7-iron on GCV-20 + #12125: dynamics RMS 264 -> 54.5 mm, export no longer diverges, impact speed -10.2 % (reference +0.9 %, replay -9.2 %). Driver unchanged (-6.1 %; no ZMP filter). Acceptance still open; no tolerance changed.
- Next: computed-torque replay speed loss through the release; regenerate club-face fixtures after #11960 and #12125 merge. Reference: calc reference, GCV-20 "Third Pass".

# Active: Clubhead Speed Timing — GCV-20 #11767, Epic #11706

- Branch `claude/gcv-20-speed-timing`. Ball contact impulse in every dynamics replay (`impact_parameters` collision), reference low-pass split at impact, release-preserving pre-contact cutoff (`pipeline/release_cutoff.py`, 25 Hz for both captures) and grip-weld projection of the tracked reference (`pipeline/weld_projection.py`).
- Full re-solve at `5f8ee41fe5`: peak timing passes (driver -3.7 ms, 7-iron -4.2 ms; limit 5 ms); impact speed does not (driver -7.4 %, 7-iron -4.1 %; limit 3 %). `test_clubhead_speed_peaks_with_the_capture_and_matches_it_at_impact` is a strict xfail with those numbers; tolerances unchanged; fixtures not regenerated.
- Next: the remaining gap is the unactuated root's ground yaw slip (DESIGN_DECISIONS decision 13). It needs a foot yaw-moment / contact model change, tracked outside GCV-20. Details: calculation reference, GCV-20 "Second Pass" and "Full Re-Solve".

# Active: Right Knee Flexes Negative Like the Left - #12057

- Branch `claude/knee-r-axis-12057`. `spec_builder.leg_extension` mirrors any knee whose hinge points to -z (`AXIS_MIRROR`), and `scripts/mirror_spec_right_knee.py` fixed the committed anthro specs. Spec-to-Rajagopal knee sign is -1 on both sides, and MyoSuite `knee_angle_r` is -1. The generated osims and the inventory hashes are refreshed.
- Canonical receipts were regenerated on ControlTower; a control at the parent commit reproduced the old receipt exactly. Calibrated driver 7.7/33.0/61.6 -> 5.6/31.0/77.0 mm, 7-iron 6.5/31.1/47.0 -> 5.0/28.8/71.2 mm. The trail knee at address is now -27/-21 deg (was pinned near 0).
- Open: the dynamics replay regresses (#12110). Trail toe-out is -2.1/-4.4 deg with `hip_rotation_r` at its -40 deg limit (#12109). Reference: `docs/research/hip_axis_mirroring/` (Right Knee Axis Convention section).

# Active: Grip Wrench Gaps - GCV-8 #11714

- Branch `claude/gcv-8-grip-wrench-gaps`. The MuJoCo grip-modelling tab shows the `grip_weld_l/r` wrench via `grip_efc` and reports "unavailable" for contact-only scenes. Drake URDF: `humanoid_urdf.right_hand_grip_analysis` gives the right-hand wrench from club Newton-Euler (`full_body_model.club_newton_euler`); the left hand is unavailable, with a reason. Pinocchio: `PinocchioForceAdapter.grip_analysis_from_allocation` routes `lambda_grip` through `grip_from_allocation`.
- Open: `allocate_trajectory` does not store `q`, so callers pass it themselves. MyoSuite per-hand emission needs the pinned `myo_sim`.

# Active: MuJoCo Native Export Backend - GCV-14 #11720

- Branch `claude/gcv-14-mujoco-backend`, stacked on PR #12026 (`claude/gcv-14-speed-variants`). `native_viewer_export` has a fifth engine, `mujoco` (`backends/mujoco_native.py` + `mujoco_worker.py`): full-body MJCF with the default appearance document (body, head, club meshes, scene) rendered by `mujoco.Renderer` through the shared view presets and glyph overlay. It replaces the earlier throwaway stick-figure evidence renderer.
- Tests: `tests/unit/tools/native_viewer_export/test_mujoco_backend.py` (registration, `unavailable_reason()`, appearance/club scene, `requires_gl` render); the speed-variant frame-count and 0 ms HUD test iterates `ENGINES`, so it covers MuJoCo.
- Evidence: ControlTower renders `gcv14/mujoco_native` (`MUJOCO_GL=osmesa`), see the PR body for frame counts and the impact frame.

# Active: OpenSim Contact Grip, OSV-7 Phase 4 #11739

- Branch `claude/osv-7-opensim-contact` (stacked on #11985). `grip_model="contact"` in `export_full_body_osim` builds pad `ContactSphere`s against closed `ContactMesh` cylinders with one `ElasticFoundationForce` per pad (`opensim/python/full_body_grip_contact.py`, `grip_contact_osim_sim.py`, meshes in `grip_contact/grip_mesh.py`). Weld stays default. CPodes is required (RK-Merson needs about 1e-7 s steps); the club is released with the weld velocity.
- Full swings (CT, accuracy 1e-8): net force peak -2.6 % (driver) and -3.1 % (7-iron) against the bushing; per-hand and internal pairs differ (hyperstatic, section 19). Evidence `evidence/grip_kinetics/contact/opensim_*_acc1e-8_*`. Slow tests run on CT with `-m 'slow or not slow'`.

# Active: Quarter-Speed Export Variants for Every Engine - GCV-14 #11720

- Branch `claude/gcv-14-speed-variants`, stacked on PR #12004 (kinematic impact time). `video_timing.frame_schedule.SPEED_VARIANTS = (1, 0.5, 0.25)` is the default of `ExportSettings.speeds` and `--speeds`; `motion_matching.export.export_video_variants` writes the same three clips for any engine's candidate (MuJoCo, MyoSuite, Drake, Pinocchio, OpenSim, Simscape logs).
- Tests: `test_speed_variants_engines.py` (frame counts from the time span for all four native engines, HUD 0 ms at the ball-passage impact) and `test_export.py` (variants per engine).
- Evidence: only MuJoCo renders on ControlTower (stick figure, no club, 1920x1080, 60 fps). Drake and Pinocchio need `playwright` and its Chromium libraries, OpenSim needs `xwd`, MyoSuite needs `myosuite.envs.env_base` (#11997); none are in the `ud-sim` image and installing them needs root or a new image (owner action).

# Active: Same-Input Parity Design-Manual Chapter - SIP P-9 #11614

- Branch `claude/sip-p9-manual-chapter-11614`; epic #11605. Provisional QMD chapter `manuals/upstreamdrift/chapters/16-same-input-parity.qmd` (renumbered from 14 after `14-muscle-qualification-evidence.qmd` landed) (KKT, closure projection, ZOH RK4, bundle, L0-L3 levels and results, failed experiments, limitations) and registry blocker `UP-D1-same-input-parity-inventory`; registry stays blocked with no calculations. `same_input_parity.tex` cross-links the chapter and compiles with pdflatex.
- Open: Simscape P-8 (#11613) must update both documents when it lands.

# Active: Contact Grip and MyoSuite Bushing, OSV-7 Phase 3 #11739

- Branch `claude/osv-7-contact-grip` (stacked on #11963). Pad contact grip with pad stiffness matched to the bushing (`grip_contact/pad_layout.py`, `pad_contact.py`); MuJoCo full swing (dt 1e-5 s), Drake and Pinocchio holds; MyoSuite bushing parity passes (worst 0.013 % peak). Quasi-static balance closes (`static_balance.py`). See GRIP_PARITY_DECISIONS.md section 19.
- Open and flagged: contact per-hand force is about 4x the bushing's with matching net force (indeterminacy, not tuned); no OpenSim contact variant; no full-swing Drake or Pinocchio contact run.
- #11986 (branch `claude/grip-contact-internal-force`, stacked on `claude/osv-7-contact-grip`): the 4x is not input drift (hand-to-hand drift 4e-13 mm); it is the hyperstatic internal force of two stiff rings (26 to 44 N per micron of mismatch). Friction softening loses the grip. Section 19 "Why the Contact Grip Carries 4 Times". No default changed.

# Active: Lead-Foot Address Corrections - OSV-6 #11737 (Stacked On #12031)

- Branch `claude/osv-6b-leg-azimuth`. With `--foot-progression` on, two corrections apply. `hip_calibration.knee_flexion_axis` removes the lateral-marker tilt from the zero twist (it was biased by 17-43 deg). `pipeline/leg_marker_constraints.py` keeps the knee and ankle marker azimuth and keeps the forefoot square and centred.
- MuJoCo address toe-out is within 2 deg on both captures, and so is Drake once the #12039 coordinate-order fix (PR #12047) lands. Pinocchio and MyoSuite evidence is on the stacked branch `claude/osv-6c-feet-evidence`. Canonical receipts are unchanged: the default-on correction is a recorded failed experiment (see `docs/research/hip_axis_mirroring/`). The finish-feasibility ratchet is strict-xfail on #12040.

# Active: Mirrored Left Hip Axis - OSV-6 #11737

- Branch `claude/osv-6-hip-mirror`. `spec_builder.hip_axis_signs` mirrors left hip adduction and rotation like OpenSim; specs regenerated by `scripts/mirror_spec_left_hip.py`. Calibrated receipts: driver 7.7/33.0/61.6 mm, 7-iron 6.5/31.1/47.0 mm. Nominal receipts were regenerated, and the old ones moved to `evidence/ground_support/anthro_*_pre_osv6/`. Reference: `docs/research/hip_axis_mirroring/`.
- The nominal 7-iron trajectory IK (269 mm) fails on `main` without the mirror too, so the bug is pre-existing (#12030). Open for OSV-6: the lead foot is still 15-32 deg out at address in MuJoCo and Drake (`hip_rotation_l` at its limit), the heel-proxy malleolus correction, and the Pinocchio/MyoSuite stills.

# Active: Lead-Foot Address Corrections - OSV-6 #11737 (Stacked On #12031)

- Branch `claude/osv-6b-leg-azimuth`. With `--foot-progression` on, two corrections apply. `hip_calibration.knee_flexion_axis` removes the lateral-marker tilt from the zero twist (it was biased by 17-43 deg). `pipeline/leg_marker_constraints.py` keeps the knee and ankle marker azimuth and keeps the forefoot square and centred.
- MuJoCo address toe-out is within 2 deg on both captures, and so is Drake once the #12039 coordinate-order fix (PR #12047) lands. Pinocchio and MyoSuite evidence is on the stacked branch `claude/osv-6c-feet-evidence`. Canonical receipts are unchanged: the default-on correction is a recorded failed experiment (see `docs/research/hip_axis_mirroring/`). The finish-feasibility ratchet is strict-xfail on #12040.

# Active: Mirrored Left Hip Axis - OSV-6 #11737

- Branch `claude/osv-6-hip-mirror`. `spec_builder.hip_axis_signs` mirrors left hip adduction and rotation like OpenSim; specs regenerated by `scripts/mirror_spec_left_hip.py`. Calibrated receipts: driver 7.7/33.0/61.6 mm, 7-iron 6.5/31.1/47.0 mm. Nominal receipts were regenerated, and the old ones moved to `evidence/ground_support/anthro_*_pre_osv6/`. Reference: `docs/research/hip_axis_mirroring/`.
- The nominal 7-iron trajectory IK (269 mm) fails on `main` without the mirror too, so the bug is pre-existing (#12030). Open for OSV-6: the lead foot is still 15-32 deg out at address in MuJoCo and Drake (`hip_rotation_l` at its limit), the heel-proxy malleolus correction, and the Pinocchio/MyoSuite stills.

# Active: Native Export Impact Time From Ball Passage - GCV-14 #11720

- Branch `claude/gcv-14-impact-evidence`. `native_viewer_export.overlay.detect_impact_time_s` called `model_appearance.club_face.impact_frame` and reported that frame's exact timestamp; on the committed driver fixture the accepted peak-speed sample sits 9.2 cm above and 9.2 cm from address (the height check alone let it through), not the true closest approach. It now returns `club_face.ball_passage`'s sub-sample `t_impact` on the same `Clubhead`-origin trajectory (one detector, OSV-10); docstring and README (`src/tools/native_viewer_export/README.md`) updated.
- Kept the `Clubhead` frame origin rather than the true face-centre point (`club_assembly.clubface_centre`): deriving the centre here would need the club spec resolved into a `ClubAssembly` and composed with the frame's rotation, a larger change than this fix's scope.
- Tests added/updated in `tests/unit/tools/native_viewer_export/test_grip_tracking.py`: `test_hud_impact_time_matches_ball_passage` (basic contract), `test_hud_impact_time_uses_ball_passage_not_the_peak_speed_sample` (reproduces the bug with a synthetic overshoot trajectory), `test_hud_impact_time_raises_when_the_head_never_returns_to_the_ball`, and the capture-A fixture test now compares against `ball_passage` instead of `impact_frame`.

# Active: Shared Club in the Musculoskeletal OpenSim Models, OSV-9 #11756

- Branch `claude/osv-9-msk-club`, epic #11726. `golf_humanoid.osim`, `golf_humanoid_scaled.osim` and the muscle model (`musculoskeletal_swing.build_musculoskeletal_model`) hold the shared club (`msk_club.py`: spec mass/inertia, shared STLs, `GripInterface` grips) in both hands: lead `WeldJoint hand_l_to_club` plus trail `WeldConstraint hand_r_to_club`, or `--grip-model bushing`.
- Hand frames and address pose: `msk_club_calibration.py` (IK to the generated model's FK at the captured address; feet planted because the generated feet are unobserved) into `models/msk_club_grip_calibration.json`; the muscle model is calibrated on build. `msk_club_tracking.track_swing` follows a generated swing; `scripts/render_msk_club.py` renders under xvfb.
- Base models are not in the clone (submodule fetch failed): set `UPSTREAMDRIFT_RAJAGOPAL_OPENSENSE` / `UPSTREAMDRIFT_MSK_BASE_MODEL`.
- Open: no MyoFullBody OpenSim export exists; static optimisation not run; scaled-model address fit is weaker (wrists at their bounds); `opensim_golf/fk.py` `State.isValid` failure predates this.

# Active: Native Platform Integration — F10i #12060

Read `docs/development/feedback_controls/F10I-NATIVE-PLATFORM-INTEGRATION-TURNOVER.md`.
The guarded controls and published native platform implementations now coexist.
Preserve Tools 86d0f28, private contract class identity, preparation blockers and
Astra's full-model qualification firewall. Actual runtime evidence and broad
capture-runtime failures are separate; integration does not complete the goal.

# Prior: Prediction Integrity Consolidation — F09p #12055

Branch `feat/f09p-prediction-integrity-12055` begins at published F09o
`98784e8374f3adf31f3b22d6bb9e54c0bfa3d081`. Read its plan and turnover.
Final preparation verification follows conversion and bundle validation, rejects
changed source/model/handles/callback/live state before stepping, and retains
complete provider-byte checks at entry/pre-execution/exit. No cross-call cache.
Controlled native short-fixture ABAB measurements retain identical state digests
and 25 percent fewer native-library bytes read. No full-capture speed claim.
Preserve original source-specific receipts; all seventeen rows, six ecosystems
and private muscular OpenSim matching plus independent replay remain required.

# Prior: Native Tools Integration — F09o #12044

Branch `feat/f09o-native-tools-integration-12044` integrates F09n `df9f3b9538`
with the admitted F01b/Tools/private-consumer stack `d232efcc69`. Read
`docs/development/feedback_controls/F09O-NATIVE-TOOLS-INTEGRATION-PLAN.md`.
Tools `86d0f28b1cc5acf61185e07e320c816c2d005512` is the admitted pin. Native
contracts use the explicit private package loader; retain one class identity.
The prior native 170-test receipt applies to its older source/pin. Fresh actual
MyoSuite 3.0/MuJoCo 3.6 evidence passes 170 tests with zero skips in 208.34s;
50 integration files and 8,122 tracked Tools files match before/after. The
separate MuJoCo 3.8 consumer lane passes 81 with three explicit optional skips.
Impact capability/served verification precedes metadata reconciliation; all
27 matrix tests pass. Read the F09o turnover for exact source/evidence scope.
Do not infer controlled speed improvement or scientific acceptance.
All seventeen model rows, six ecosystems and muscular OpenSim remain required.

# Prior Native Command Solve — F09n #12036

Branch `feat/f09n-native-command-solve-12036` starts at F09m `ab0a1a37d2`.
Reuse the exact shared F02 kernel/test dependency from PR #12024 `4f9d2e4b9f`.
The new native solve admits/replays fallback first, applies complete-horizon
command/increment and declared state margins, and requires guarded candidate
improvement plus independent canonical replay. Source/state/criteria mutation
fails. No live command, private fit, contact, physiology or parity is qualified.

Read `docs/development/feedback_controls/F09N-NATIVE-COMMAND-SOLVE-TURNOVER.md`.
The initial 169-test SDK campaign predates the final criteria-mutation fix.
The final 170-test campaign passes with zero failures/errors/skips in 250.31s;
all twenty executed hashes match current source. Global central five gates,
pinned mypy hook, LoD/DRY/architecture/title/size checks pass. The synthetic
objective improves 16.3614 to 0.000299930, but whole solve takes 44.93s/189
evaluations for three native steps: rapid matching remains unqualified.
Preserve genuine RED receipts, historical runs, missing local mypy-provider
result and stopped exploratory type check. Parent CI still needs actual Tools
capability probes/served-bundle reconciliation; the monolith register is refreshed.
Current context and normal commit/push validation remain required. Full
epic remains active. Sol/Luna/Astra account quota exhaustion is unchanged.

# Prior Native Tracking Objective — F09m #12023

Branch `feat/f09m-native-tracking-objective-12023` starts at published F09l
`d5f003614426aef0ec6fdf2c0ed203942e25101c`, draft PR #12017. Read
`docs/development/feedback_controls/F09M-NATIVE-TRACKING-TURNOVER.md` for frozen
objective semantics, TDD and promotion boundaries. Cost scales are physical
magnitudes; command slew cost does not enforce hard slew. No fitted private
motion or physiological acceptance is claimed. All seventeen models and six
ecosystems remain required, ending in muscular OpenSim and independent replay.

# Active: Persistent Project MyoSuite Native Search — F09l #12014

Branch `feat/f09l-persistent-native-search-12014` starts at published F09k
`d50db51b603e967c9e09a4b3c31f7f97a0fdaaac`, draft PR #12011. Read
`docs/development/feedback_controls/F09L-NATIVE-SEARCH-TURNOVER.md` for the
provisional-search/guarded-promotion authority boundary and retained TDD evidence.
The shared forecast admission extraction changes executed source identity;
historical F09k receipts apply to their published head, not this new source.

## Previous Forecast Foundation

Branch `feat/f09k-owned-sdk-forecast-12009` starts at F09j
`43e925dd7171f12593d6c8149e3468c720048bc7` (draft PR #12005).
The owned forecast restores complete current native state and reuses the SDK
recorder. Read chapter 38 and
`docs/development/feedback_controls/F09K-OWNED-FORECAST-TURNOVER.md` for lifecycle,
actual executed evidence and remaining boundaries. Preserve the live task and
the separate independent replay authority. All 17 model rows and six ecosystems
remain required; the endpoint is a private-capture-matched muscular OpenSim
model with full-state, full-horizon independent excitation/contact replay.

# Active: Project MyoSuite Feedback Recording — F09j #12003

Branch `feat/f09j-project-feedback-recording-12003` starts at F09i `976feaf051`.
Actual SDK campaign passes 89 tests with zero skips. The feedback source shares
the admitted recorder loop; immutable snapshots, callback mutation rejection and
independent replay are tested. Read chapter 38 and
`docs/development/feedback_controls/F09J-FEEDBACK-RECORDING-TURNOVER.md`.
Source/parameter provenance is declared, not signed. No production assembly,
mocap fit, muscle-only physiology or all-model completion is claimed.

# Active: Project MyoSuite Task and Independent Replay — F09i #11983

Draft PR #11996 is based on surviving parent #11857 after intermediate parent
merges; it remains unarmed. Exact original-source nominal driver/iron duration
replay now passes with poor contact/constraint diagnostics retained. CI exposed
install-pin mismatches and two exporter architecture-budget violations; those
are corrected locally. The fingerprint scratch optimization retains every
per-step check and has six native 3.8 RED-to-GREEN storage/hash regressions.
Actual SDK revalidation after these local changes passed 76 tests with zero
failures, errors or skips; all five central pre-PR gates also pass.

Branch `feat/f09i-myosuite-task-producer-11983` stacks on published F01f source
`73319871c74dbafeefc6548081e606cadc61de54`; parent merges and actual Tools main
authority remain required. Read canonical chapter 38 and
`docs/development/feedback_controls/F09I-PROJECT-TASK-TURNOVER.md`. Actual SDK
production probes cover resource closure, T01/profile freeze and full-state
uninterrupted/suffix/changed-future replay. Preserve original source bytes,
separate SDK/helper/native-provider identities and all-model/full-horizon gates.
Native evidence does not qualify anatomy, capture fit or muscle-only OpenSim.
Read-only diagnosis found active grip-site gaps and no native foot contact at
reset. Initial assembly must preserve source and distinguish reference-derived
from optimizer-assembled poses; qualify named constraints/support and full-state
reload before fitting. Tools-main ancestry is the current completed CI hold.

# Active: Same-Input Bushing Grip Parity, OSV-7 Phase 2 #11739

- Branch `claude/osv-7-grip-parity`. The shared OpenSim `BushingForce` law (`grip_contact/bushing_law.py`) and a `SimmSpline`-equivalent prescribed input (`grip_contact/prescribed_motion.py`) drive MuJoCo (`mjcb_passive`, RK4 1e-4 s), Drake (native `LinearBushingRollPitchYaw`, RK3 error controlled) and Pinocchio (`aba`, DOP853) on a free club, with each engine's own weld FK. Metrics are in `grip_contact/parity.py` (peak 5 %, RMS 2 % of peak), routed through GCV-10 `GripSeries`.
- Driver: every engine passes every quantity. The worst case is the Drake squeeze, 0.96 % peak and 0.42 % RMS, from the RPY and midpoint convention. See GRIP_PARITY_DECISIONS.md section 18. Evidence: `evidence/grip_kinetics/parity/`. Plots: `~/Videos/Parity Audit/golfer_realism/grip_kinetics/parity/`.
- Full-swing parity tests are `slow` (10 to 30 minutes per engine and club on a loaded host). Open: a contact-model grip, and Drake with the shared law to isolate its integrator.

# Active: Simscape URDF Exchange — #11569 Task 3

- Branch `claude/simscape-urdf-exchange-11569`; commit `SELF`. `smexport` is absent in R2025b, so `scripts/matlab/simscape_model_inventory.m` reads joints (type, DOF) and solid/inertia masses (unit-converted, density x volume where needed; NaN when not evaluable) from the canonical `GolfSwing3D_Kinetic` and from an `smimport` of the spec `golfer.urdf`. `export_simscape_urdf_exchange.m` writes `tests/fixtures/simscape/simscape_urdf_exchange_receipt.json`; `src/engines/simscape/urdf_exchange.py` diffs it against the URDF.
- R2025b result: smimport round trip exact (43/43 coordinates, 77.969 kg, error 0). Canonical 27 coordinates vs spec 43; totals 77.606 vs 77.969 kg, but the Simscape model has no legs and puts +17 kg in the shoulder bars (`HubtoLS`/`HubtoRS`), +4.3 kg trunk, +2.0 kg head/neck.
- Next: #11569 task 1 (inverse dynamics) or task 4 (run-102 playback video). The mass redistribution is a model decision for the owner.

# Active: GRF Engine Wiring and CoP Consolidation - GCV-2 #11708, GCV-6 #11712

- Branch `claude/gcv-2-grf-wiring`. MuJoCo, MyoSuite, Drake and OpenSim emit `contact:grf_/free_moment_/moment_com_` per foot and net from native contacts; Pinocchio from shared-contact-law `ContactSample`s (else `ground_reaction_unavailable`); Simscape is unavailable (GCV-3 #11709). `BundleOverlayProvider` uses `ground_reaction_overlay`, so the native videos show per-foot and net GRF.
- Every CoP goes through `biomechanics.ground_reaction.center_of_pressure`; legacy helpers are `DeprecationWarning` shims. Tests: `tests/integration/cross_engine/test_ground_reaction_parity.py` plus per-engine `*ground_reaction*` tests (static stance net Fz = weight within 2 %).
- GCV-5 (#11711) plots, API and web display landed in #11781; the remaining #11711 item is the real-MuJoCo-run plot sheet. Open for GCV-2: Pinocchio GRF render (1x + 0.5x) on an engine host.

# Active: Native OpenSim Geometry — #11903

Scoped F07/F08 branch `feat/feedback-native-markers-11903`. Read canonical
`manuals/upstreamdrift/chapters/26-native-opensim-geometry.qmd` and
`docs/development/feedback_controls/F07_NATIVE_GEOMETRY_TURNOVER.md` before
continuing. Explicit native frame/station geometry and bounded shared IK are
implemented; metadata-only physical placeholders fail closed. The unchanged
520-muscle candidate's three-pose pelvis probe remains unqualified and reaches
its source rotation bound. Preserve native replay, anatomy and registration gates.

# Active: MeshCat Camera Framing - NV-9 #11697

- Branch `claude/nv-9-meshcat-framing-11697`; epic #11673. MeshCat kept the 75 deg three.js default FOV; `MeshcatPage` now sets the shared `golf_view_presets.VIEWER_FOV_Y_RAD` (0.7 rad, OpenSim's value) on entry and raises if the page has no viewer camera. New `golf_view_presets.framing`: `projected_extent` and `fit_distance_m` (15 % margin) with unit tests.
- Open (engine host): verify the Drake/Pinocchio renders at 720p, feed the per-swing body bounding box from engine FK into `fit_distance_m` per view, and judge glyph legibility.

# Active: Run-102 Simscape Playback MP4 — #11569 Task 4

- Branch `claude/run102-playback-mp4-11569`; commit `SELF`. `export_video(..., size_px=(w, h))` and `cross_engine_replay.render_replay_frames(size_px=...)` set the frame size (default 480x480 unchanged; text scales with height). Run-102 marker playback rendered from the committed `two_window_fit_9967_102/candidate.npz` at 1920x1080, 60 fps, 1x / 0.5x / 0.1x into DeskComputer `~/Videos/Parity Audit/simscape/run102_playback/` (not committed).
- This is the marker-overlay playback (capture vs Simscape markers), not a Mechanics Explorer render (a GUI window, which the headless rule excludes on agent hosts). Run-103 is still blocked (`native_gate.json`).

# Active: Muscle Qualification Evidence Guard — F07 #11791

Branch `fix/feedback-muscle-evidence-11791`; current commit `SELF`; PR #11814. Scoped child #11810: parameter-only orchestration now returns no native replay and no unperformed audit successes. The test-first commit records two failures; 31 focused tests pass after correction. Read [Guard Turnover](../../../docs/development/feedback_controls/F07_MUSCLE_EVIDENCE_TURNOVER.md). Merged current main while preserving its impact chapter and this evidence chapter. Next: normal protected PR checks; F07 and native model qualification remain open.

# Active: Native Muscle-Model Asset Admission — #11819

Branch `feat/feedback-native-admission-11819`, commit `SELF`. Source pinning and
native structural inventory remain scientifically unqualified; caller anatomy
maps and complete dependency closure require review. Read
[F07 Turnover](../../../docs/development/feedback_controls/F07_MODEL_ADMISSION_TURNOVER.md)
and canonical manual chapter 18. F07 #11791 remains open; no external full-body
model, raw capture, native optimizer or replay was qualified by this slice.

# Active: Feedback Controls Planning — #11784

Documentation branch `docs/feedback-controls-11784`; commit `SELF`. Read [Design](../../../docs/development/feedback_controls/DESIGN.md), [Issue Dependencies](../../../docs/development/feedback_controls/IMPLEMENTATION_PLAN.md) and [Turnover](../../../docs/development/feedback_controls/TURNOVER.md). Goal: all-model six-engine parity culminating in muscle-driven OpenSim and independent excitation replay. Planning only; no new model/video is qualified. Next: F01 inventory/gate freeze, coordinate MOSAIC #11532 and parity #11605.

# Active: Impact-Phase Bushing Grip, OSV-7 #11739

- Branch `claude/osv-7-impact-grip`. The OpenSim bushing grip is driven over 0 to 1.8 s by the OSV-10 fits (`tests/fixtures/club_face/swing_q_*.npz`), mapped by name with `grip_contact.load_coordinate_swing`. Loop closure is below 0.001 mm and hand speeds are within 12 % of the measured wrist markers.
- Deflection is 0.56 / 0.58 mm and 0.84 deg, inside the bounds. Owner decision on PR #11774: the flat 500 N internal-force bound is replaced by `grip_contact.couple_check` (squeeze at most 50 N; transverse pair equals the couple/d Newton-Euler prediction within 5 %, 2 N m noise floor). Both pass for driver and iron (squeeze 3.3 / 3.9 N), so the full-window test is no longer an xfail. The club-welded-to-hand demand is 89 / 87 N m against the realised 100 / 99 N m (bushing amplification). Grip-frame spacing is 80.3 mm, not 76. See DESIGN_DECISIONS.md section 17.
- Next: MuJoCo (soft weld) and Drake (`LinearBushingRollPitchYaw`) bushing parity on the same fixtures. The plan is on #11739.

# Active: Ground Reaction Design Manual Slice - GCV-18 #11724

- Branch `claude/gcv-18-grf-design-manual`. Provisional QMD chapter `manuals/upstreamdrift/chapters/10-ground-reaction.qmd` (GCV-1 equations, conventions, unavailable values, symbols, tests, limitations) and registry blocker `UP-D1-ground-reaction-breakdown-inventory`; the registry stays `blocked-inventory-required` with no calculations. User manual §12.4/§12.7 now state what `grf_metrics.py`/`stability_metrics.py` do not compute and give the correct CoP and free-moment equations.
- Open in #11724: grip wrench (GCV-7) and impact parameters (GCV-15) chapters, ADR-0052, force-overlay and native-export user guides, shared-infrastructure and C4 entries.

# Active: Club Force and Torque Overlays - GCV-10 #11716

- Branch `claude/gcv-10-grip-overlays`; epic #11706. Per-hand, net-at-midpoint and couple glyphs (groups `grip_per_hand`, `grip_net`, `grip_couple`, `grip_mof`) carry the `split_method` label; unavailable quantities are listed as unavailable, never zero (`force_overlay/grip_frame.py`, frame metadata `grip_unavailable_labels`).
- Plots: `biomechanics/grip_plot_model.py` (series), `GET /analysis/grip-wrench`, web `GripWrenchCharts.tsx`, PyQt tile `grip_wrench_plots` (`src/tools/grip_wrench_plots/`). `analysis.grip_wrench` parity gap closed.
- Native export: `--grip` adds the overlay (Drake uses its own KKT multiplier; other viewers show the MuJoCo plant at that engine's pose) and writes `<swing>_<engine>_grip_wrench.json`; `--views hands_closeup --no-grid` tracks the grip midpoint per frame (`view_lookats`, `WorkerJob.lookats`).
- Videos: `~/Videos/Parity Audit/forces_and_impact/club_forces/` (MyoSuite/MuJoCo arena, 1080p60, 1x/0.5x, clean 0.25x impact clip, plot PNG). Follow-up: Drake render did not finish under load (14 frames in about 12 minutes, then a Playwright EPIPE crash); rerun when the host is idle.
- Open: `hands_closeup` azimuth is judged by eye only; Pinocchio gives a net allocation only and OpenSim is unavailable.

# Active: Lift Pack Parity Baseline - LIFT-1 #11741, Epic #11740

- Branch `claude/lift-1-pack-parity-baseline` (PR #11771). Audit package `src/shared/python/lifting/pack_audit/`; run `python3 scripts/lifting/run_pack_parity_baseline.py`; results in `docs/development/lifting/PACK_PARITY_BASELINE.md` and `pack_parity_baseline.json`.
- Finding: same-q FK, feet and total mass agree across the four packs; grips, start poses, limits, phases, bench mass and contacts do not. Nine new pack issues filed (MuJoCo_Models#427-#428, OpenSim_Models#414-#416, Drake_Models#390-#391, Pinocchio_Models#449-#451).
- Next: LIFT-2 shared exercise spec.

# Active: Shared Ground Reaction Core — GCV-1, Epic #11706

- Branch `claude/gcv-1-ground-reaction` (PR #11733): per-foot and net GRF, CoP, free moment and moment about the CoM in `src/shared/python/biomechanics/ground_reaction.py`. Next: GCV-2 (#11708) engine wiring.

# Active: Force Arrow Scaling — GCV-4, Epic #11706

- Branch `claude/gcv-4-arrow-scaling` (PR #11738): body-weight and peak arrow scale modes; overlay route takes a `ForceOverlayQuery` dependency. Next: GCV-5 (#11711) plots, API and web.

# Active: Club-Face Orientation Residual, OSV-10 #11759

- Branch `claude/osv-10-face-roll-refit`, stacked on PR #11752 (OSV-8). The shared marker IK adds `sqrt(w)(R(q)a - R_capture a)` on the `Clubhead` frame from the capture head triad (`motion_matching/club_face_target.py`, `FACE_ORIENTATION_WEIGHT = 3`, `--face-weight 0` = marker-only). It applies in the full-capture IK, the consistency re-solve and the ZMP/shooting re-solves through the existing `axis_targets` path.
- Fixtures: `tests/fixtures/club_face/*` are regenerated only by `python3 -m scripts.regenerate_club_face_fixtures --work RUNS --capture {driver,iron}` (about 40 min each); `provenance.json` holds the hashes, calibrated triad offsets and before/after face events.
- Result (model vs capture, each at its own sub-sample impact `club_face.ball_passage`): the driver impact is 11.3 vs 8.8 deg (was 29.5) and the iron 7.6 vs 8.4 (was 17.9); top and address are within 1.1 deg. Marker RMS: driver IK +2.5 %, iron flat; dynamics improved. The strict xfail is replaced by `test_face_tracks_the_capture_at_address_top_and_impact` (5 deg, every engine's FK). MyoSuite was not installed locally.
- Open: the model's peak clubhead speed falls about 24 ms before the ball (the capture's falls about 3 ms before it). Reference: `docs/research/simscape_matching_reference/simscape_matching_reference.tex` (OSV-10 section).

# Active: Clubface Roll at Address, OSV-8 #11755

- On PR #11752. The matched hand-club chain leaves the club roll to fitted wrist constants, so club `+x` was open 30.7 deg (driver) and 44.8 deg (7-iron) at address in every engine. One shared constant, `ADDRESS_SQUARE_FACE_ROLL_DEG` in `model_appearance/club_assembly.py`, now rolls the head about the shaft (and defines `clubface_vector`); `club.face_roll_deg` in a spec overrides it.
- Tests: `tests/unit/model_appearance/test_clubface_square_at_address.py` (each engine's own FK at the captured address pose, `tests/fixtures/club_face/address_poses.json`). OpenSim STLs and `provenance.json` regenerated.
- Impact: `club_face.impact_frame` (shared `detect_impact_index`, closest-approach fallback, raises if not at the ball). True impact face is +30 deg (driver, t=1.327) and +18 deg (iron, t=1.337) open, while the capture head triad is about +2 deg: the IK/matched trajectory under-rotates the club through release (grows from 4 deg at the top), so a constant roll cannot fix it; needs an IK refit with head-triad weight (test `test_face_is_square_at_impact` is a strict xfail). `tour_matching` club models are unrolled.

# Active: Visible Head and Neck (GCV-12) - #11718

- Branch `claude/gcv-12-visible-head`; epic #11706. The anthropometric specs already carry a `Head` body on a three-axis neck at the cervicale; only the native Simscape spec (v1/v2) has none, so the head rides the `Head` body (follows the fitted neck) and falls back to the torso for v1/v2 in MuJoCo only.
- Code: `model_appearance/head.py` (procedural head, face, ears, neck, hair or cap), `mujoco/python/head_visual.py`, native viewer `backends/_head.py`; schema fields `head` and `body_model`. Visual only: no mass, inertia or DOF change (identity tests). Gaze channel `head.orientation_override` plus `drive_visual_head` for OSV-3 #11729.
- Blend balls are capped at 1.1x the adjoining limb radius (test); `*hubto*` bodies are a small `shoulder` part, not torso-sized pads.
- Not done: `body_model: meshes` (rejected until CMB-6 #11657), web `GolferModel.tsx` head. Owner renders in `~/Videos/Parity Audit/forces_and_impact/visuals/head/`.

# Active: Head Gaze Stabilisation - #11729

- Branch `claude/osv-3-head-gaze`; epic #11726. `motion_matching/gaze.py` (eye point, gaze error, schedule, metrics, neck IK), `pipeline/gaze_residual.py` (soft residual, receipt `head_gaze`), `model_appearance/ball.py` (the one ball-at-address function, reuse in GCV-13). `--gaze-weight` defaults to 0 (marker-faithful).
- Reference: `docs/development/full_body_models/HEAD_GAZE_REFERENCE.md` (+ `.tex`). Capture-A driver, weight 10: theta_gaze RMS 21.3 to 0.8 deg, marker RMS 28.0 to 33.8 mm. Gaze axis is calibrated at address (nominal +x is 37 deg off).
- Open: iron and other engines, neck PD in forward dynamics, MyoSuite neck map audit (X is lateral bending, Y flexion; map not changed), published tour head ranges.

# Active: Compliant Bushing Grip Model - #11739

- Branch `claude/osv-7-bushing-grip` (phase 1, Refs #11739 #11726): `src/shared/python/grip_contact/` interface, OpenSim `grip_model="bushing"` (`weld` default unchanged, `contact` raises), `split_method="bushing"`.
- Valid window 0 to 0.94 s with designed damping (zeta 0.7): per-hand peak 142/162 N, internal 151 N (bound 500), 0.16 mm and 0.22 deg. Beyond 0.95 s the committed OpenSim IK candidate is unusable (marker RMS 262 mm, branch switches); the full-window test is a strict xfail.
- Next: phase 2 contact model, other engine parity, full 1.8 s run, `golf_humanoid.osim` builder.
- Next: qualified closure-consistent OpenSim IK input, then re-run the receipt; phase 2 contact model, other engine parity, `golf_humanoid.osim` builder.

# Active: Impact Parameters Panel - GCV-17 (#11723)

- Branch `claude/gcv-17-impact-panel`, stacked on GCV-16 (PR #11769, adapters); epic #11706. Shared card model `impact_parameters/panel_model.py` feeds `GET /api/analysis/impact-parameters`, the PyQt6 dock `src/tools/impact_parameters_panel/` (tile `impact_parameters`) and web `ImpactParametersPanel.tsx`. Parity entry `analysis.impact_parameters`.
- Run series come from `run.simulation_data["clubhead_series"]`, an engine `get_clubhead_series()`, or a MuJoCo run with a `clubhead` body; otherwise the card is unavailable with a reason. Matched-swing ledger ids are not simulation run ids, so that view shows unavailable until candidates carry a club series.
- Next: wire engine runs to record `clubhead_series`; video HUD stamp (GCV-14); Impact Explorer prefill is owned by Tools (#9546).

# Active: ClubheadSeries Engine Adapters - GCV-16 (#11722)

- Branch `claude/gcv-16-clubhead-adapters`; epic #11706. Package `src/shared/python/impact_parameters/adapters/`: one `ClubFaceSpec` (face centre and axes in the club body frame) and one `rigid_body_series` kernel; MuJoCo/MyoSuite, Drake, Pinocchio, OpenSim and Simscape adapters only supply their own FK.
- Impact time: `adapters/impact_time.py::select_impact_index` is the single integration point for the OSV-8/OSV-10 closest-approach rule (not on main yet). `ClubFaceSpec` default centre is the club body origin until GCV-11 face geometry lands.
- Next: GCV-17 impact panel consumes these adapters.

# Active: Centroidal Feasibility Filter V2 — #11669

- Branch `feat/centroidal-filter-v2-11669` (PR #11703); epic #11667. Next: Balance-3 contact-consistent inverse dynamics.

# Active: High-FPS Video Frame Schedule — GCV-14, Epic #11706

- Branch `claude/gcv-14-frame-schedule` (PR #11734): half- and full-speed video variants. Next: impact-time detection from clubhead kinematics.

# Active: Shared Grip Wrench Core — #11713

- Branch `claude/gcv-7-grip-wrench`; epic #11706. Module `src/shared/python/biomechanics/grip_wrench.py` (hand-on-club wrench, midpoint net force and couple, contact-moment/free-torque split, per-hand MOF, club-local frame, `split_method`).
- Simscape fixtures (`tests/unit/engines/simscape/test_force_channels.py`) have only total hand force, LH MOF and midpoint couple, no per-hand forces, so the Simscape cross-check is deferred to GCV-9 (#11715).
- Next: engine adapters populate `ContactReaction.grip_wrench` through `to_contact_reaction_wrench`.

# Active: Finish Feasibility Balance - #11667

- Balance-1 (#11668), branch `feat/finish-feasibility-metrics-11668`: `pipeline/finish_feasibility.py` adds ZMP-inside fraction, friction-cone utilisation, foot slide and yaw pivot, pelvis yaw error and vertical force range to `dynamics.finish_feasibility` (reference and simulation). `finish_feasibility_cli` annotates saved runs.
- Baseline: `evidence/ground_support/finish_feasibility_baseline.json` (driver reference ZMP inside 0.29 over 1.0-1.5 s, iron 0.55). The committed canonical receipts were not edited: their trajectories are not committed and a rerun does not reproduce their spec hash.
- Next: Balance-2 (#11669) centroidal feasibility filter v2; a linearised joint-space QP prototype has not yet reduced the outside fraction.
