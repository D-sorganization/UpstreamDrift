# Character Builder Docs and Tutorial, CMB-12 #11663

- User docs: `docs/help/character_builder.md`, `docs/user_guide/character_builder_quickstart.md`, `character_presets.md` and the end-to-end `character_builder_tutorial.md` (build, dress, swap the club, export spec/URDF/MJCF/OpenSim/glTF, desktop walkthrough, simulate in MuJoCo). Desktop screenshots regenerate with `scripts/capture_character_builder_screenshots.py`.
- Every number in the tutorial is checked by `tests/unit/character_builder/test_cmb12_*.py`. The spec hash `4ee2c43d` follows the driver document, so a change to the canonical driver spec must update the tutorial and these tests together. Step 6 runs only where `mujoco` imports; its exported MJCF has no floor or actuators, so the body free-falls (-0.198 m after 0.2 s).
- Open elsewhere: smooth skin blend weights and MuJoCo per-segment meshes (CMB-6 #11657), the Playwright flow (CMB-7 #11658, owner decision on adopting Playwright), lower limbs not scaling with `mass_kg` (#12199).

# Active: Adopted Per-Capture Grip Rotation, #12221

- Branch `claude/12221-grip-adopt`, stacked on #12270. Applies the fitted `grip_rotation_deg` to `full_body_spec_anthro_driver.json` only; calibrated driver commands now add `--fix-club-markers`. Driver receipts regenerated on ControlTower (address / IK / dynamics, mm): `anthro_driver_seeds` 11.0 / 30.3 / 82.0, `anthro_driver_drake` 11.0 / 30.3 / 86.3 (dynamics worse), `anthro_driver` 50.9 / 46.5 / 71.9. Replay is sensitive at 3 to 15 mm to tiny parameter changes (CANONICAL_RUN.md §2A).
- Driver club-face fixture regenerated at `9fe74740a5` (impact speed -2.5 %, peak 4.4 ms early, face 16.88 / 14.96 deg; meets GCV-20 and OSV-10); #12266 no longer blocks it. The MSK driver grip calibration and both `golf_humanoid*.osim` clubs were regenerated from it (`rebuild_club`). The OpenSim bushing grip exceeds its 2 deg / 50 N full-swing bounds only in the 6 ms ball-impact window (2.62 deg, 51.3 N): strict xfail, decision on #12272.
- 7-iron deferred: document, OpenSim export and receipts unchanged (4.0 / 28.4 / 109.9 and 41.5 / 45.2 / 104.9). Its refit passes on the tracked swing but its same-input export gives -5.0 %, -5.1 ms, 8.71 deg face, which would redden eight face tests; fit receipt stays committed for a follow-up.
- The PR awaits the owner decision on making `--fix-club-markers` the calibrated default.

# Active: Per-Capture Grip Refit on the Fixed Clusters, #12221

- Branch `claude/12221-grip-refit`, stacked on #12265. Landed: refit tooling (`grip_fit.weighted_frames`, `compose_rotation_deg`, `apply_grip_rotation`, `scripts/set_spec_grip_rotation.py`), the per-capture fit receipts `fit_grip_rotation_receipt_driver.json` / `_iron7.json`, and the evidence. Documents are NOT changed.
- Fixture flags with the cluster fixed: driver speed −1.9 %, face gap 0.56°; 7-iron −1.9 %, 0.62°; no wrist on its bound at address. The refit is invalid with the cluster free (static trial moves the capture face). Calc record: "Per-Capture Grip Rotation on the Fixed Clusters".
- Open: owner decision on `--fix-club-markers` default; then apply with `scripts/set_spec_grip_rotation.py` and regenerate receipts; #12266 blocks the driver fixture export.

# Active: 7-Iron MSK Grip Calibration, #12042

- Branch `claude/12042-iron7-msk-cal` (stacked on #12247). `msk_club_grip_calibration.json` now has `iron7` entries for `golf_humanoid` (grip gaps 0.04 / 0.02 mm, landmark RMS 14.9 mm) and `golf_humanoid_scaled` (0.24 / 0.25 mm, 40.0 mm), from the existing `msk_club_calibration --club iron7` CLI. `render_msk_club.py` now draws and measures the `--club` it tracks: before, the viewer loaded `golf_humanoid.osim` with its committed driver, so a 7-iron render showed the driver.

# Active: Neutral Hip Zero-Twist and the MSK Turn Render, #12042

- Branch `claude/12042-turn-slices`. `hip_rotation_zero` places `hip_rotation = 0` at the measured address thigh twist, and the ±40° bound moves with it. The canonical specs carry +45/−26° (lead/trail), so the lead hip is pinned at +40° at impact. `--hip-zero-twist neutral` (opt-in; default `capture` keeps the canonical receipts) uses 0/0. Pelvis turn is then within 3.2° of the markers at top and impact on both captures, with IK 25.6/22.9 mm. FD is 51.2 mm on the driver, but 91.2 mm on the 7-iron (after impact). Evidence: `evidence/hip_zero_twist_12042/`; calc record in `hip_axis_mirroring.tex`, section "Neutral Hip Zero-Twist and the Pelvis at Impact".
- `render_msk_club.py --capture --club --mode` tracks with the capture turn targets and planted feet (shared `msk_turn_targets.tracking_inputs`).
- Open: owner decision on the `neutral` default (regenerates canonical receipts and fixtures); trail hip now reaches −40°; upper-trunk over-twist at the top (slice 7).

# Active: Measured Driver Club Cluster, #12221

- Branch `claude/12221-driver-cluster`, stacked on `claude/12221-fixed-triad`. `evidence/club_markers/driver.json` (from the calibrated `anthro_driver_seeds` receipt at `1bef553457`) is now in the driver document. `anthro_driver`, `anthro_driver_seeds` and `anthro_driver_drake`, the OpenSim driver export, the inventory hashes and the tutorial hash `4ee2c43d` are regenerated.
- Fixed cluster, fixture flags: capture face 5.50 / 14.96° (the canonical frame), model impact face 16.13°, speed −3.6 %, peak −0.3 ms. The canonical seeds receipt reads 7.5 / 30.5 / 82.6 mm (was 6.4 / 29.5 / 80.8). The nominal driver reads 41.7 / 46.1 / 73.6 mm (was 42.6 / 52.7 / 97.3).
- Open: the same-input export of the fixed driver run diverges at impact; refit `grip_rotation_deg`; decide whether `--fix-club-markers` should be the default. Calc record: "Measured Driver Club Cluster".

# Native Wrapped-Path Closure Handoff — #12234

`native_wrapped_path.py` carries a source-hash-bound declaration for
WrapCylinder/WrapEllipsoid/WrapSphere geometry, exact PathWrap targets and
owners, and ConditionalPathPoint coordinate ranges. The constrained muscle
path remains default-reject unless the caller supplies the declaration. Native
reference observation reports the evaluated route without changing it. It
does not prove global worklessness or guide-work bounds and makes no
physiological claim. Eleven OpenSim 4.6 wrapped-path fixtures pass; the
shared Chapter 56 typed findings/request integration is present in the
current feature history. See provisional Chapter 57 and
`docs/development/native_wrapped_path_diagnostic.qmd`.

# Native Contact Source Closure Handoff — #12244

Commits `316b011324` and `b48f7b053b` bind the exact-source contact declaration,
shared sampler, and `opensim_force_torque.py` implementation digest into
constrained replay identity. OpenSim 4.6 tests cover Smooth and HuntCrossley
force identities, sphere/half-space geometry closure, source mutation, changed
provider digest, default rejection, and finite constrained Smooth wrench
sampling. HuntCrossley constrained replay remains untested because strict
self-contained XML validation previously rejected its generated fixture;
the parser fix and new source audit tests now pass. Request/runner serialization,
preparation closure, export identity, and fresh-replay digest checks are
integrated. No capture contact, support/grip, or physical qualification is
claimed. See provisional Chapter 58 and
`docs/development/feedback_controls/F07_NATIVE_CONTACT_SOURCE_CLOSURE_TURNOVER.md`.

# Active: Fixed Club Cluster in the Static Trial, #12221

- Branch `claude/12221-fixed-triad`. Opt-in `--fix-club-markers` (needs `--static-seeds`) keeps the six `Clubhead` markers at the document's per-club offsets, so a grip rotation can no longer move the capture's own face through the static trial. Default receipts are unchanged.
- Fixture flags at `87ba5b2f39`: 7-iron −15.6 % (control, corrected #12210 cluster) → −2.6 %, peak 0.0 ms, impact face 7.18 vs 3.21°. The driver document's cluster puts the capture address face at −20.8°, so it is not a valid face frame.
- Next: a measured driver cluster (#12210 procedure), then refit `grip_rotation_deg` on the fixed clusters. Calc record: `simscape_matching_reference.tex`, "Fixed Club Cluster in the Static Trial".

# Active: Club-Face Fixtures on the Current Spec, #12117

- Branch `claude/12117-club-face-fixtures`. Fixtures from `657c811bcd` (#12145 sources; input receipts value-identical to main's #12166 receipts). Feet now flat (3.2–3.8 cm), address face follows the capture (5.43 / −5.41° vs 5.50 / −5.35°).
- `club_face.impact_frame` now enforces the 0.15 m ball radius on the peak-speed frame (the driver replay peaks 0.351 m short of the ball). OSV-8 and OSV-9 address-face tests compare with the capture's address face (2° unchanged). OSV-9 MSK grip calibration and both `golf_humanoid*.osim` regenerated (control reproduces main byte for byte). Calc record: `simscape_matching_reference.tex`, "Fixtures Regenerated on the Current Spec".
- Strict xfails: 7-iron impact face 10.97 vs 3.21° (#12221, club-in-hand rotation); GCV-20 speed −6.1 / −10.2 % (#11767). Toe-out miss filed as #12222.

# Active: Drake and MuJoCo IK Parity, Weld Rotation Row — #12039

- Branch `claude/12039-drake-address-parity`. Drake's dual-grip weld rotation row was the small-angle vee with Jacobian `J_a - J_b` (up to 0.84 off finite differences); MuJoCo's is the principal log with its exact Jacobian. Both now use `full_body_ik._closure_rotation_row`; Drake `closure_error` reports the principal angle. Tests: `tests/unit/motion_matching/test_drake_mujoco_ik_parity.py`.
- Address stages (static seeds, foot progression capture), Drake minus MuJoCo: driver +1.64 → 0.00 mm, 7-iron −0.54 → +0.02 mm; toe-out identical. Capture-A Drake receipt regenerated: 6.3 / 29.2 / 78.8 mm against MuJoCo's 6.3 / 29.5 / 80.8 mm. Evidence `evidence/foot_progression/drake_parity_12039/`; calc record "Weld Rotation Row Parity" in `simscape_matching_reference.tex`.

# Active: Nominal 7-Iron IK and the Club-Face Triad Gate, #12030

- Branch `claude/12030-nominal-iron-ik`. Root cause: `full_body_spec_anthro_iron7.json` carries the driver's clubhead triad offsets (edges 92/52/96 mm against the 7-iron capture's 106/127/40 mm), so the nominal run's face residual (OSV-10) fitted a meaningless rotation. `club_face_target.observe_frame_rotations` now raises `TriadShapeMismatchError` above a 5 mm median rigid residual (0.2 mm when matched, 42.4 mm here). Face targets, the impact split and the receipt's `ik.face_orientation` record the reason. The IK report adds `frame0_marker_rms_m`.
- Nominal 7-iron: IK 264.8 → 67.9 mm (frame 0 70.2 mm against a 70.4 mm address), dynamics 1355 → 79.2 mm, 0 IK range-of-motion flags. The nominal driver is unchanged by the gate; its dynamics 79.3 → 97.3 mm comes from main's GCV-20 code (a control on main is identical). Calc record: `simscape_matching_reference.tex`, subsection "Triad-Shape Validity Gate".
- Open: correct the 7-iron document's nominal head triad, which changes every 7-iron receipt's base spec hash.

# Physical Inertia Helper Handoff — #12183

`validate_physical_inertia` now uses sorted principal moments instead of frame
diagonals, and absolute symmetry tolerance with zero relative tolerance. Two
rotated impossible tensors and a masked asymmetry failed before the fix; affected
calibration/human-prior tests now pass 25 cases. Chapter 51 gives reproduction
and numerical tolerances. Values are preserved on success. Native source
admission and optimizer enforcement are separate integration work; physical
source provenance, matching and all-model qualification remain open.

# Active: Closed-Loop Gaze Neck, #11729

- OSV-3d. Branch `claude/osv-3d-gaze-closed-loop` (PR #12194; builds on the merged #12154). `--fd-neck gaze-closed`: `GazeNeckFeedback` is a kkt `reference_hook(t, q, q_t, v_t)` (`tracking_controller.py`) that re-solves the neck on the simulated state once per capture frame and adds the correction and its rate to the neck position/velocity targets. Receipt: `dynamics.head_gaze.feedback`.
- Result (`evidence/head_gaze/fd_neck_closed_loop.json`): release-window RMS 17.2→1.7 deg (driver), 8.6→1.2 deg (7-iron); driver hold window 1.1→2.9 deg (ball-impact torso transient). Position-only first version lagged (hold 4.8 deg); the rate term is required.
- Open: MyoSuite/OpenSim neck actuation and other engines, a gaze schedule that lets the head turn with the torso (neck still at range on 93/69 updates).

# Active: Forward-Dynamics Gaze Neck, #11729

- Branch `claude/osv-3c-gaze-neck-tracking`. `motion_matching/pipeline/gaze_tracking.py`: `--fd-neck gaze` re-solves the tracked reference's neck to the gaze schedule (model FK, bounded, after the feasibility filters); `dynamics.head_gaze` reports schedule error per window for the tracked reference and the replay. `render_head_gaze_clips --trajectory replay` renders the replay; `scripts/summarize_fd_neck_tracking.py` writes `evidence/head_gaze/fd_neck_tracking.json`.
- Result: replay schedule error RMS address→impact 20.3→1.8 deg (driver), 19.0→1.2 deg (7-iron); release 15.4/6.3 deg from open-loop torso error; neck yaw saturates ±80 deg on 213/114 frames; head-marker RMS doubles. Published head rotation at the top: −33.6/−25.4 deg (Zhang et al. 2026).
- Open: closed-loop gaze controller (neck re-solved from the simulated torso), MyoSuite/OpenSim neck actuation and other engines' receipts, a gaze schedule that lets the head turn with the torso.

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

Branch `fix/feedback-muscle-evidence-11791`; current commit `SELF`; PR #11814. Scoped child #11810: parameter-only orchestration now returns no native replay and no unperformed audit successes. The test-first commit records two failures; 31 focused tests pass after correction. Read [Guard Turnover](docs/development/feedback_controls/F07_MUSCLE_EVIDENCE_TURNOVER.md). Merged current main while preserving its impact chapter and this evidence chapter. Next: normal protected PR checks; F07 and native model qualification remain open.

# Active: Native Muscle-Model Asset Admission — #11819

Branch `feat/feedback-native-admission-11819`, commit `SELF`. Source pinning and
native structural inventory remain scientifically unqualified; caller anatomy
maps and complete dependency closure require review. Read
[F07 Turnover](docs/development/feedback_controls/F07_MODEL_ADMISSION_TURNOVER.md)
and canonical manual chapter 18. F07 #11791 remains open; no external full-body
model, raw capture, native optimizer or replay was qualified by this slice.

# Active: Feedback Controls Planning — #11784

Documentation branch `docs/feedback-controls-11784`; commit `SELF`. Read [Design](docs/development/feedback_controls/DESIGN.md), [Issue Dependencies](docs/development/feedback_controls/IMPLEMENTATION_PLAN.md) and [Turnover](docs/development/feedback_controls/TURNOVER.md). Goal: all-model six-engine parity culminating in muscle-driven OpenSim and independent excitation replay. Planning only; no new model/video is qualified. Next: F01 inventory/gate freeze, coordinate MOSAIC #11532 and parity #11605.

