# Scapula Matching and Measured-Marker Previews

Moving-scapula whole-body matches use a bilateral protraction prior of 7.5 degrees by default. Address is bounded to 5–10 degrees, and the backswing to 5 degrees through the existing registry limit of 25 degrees. Native primitive signs are mirrored: left Rx is positive protraction; right Rx is negative protraction. Joint roles resolve the native coordinates for each model instead of assuming joint numbers.

The top is the existing measured pelvis-yaw phase proxy, with missing observations excluded. An explicit `scapula_backswing_end_frame` can replace that proxy. The bounds widen with a raised-cosine release between the top and the capture peak-speed proxy. These are matching assumptions; neither scapular anatomy nor ball contact is measured by this policy. Set `scapula_protraction_deg=0` to retain legacy fitting behavior and output layout.

Preserve a selected match's dimensions and offsets when comparing this prior. Review measured-position RMS, head/foot orientation, native torso and pelvis displacement, and frame-to-frame branch continuity together. Reduced torso-hub displacement is a geometric result, not a measurement of upper-body center of mass.

`gs3dx_capture_marker_overlay(cap, frames)` returns every measured capture channel in the same fixed address-waist frame as the joint-centre targets. It keeps residual-invalid and nonfinite samples missing, selects the exact pose frame indices, and reports channel labels and per-frame valid counts. It does not fill missing display markers. The older `gs3dx_marker_overlay(jc, frames)` remains available for joint-centre diagnostics.

`gs3dx_render` defaults to 1920×1080 video frames and MPEG-4 quality 100. Supply `resolution=[width height]` and `video_quality` to override these settings. Dimensions must be even and at least 64 pixels; quality must be 0–100. Still exports use the requested pixel dimensions as well. Video frames use explicit offscreen pixel export, avoiding display-size limits in `getframe`. A renderer request that falls back to GIF does not establish successful MP4 delivery.

For matched Human exports, run both marker modes in the same output directory:

```matlab
for marked = [true false]
    report = gs3dx_match_export("capture-A", ...
        registry_repo=registry_repo, output_dir=output_dir, ...
        views=["face-on","down-the-line"], overlay_markers=marked, ...
        resolution=[1920 1080], video_quality=100);
end
```

The fixed animation bounds include all finite measured markers. Pass the same `scene_markers` array to the renderer in both modes to preserve identical framing.

Marker and clean filenames, stills and provenance files have different suffixes so one mode cannot overwrite the other. Both modes can reuse the same capture/model/geometry/solver-identity checkpoint. Repeat for the owner capture to obtain eight videos. The exporter remains restricted to Human; the shared fitting policy resolves moving-scapula roles across compatible variants.

Before sharing, verify actual decoded dimensions, frame rate, all frame counts and the two-view visual result. Keep private C3D captures, trajectories and owner videos outside Git. These previews demonstrate inverse-kinematics geometry; forward dynamics and contact qualification remain separate work.

## Quiet Torso Refinement

The refined Desktop matches directly fit measured BackTop, BackLeft and BackRight points. The four waist points already inform the pelvis and derived lower-body targets. Highlight all seven points using `highlight_markers=markers(:,body_indices,:)`; pass the complete measured array as `markers` and `scene_markers`. Missing raw samples stay NaN and are never displayed as interpolated observations.

Native spine bands are excursions about the original keyed address seed: 15 degrees for each Spine Tilt coordinate and 45 degrees for torso twist. Bounds are hard through the measured backswing-top proxy, then become soft hinge penalties. In this opt-in spine profile, hard scapula bounds release after top while the scapula posture pull fades to impact. This avoids the large residual outliers seen when bounded trust-region tracking was imposed through the whole swing.

The two-coordinate neck cannot independently reproduce arbitrary head yaw. `head_orientation_mode="axis"` tracks the calibrated head long axis; the existing full 3D head error remains separately reported. The selected profile has small head-axis errors but full orientation errors around 43 degrees. It is not qualified as a full head-orientation match.

Keep the capture-specific dimensions and original 14 target offsets fixed. Calibrate the three back offsets once from the original address pose:

```matlab
% mdl is loaded with the selected original geometry; baseline is the original
% keyed native fit. jc.head_R and jc.gap.head_R are calibrated head targets.
[raw, coverage] = gs3dx_capture_marker_overlay(cap, 1:cap.n_frames);
[found, back] = ismember(["BackTop","BackLeft","BackRight"], coverage.labels);
assert(all(found));
jc.back_marker_points = raw(:,back,:);
ks = simscape.multibody.KinematicsSolver(mdl);
jp = ks.jointPositionVariables;
[keys, ids] = gs3dx_joint_keys(mdl,jp);
[found, at] = ismember(ids,string(baseline.joint_ids)); assert(all(found));
seed = struct('joint_keys',keys,'units',string(jp.Unit), ...
    'joint',baseline.joint(at,1),'status',1);
world = find_system(mdl,'LookUnderMasks','all','FollowLinks','on', ...
    'ReferenceBlock','sm_lib/Frames and Transforms/World Frame');
paths = string(unique(jp.BlockPath,'stable'));
torso_base = char(paths(contains(paths,'Left Scapula')));
addFrameVariables(ks,'p','Translation',[world{1} '/W'],[torso_base '/B']);
addFrameVariables(ks,'r','Rotation',[world{1} '/W'],[torso_base '/B']);
grounded = strcmp(mdl,char(gs3dx_names().variants.fullbody));
roles = gs3dx_ik_joint_roles(jp,[],grounded_legs=grounded);
addTargetVariables(ks,roles.target_ids);
addInitialGuessVariables(ks,roles.closed_ids);
addOutputVariables(ks,string(ks.frameVariables.ID));
[found,t] = ismember(roles.target_ids,string(baseline.joint_ids)); assert(all(found));
[found,g] = ismember(roles.closed_ids,string(baseline.joint_ids)); assert(all(found));
[native,status] = solve(ks,baseline.joint(t,1),baseline.joint(g,1)); assert(status==1);
R0 = eul2rotm(deg2rad(native(4:6).'),'XYZ');
back_offsets = R0.'*(jc.back_marker_points(:,:,1)-native(1:3));
delete(ks);
poses = gs3dx_whole_body_ik(jc,model=mdl,frames=baseline.frames, ...
    offsets=baseline.offsets,initial_pose=seed,backward=false, ...
    posture_weight=.04,smooth_weight=.025,gap_weight=.1,rom_weight=.025, ...
    foot_orientation_weight=.1,head_orientation_weight=head_weight, ...
    head_orientation_mode="axis",scapula_protraction_deg=7.5, ...
    back_marker_weight=.5,back_marker_offsets=back_offsets, ...
    spine_bend_excursion_deg=15,torso_twist_excursion_deg=45, ...
    spine_bound_weight=.35,spine_posture_weight=.08,spine_smooth_weight=.08, ...
    gap_target_weight=struct('pelvis',1,'hip_L',1,'hip_R',1,'knee_L',1, ...
      'knee_R',1,'ankle_L',1,'ankle_R',1));
```

Use head weights 0.2/0.16 for Model Swing 1/2. The back-marker offsets are in the torso base frame, in metres. The native spine bounds are converted from degrees to radians internally. Raw back RMS, 14-target RMS, native spine excursions and full head error are separate diagnostics. Soft post-top bands permit small excursions beyond the stated backswing limits. Back and spine terms default to disabled, preserving the legacy opt-out path.

The exporter uses neutral Model Swing titles and filename prefixes by default; `display_name` can set another neutral title. Its default solving profile remains the basic scapula profile. Use the explicit whole-body recipe above for quiet-torso refinement and render its poses with `gs3dx_render`.

The pixel exporter preserves aspect ratio and the figure canvas. Small native pixel rounding is corrected only in verified white borders; model pixels are never rescaled or cropped. Native portrait/landscape circle tests detect the previous stretching defect. Shareable files use neutral titles and companion metadata; detailed capture hashes and private source paths remain in private provenance.

## Native Targets Across Moving-Scapula Variants

The requested scope includes all 13 registered moving-scapula variants. `target_scope="auto"` selects the 14 canonical position targets for a complete native leg graph, or pelvis, both shoulders/elbows/wrists and club for an upper-body-only graph. Baseline, Slim and Quat use the latter eight targets. A partial or ambiguous lower-body graph is rejected. Explicit `whole_body` requires the complete graph; explicit `upper_body` selects the eight available targets on either supported graph.

Models without native feet add no foot frame outputs. Their foot-orientation diagnostics remain missing, and positive foot-orientation weights are rejected. Upper-body results expose `target_scope='upper_body'` and capability fields. Existing full-body result layouts remain unchanged for legacy compatibility. Gap overrides are validated against canonical names, then selected in the exact active-target order.

FullBody is an earlier construction stage with feet rigidly connected to World. Its automatic scope selects the eight upper-body targets; its hip, knee and ankle coordinates become native loop-closure guesses instead of independent fitting coordinates. This preserves its fixed-foot topology. Explicit `whole_body` is rejected for this stage, and foot-orientation fitting is unavailable. Capability metadata distinguishes existing fixed feet from missing feet. Contact and later stages retain their floating-foot whole-body layouts. Acceptance of this adapter requires native matching and closure evidence; source changes alone do not qualify it.

Keyed initial poses use exact native model keys and units. The caller obtains the optional `native_schema` from KinematicsSolver for non-Human variants, validates spherical axes and verifies native loop closure. The original Human 48-coordinate initialization contract is retained. These seeds do not transfer missing joints from another model.

Current native qualification: complete 55/46-frame trajectories exist for both captures on all 13 registered variants. The 12 floating-foot or upper-body variants have approximately 13--24 mm mean measured target RMS; their eight- or fourteen-target layouts must be considered when comparing scores. The fixed-foot FullBody stage has 27/34 mm mean and 43/53 mm peak RMS on its eight active targets. Both complete FullBody reruns, 95 native contract/regression tests and final entire disabled-feature Human output/default-explicit parity have natural process exit-zero receipts. Original marker offsets and saved FullBody model bytes are preserved. The three back-marker RMS values remain separate diagnostics; fixed-foot FullBody back RMS is approximately 52/61 mm, so it is not the closest body-marker representation.

FullBody uses alternating native upper-body and six-coordinate pelvis fits. Every accepted block refreshes the native loop-closure guesses. Larger central differences on this stage avoid the earlier finite-difference stalls, while the separate coordinate blocks keep a failed pelvis step from freezing the arms. The monolithic 376--532 mm fits and intermediate 128--169 mm finite-difference fits remain rejected. Scapula and spine bounds are enforced through the backswing exactly as on the other models.

The renderer measures the two welded foot solids relative to the moving pelvis, then composes the native pelvis/foot transforms to World. Direct World-to-welded-foot frame variables are rejected by native KinematicsSolver. Independent native RED/GREEN probes reproduce the expected fixed-foot transforms within 1e-7 across two moving-pelvis poses. A capture-free regression also checks two upper-body poses, fixed-foot transforms and native 1920x1080 PNG output.

Final all-model replay and media delivery remain in progress. Baseline, Slim and Quat have native exit-zero replay/export receipts for 24 neutral 1080p videos, with all 1,212 frames independently decoded and Desktop copies hash-verified. The eight previously qualified Human videos remain unchanged. The older all-model replay passed numerical assertions but hung at shutdown (watchdog 125); smaller final native batches replace that acceptance gap. No source merge or full all-model media completion is implied by completed fit or schema checks.
