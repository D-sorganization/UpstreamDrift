# Implementation Handoff — Exploratory GS3DX Simscape Model (#10950)

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-simscape-gs3dx`
- Branch: `feat/simscape-gs3dx-exploratory`
- Baseline commit: `2d5830d18` (origin/main)
- Implementation commit: `SELF`
- Pull request: #10963 (draft) https://github.com/D-sorganization/UpstreamDrift/pull/10963
- Governing issue/epic: #10950 (children #10951–#10959, #10985, #10986, #11011; in progress #10979)
- Lease session: `claude-deskcomputer-20260926-gs3dx`
- Development log: `DL-#10950`

## Objective and Status

- Objective: build agent-editable `GS3DX_` clones of `GolfSwing3D_Kinetic.slx` in
  `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/`,
  move shoulders and hip to quaternion joints, and grow to a full-body model under
  the 1,000 non-virtual-block Home-license limit, without touching the originals.
- Status: GS3DX-1 (#10951) scaffold, GS3DX-2 (#10952) regression harness, GS3DX-3
  (#10953) block budget, GS3DX-4 (#10954) `GS3DX_Slim`, GS3DX-5 (#10955) `GS3DX_Quat`
  quaternion shoulders, GS3DX-6 (#10956) quaternion hip, GS3DX-7 (#10957) lower body and
  GS3DX-8 (#10958) weld stance + integration run and GS3DX-9 (#10959) visual QA done;
  follow-up GS3DX-10 (#10979) drives the legs and refits the pelvis inputs. GS3DX-11
  (#10985) data audit and GS3DX-12 (#10986) ground contact done (`GS3DX_FullBodyContact`).
  #11011 typical segment masses done (`GS3DX_Golfer`, 80 kg de Leva) plus the force-plate-free
  total GRF from the capture (`gs3dx_kinematic_grf`); owner: no force plates will be captured.
  #10979 in progress (owner: anthropometry unknown, match it to the data): `GS3DX_Fit` segment
  lengths from the capture, a least-squares whole-body IK, and the hand-on-grip geometry from the
  capture (`gs3dx_fit_grip`; out-of-sample RMS max 32 -> 17 mm).
- Completed:
  1. `gs3dx_setup` (session path, cache redirect, no `savepath`), `gs3dx_names`,
     `gs3dx_assert_no_shadowing`, `gs3dx_save_model` (write guard),
     `gs3dx_original_manifest`, `gs3dx_clone_baseline`.
  2. `models/GS3DX_Baseline.slx` + `GS3DX_KD_{Gimbal,Revolute,Universal}.slx` generated
     headlessly in R2025b; 12 `ReferencedSubsystem` blocks re-pointed.
  3. `tests/test_gs3dx_safety.m` 5/5 pass.
  4. #10952: `gs3dx_simulate`, `gs3dx_flatten_bus` (1 kHz grid), `gs3dx_compare`
     (`max_abs <= atol + rtol*range`, atol 1e-6, rtol 1e-3) and
     `gs3dx_capture_original_baseline`. Baselines per drive (`gs3dx_drive`,
     `gs3dx_baseline_file`): `original_GolfSwing3D_Kinetic_impact_0p3S.mat` (regression
     drive, 344 steps) and `..._persisted_0p3S.mat` (model's saved inputs, 4,236 steps).
     GS3DX_Baseline matches both on all 413 signals with identical step counts.
  5. #10953 (agy draft, reviewed and corrected): `gs3dx_block_budget`,
     `gs3dx_license_limit_probe`, `docs/BLOCK_BUDGET_FINDINGS.md`. Limit verified exactly:
     1,000 non-virtual blocks simulate, 1,001 fail at compile.
  6. #10954: `gs3dx_build_slim` copies Baseline/KD to `GS3DX_Slim`/`GS3DX_KDS_*` and
     `gs3dx_direct_torque_drive` rewires each torque axis converter -> joint InputTorque
     (drops Ideal Torque Source, Rotational Multibody Interface, Reference). 672 -> 609
     non-virtual blocks (63 saved). Slim matches the original on the impact drive
     (413/413 signals, 344 steps, clubhead max diff 6e-14 m).
  7. Found the persisted drive is ill-conditioned (clubhead > 4 km/s by 0.3 s; a 1e-9
     RelTol change moves the clubhead 2.4 m), so it cannot prove any restructuring;
     the impact drive is the regression drive. `docs/SENSITIVITY_FINDINGS.md`.
  8. #10955: `gs3dx_build_quat` copies Slim/KDS_Gimbal to `GS3DX_Quat`/`GS3DX_KDS_Spherical`;
     `gs3dx_gimbal_to_spherical` swaps the Gimbal Joint for a Spherical Joint keeping the
     subsystem interface (ports, mask, 21 bus elements) via `gs3dx_xyz_map` (Euler X-Y-Z <->
     quaternion; `XYZ Torque` and `XYZ Kinematics` MATLAB Function blocks, `Angle Reference`
     integrator for the 360-degree branch). 609 -> 599 blocks. Isolated rig
     (`gs3dx_joint_rig`): all 21 signals match the Gimbal to <= 2e-7 of peak at RelTol 1e-8.
     Full model: the Low-priority right shoulder assembles differently in a closed chain, so
     `gs3dx_pinned_drive` pins its start state; then Quat converges to Slim as RelTol tightens
     (clubhead gap 13.9 mm / 0.54 mm / 0.039 mm at 1e-3/1e-5/1e-7), is closer than Slim to the
     converged answer at every tolerance, and takes 7-9% fewer steps.
     `docs/QUATERNION_SHOULDERS.md`.
  9. Refactors: `gs3dx_physical_graph` + `gs3dx_redraw_nets` (shared net surgery),
     `gs3dx_copy_models` (copy guard), `gs3dx_simulate` `block_parameters` option,
     `gs3dx_repoint_references` skips subsystem-reference internals.
  10. #10956: `gs3dx_quaternion_swap` is the shared swap core (port map per joint type);
      `gs3dx_gimbal_to_spherical` and `gs3dx_bushing_to_6dof` are thin specs over it.
      The hip Bushing Joint becomes a 6-DOF Joint (prismatics copied, rotation quaternion);
      `GS3DX_Quat` is 594 blocks (15 below Slim). `gs3dx_hip_rig` (+ `gs3dx_rig_scaffold`,
      `gs3dx_rig_simulate`, shared with the refactored joint rig): all 30 HipLogs signals match
      the Bushing to <= 1.9e-6 of peak. Convergence with the hip: 14.8 / 0.61 mm at 1e-3 / 1e-5.
  11. #10957/#10958: `gs3dx_build_lower_body` copies Quat to `GS3DX_FullBody` and adds a
      `Lower Body` subsystem from `gs3dx_leg_table` (de Leva anthropometry; Spherical hip,
      Revolute knee, Universal ankle KDS). `gs3dx_stance_frames` measures pelvis/shoulder frames
      on Quat at t0 (unsaved) to lay out a leg frame; feet are framed rigidly to World (weld).
      751 blocks (cap 900). Starts in Quat's exact state, knees assemble at -28.14 deg, 0.3 s
      impact window completes (348 steps); passive legs make the pelvis diverge and the right
      knee hyperextend (+67 deg). `gs3dx_contact_trial`: contact = +2 blocks, 1.8x wall time.
      `docs/FULL_BODY.md`. `gs3dx_layout_qa` renders diagrams to `docs/screenshots/` (0 overlaps).
  12. #10985 data audit: `gs3dx_capture_stance` reads `data/C3D_TA_Driver.c3d` via pyenv ezc3d
      (654 frames, 360 Hz, **no force plates / analog: no GRF data**); stance 0.65 m ankle width,
      foot yaw -2.15/+8.15 deg -> `gs3dx_leg_table` `.stance`. Mass double-count (upper body 77.6 kg,
      FullBody 109.4 kg), foot/thigh proxy conflicts, and the impact drive's pelvis path being out
      of reach of planted feet: `docs/DATA_AUDIT.md` (owner decision on trunk mass pending).
  13. #10986 ground contact: `gs3dx_leg_fk` (matches Simscape to 1e-9) and `gs3dx_leg_ik` (damped
      Gauss-Newton); `gs3dx_build_contact` copies FullBody to `GS3DX_FullBodyContact`: welds removed,
      3 sole spheres per foot on one Infinite Plane, pelvis joint unactuated (NoTorque, 4 converters
      deleted), stance-hold leg servo (matrix Gain on [q; qd] + existing Constant), IK start angles and
      rates (High). `gs3dx_contact_check`: Newton momentum balance, foot slip/lift, GRF. From rest it
      stands (slip 2 mm, no lift, Newton 0.71/3.22 N*s); the impact drive tips it over the right foot
      (131 N*s start momentum, gain-independent) -> needs the #10979 refit. `docs/GROUND_CONTACT.md`.
  14. **Block budget correction:** the Home license counts the _compiled_ model; FullBody is 751
      uncompiled but 945 compiled (Quat 594 -> 740). `gs3dx_block_budget(mdl, compiled=true)` adds
      `.compiled_total`; FullBodyContact compiles to 967 with a 25-block validation reserve.
  15. #11011 typical masses: `gs3dx_anthropometry` (one de Leva table: masses + COM fractions;
      `gs3dx_leg_table` now reads its leg masses from it, values unchanged); `gs3dx_build_golfer`
      copies FullBodyContact to `GS3DX_Golfer`, sets 15 upper-body solids to `Golfer*` kg variables
      (block count unchanged, 967 compiled). Sensed mass 80.393 kg = 80 + 0.393 equipment (was 109.4).
      From rest it stands (Newton 0.46/2.37 N\*s, slip 1.4 mm, no lift); the impact drive still tips it
      (start momentum, and the drive torques were fitted to the 77.6 kg upper body).
      `gs3dx_capture_markers` (C3D loader, club clusters, impact frame, address target frame; shared
      with `gs3dx_capture_stance`); `gs3dx_kinematic_grf`: total GRF = sum m a - (M+m_club) g from
      the capture COM: address 1.001 BW, pre-impact peak 1.24-1.33 BW ~60 ms before impact (6-15 Hz),
      club must be included (+0.57 BW). Plan without force plates: `docs/ANTHROPOMETRY.md`.
  16. #10979 anthropometry from the data (`docs/FIT.md`): `gs3dx_capture_joint_centres` (joint
      centres + segment lengths from the markers; RShoulderTop rebuilt from RShoulderBack),
      `gs3dx_fit_lengths` + `gs3dx_build_fit` -> `GS3DX_Fit` (11 upper-body cylinders re-pointed at
      `Fit*` vars the drive file cannot overwrite, ThighLength/ShankLength set; block count unchanged,
      967 compiled; KinematicsSolver FK: forearm 0.280, pelvis-hub 0.490 vs data 0.278/0.488).
      `gs3dx_whole_body_ik`: KinematicsSolver refuses over-determined targets (status -3), so
      lsqnonlin LM over 33 independent coordinates with the solver as FK (5 ms, right arm closed by
      the grip weld loop); per-body marker offsets calibrated by alternation; forward+backward
      tracking; gap-filled samples (`jc.gap`: club head 445, 519-551, 653-654) are dropped - fitting
      them lost the whole follow-through (0.5 m). Whole trial (63 min): RMS 7.1 mm median, 20.3 mm
      max. Out of sample (offsets from the backswing, fit the last 0.25 s): pelvis/legs/shoulders
      within 1-3 cm, club head 40 mm, wrists 52/88 mm -> the model's hand-on-grip geometry (literal
      grip cylinder lengths, hand sphere radius 2 in, standoffs) is the next thing to fit. Frame
      `Rotation` outputs = intrinsic XYZ (verified 3e-16).
  17. #10979 hand-on-grip geometry (`docs/FIT.md` section 4): `gs3dx_fit_grip` (Kabsch club pose
      from the six club markers, sphere-fit functional wrist centres of WristTop in the club frame,
      shaft axis from the IK model's hand-sphere centres). The original grip put both wrists 1 in to
      the same side; the data has them 77 mm apart along the grip and 72-74 mm apart across it, on
      opposite sides. The split across the shaft is not identifiable (the club-head marker offset
      absorbs a sideways shaft shift), so it is split equally. `gs3dx_build_fit` re-points the 6 grip
      solids at `FitButtToLeadHand` 3.08, `FitHandSpacing` 3.03, `FitGripToShaft` 4.39 (butt to
      shaft stays 10.5), `FitLeft/RightWristStandoff` 1.46 in, and flips the lead standoff by swapping
      its frames' end features (parameter-only, 967 compiled). Out of sample: wrists 88/52 -> 18/15 mm
      median, club head 40 -> 29, RMS max 32 -> 17 mm, wrist marker offsets 6-7 -> 2.8 cm; the grip
      estimate is a fixed point within 0.07 in. `ik.model` now records the fitted model. Whole trial on
      the fitted grip (43 min): RMS 7.0 median / 16.1 p95 / 19.1 mm max, offsets <= 47 mm (scratch
      `ik_full_grip.mat`, not committed; regenerate with `gs3dx_whole_body_ik(jc, verbose=true)`).
  18. #10979 leg servo references (`docs/FIT.md` section 5): `gs3dx_leg_reference(ik, jc, cap)` -> 12
      servo angles/rates from the IK pelvis path (joint j1 follower = 'Lower Torso') and the measured
      foot path (ankle centre + ankle/ToeIn/ToeOut triad rotation; not planted: trail heel +36 mm at
      impact), feet levelled +-9.3 mm, 10 Hz Butterworth, a constant torsion per foot (+24.0/-17.9 deg,
      the universal ankle has no axial DOF), exact `gs3dx_leg_ik`, reach clamp <= 1.5 mm (post impact).
      Knees vs capture to impact 19/20 mm median, 48/55 mm at impact (the missing ankle axial DOF).
      `gs3dx_build_fit_legs` -> `GS3DX_FitLegs`: Leg Torque Commands Constant -> From Workspace
      (one for one, 967 compiled), start state + `LegReferenceStart` struct (pass it after the drive:
      the drive file sets PlaneTilt 22.5 vs the saved 30 deg, which tilted the start pelvis 7.5 deg),
      ground under the start feet (capture frame = World). Standing from rest: slip 1.8/1.5 mm, lift 0,
      Newton pass. The impact drive is PASSIVE (ModelingMode 0 zeroes every upper-body torque).

## Files and Decisions

- The originals are only `copyfile`d, never loaded or saved, so R2025b cannot re-save them
  (the original .slx was last saved in R2025a).
- The exploratory folder sits outside `matlab/src/`, so `setup_matlab_environment`'s
  `genpath(src)` never adds it to the path.
- Shadowing risk is real: during the probe, a same-named scratch copy won `which()`.
- The baseline is stored in double precision: float32 rounding alone (1.9e-6 on the
  constant GolferMass) exceeded atol; the tolerance was not widened.
- The model needs `matlab/src/functions` on the path (`HexPolyInputFunction`), so
  `gs3dx_setup` adds it as `dependency_dirs`.
- Per-joint cost: KD Gimbal 37 non-virtual blocks (21 converters), Universal 29 (17),
  Revolute 21 (13). Each torque axis uses Simulink-PS -> Ideal Torque Source ->
  Rotational Multibody Interface + Reference (4 blocks) although the joint axis is
  already `InputTorque`; a direct PS torque input saves 3 blocks per axis (63 total).
- Simulink physical lines: a branched net lists its segments only on the trunk
  (`LineChildren`, which includes the trunk itself); branch segments report no parent or
  end ports. `gs3dx_direct_torque_drive` finds nets by union-find over all segments.
- Count blocks on a freshly loaded model: after a simulation in the same session
  `find_system` counts differ (+101 on Slim).
- The KDS Bus Creator is named `Bus
Creator` (newline); find it by BlockType.
- A MATLAB Function block is direct feedthrough, and the sensed acceleration depends on the
  torque, so torque and kinematics mapping are two blocks (one would form an algebraic loop).
  The damping Constant needs SampleTime 0: constant-time blocks may not touch the shoulder
  input-function loop.
- A Spherical Joint has one target per quantity: the mask init asserts equal Rx/Ry/Rz
  priorities, so set all six priorities in one `set_param` call.
- Impact-input mode sweep: mode 3 ill-conditioned, mode 2 torqued but 1e8 N\*m, mode 1 fails
  early. Not gimbal lock: Quat (quaternion hip) fails at 19.9 ms with hip Y at 49° while the
  mode-1 hip torque command reaches ~2e7 N\*m. The drive blows up, not the joint.
- `gs3dx_physical_graph` on a subsystem: `find_system` lists the subsystem itself; exclude it.
- Rig constant signals are logged once; `gs3dx_rig_simulate` expands them to the output grid.
- The hip rig's default torques keep the Bushing's Y angle within -72..62°; 6x larger
  torques reach 84° (near singular, 1.8e4 deg/s).
- Welded legs close a loop through the pelvis joint: Simscape ignores targets when every joint
  in a loop has one, so the leg hips have no target (knee/ankle Low pick the branch).
- A freshly added Subsystem Reference does not list its contents to `find_system`; resolve
  ports from the referenced file (`gs3dx_port_source`).
- Infinite Plane ports: frame on the left, geometry on the right; a Brick with exported
  geometry is the other way round.
- `*.png` is git-ignored outside the root `docs/`; the exploratory screenshots are force-added.
- A name on a branched signal line is drawn on every branch; name signals on the outputs of
  a virtual Demux instead (`Actuator Torque` in the hip/shoulder subsystems).
- The hip bus element is `AngularPosition Z` (space) in the bus; logged Datasets expose it as
  `AngularPosition_Z`.
- Deleting one line of a physical net deletes the whole net: removing the foot weld also cut
  ankle Distal -> Foot COM, which `gs3dx_build_contact` redraws and asserts.
- `gs3dx_stance_frames` closes the model it simulates: read model-workspace values first.
- The impact drive file sets `LowerTorsoMass`, `UpperTorsoMass`, `*ShoulderMass`, `*ArmMass` (unused
  by any solid); the new masses use `Golfer*` names so a drive can never overwrite them.
- A scratch `poly.m` in the MATLAB run folder shadowed the built-in (broke `butter`); renamed.
- `gs3dx_build_lower_body`/FullBody tests still cap the _uncompiled_ count at 900; FullBody's
  compiled count (945) would fail that cap. Left as is (existing test); flagged in the PR.

## Validation

- From an empty cwd: `matlab.exe -batch "addpath('<exploratory_gs3dx>'); info=gs3dx_setup(); runtests(fullfile(info.root,'tests'))"`
  → 85 passed / 0 failed (2026-09-27, with test_gs3dx_fit 7/7 incl. the fitted-grip out-of-sample
  IK and the grip fixed point; the IK test takes ~5 min). Earlier 83 (test_gs3dx_fit 5).
  Plus `runtests('test_gs3dx_fit_legs')` → 4 passed / 0 failed (2026-09-27; setup runs a
  whole-body IK, 11–25 min; clamp 0.0 mm, knee median 25/31 mm, p95 58/61 mm; standing slip
  1.8/1.5 mm, lift 0).
  Earlier 78 (after #11011: golfer 6, capture +2 kinematic GRF; earlier 70 after #10985/#10986: capture 4, leg kinematics 4, contact 7
  new; the FK test failed once on a closed-model handle, fixed and rerun 4/4). The capture tests
  need MATLAB pyenv with ezc3d (they are skipped otherwise).
- `gs3dx_layout_qa` on the rebuilt diagrams: 0 overlapping blocks.
- The original model simulates headlessly (0.3 s of swing takes about 223 s cold); the model workspace is embedded (676 vars).

## Blockers and Risks

- MATLAB tests need a licensed R2025b; they are not part of the Python CI.

## Next Steps

1. Owner review of draft PR #10963; mark it ready once reviewed (a GUI open-check via
   computer use needs the owner to grant app access interactively).
2. #10979 upper-body drive (owner: inverse dynamics plus PD tracking): inverse dynamics of the
   whole-trial IK motion as feedforward joint torques plus PD tracking of the IK joint angles on
   `GS3DX_FitLegs`, in place of the passive impact drive; then read the pelvis residual and compare
   the summed contact GRF with `gs3dx_kinematic_grf` (1.24-1.33 BW peak ~60 ms before impact).
3. Optional owner inputs: the golfer's height/mass (mass is not identifiable from markers).
