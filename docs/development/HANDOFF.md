# Implementation Handoff — Exploratory GS3DX Simscape Model (#10950)

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-simscape-gs3dx`
- Branch: `feat/simscape-gs3dx-exploratory`
- Baseline commit: `2d5830d18` (origin/main)
- Implementation commit: `SELF`
- Pull request: #10963 (draft) https://github.com/D-sorganization/UpstreamDrift/pull/10963
- Governing issue/epic: #10950 (children #10951–#10959, #10985, #10986, #11011; in progress #10979)
- Lease session: `claude-gs3dx-shape-20260928` (#10979, renewed to 2026-09-29T15:22Z)
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
  follow-up GS3DX-10 (#10979) drives the legs and refits the pelvis inputs; balance through the leg servo holds the body over its feet (section 7 of `docs/FIT.md`). GS3DX-11
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
  19. #10979 upper-body tracking (`docs/FIT.md` sections 3 and 6). Regularized whole-body IK options
      (`posture_weight`, `smooth_weight`, `backward`, `gap_weight`; defaults off, test_gs3dx_fit 7/7):
      the unregularized trunk jumped branches (torso 464 deg range); with 0.01/0.02/false/0.5 the whole
      trial fits 6.3 mm RMS median (25.2 p95), pelvis step <= 1.3 mm to impact (scratch
      `ik_full_reg.mat`, ~1 h). `GS3DX_FitLegs` REBUILT from it (torsion +23.7/-16.8, clamp 1.6 mm,
      knees 27/32 mm median). `gs3dx_upper_body_reference` -> 12 chart references (12 Hz).
      `gs3dx_build_fit_track` -> `GS3DX_FitTrack`: each upper-body chart gets feedforward + PD
      (`gs3dx_track_torque`, `UpperBodyTracking`, ModelingMode stays 0 = free pelvis), inertia-scaled
      gains (`gs3dx_track_gains`, 6 Hz, zeta 1), and REWIRED inputs: the original LW chart reads the
      LEFT SCAPULA angles and LE/LF/RF/RW read tags no Goto writes (wrists ran away 14,800 deg). No
      block added (967). `gs3dx_track_learn` (ILC from the Simscape log; signal logging passes the
      license limit): full swing to impact angle RMS 0.96 -> 0.25 deg in one update, but PD RMS
      22.1/10.3/14.7/20.7 N*m grows from iteration 3 (Q-filtering all of F did not help; torso keeps
      34 N*m at 0.19 deg); the model holds the iteration-2 feedforward. Whole body to impact: pelvis
      206 mm RMS / 502 mm at impact off the capture (steady X drift, feet slide 8 cm, lift 9/20 cm:
      the body TIPS OVER ITS FEET - joint tracking alone does not balance); vertical GRF 0.20 BW RMS
      from the capture, peak 1.43 BW is the landing at 0.05 s. 17 min per full-swing simulation.
  20. #10979 balance (`docs/FIT.md` section 7): `gs3dx_build_fit_balance` -> `GS3DX_FitBalance`
      (973 compiled; contact-check sensors +10 = 983). Whole-mechanism Inertia Sensor -> Goto ->
      State-Space rate; `Leg Torque Commands` -> MATLAB Function `gs3dx_balance_command`: servo
      angle + G(t) shift (shift = -Kp e - Kd de, 3-axis COM error, limit 0.1 m) + per-leg G(t) kf
      e_foot (ankle GlobalPosition via Bus Selectors vs `ref.feet`). `gs3dx_balance_gain` = foot-fixed
      damped inverse leg Jacobian (12 x 3 x frames). To impact: pelvis off the capture at impact 503 -> 62-81 mm (FIT.md
      table; 3-axis + foot gain 1: pelvis 44 RMS / 81 mm at impact, COM 38.5/52.0 mm). Gains raised
      to Kp 3 / Kd 0.4 (builder default): COM 17.9/25.7 mm, pelvis 27/73 mm. Support peak 2.4-2.7 BW
      just before impact (capture 1.27) is DEMANDED BY THE COM REFERENCE (pelvis ref + balance-off
      offset implies 2.31 BW; 40 mm vertical range vs capture 19 mm; model COM within 6 mm of it) -
      not a foot defect: per-contact forces (`gs3dx_contact_check` `.contacts`) show a broad hump
      1.23-1.31 s, trail foot on its inner toe sphere. Rejected: ankle servo 10 N\*m/deg (falls, pelvis
      906 mm at impact); foot-tilt feedback via ankle `Rotation Transform` (worse COM 49/75 mm; patch
      kept in the session scratchpad only). The model is built with axes 3, gains [3 0.4], foot 1.
      Earlier "zero logged samples" errors were very likely a FULL C: DRIVE (Simulink turns off
      recording under low disk space), not only concurrency.
  21. #10979 references and inertia (2026-09-28): `com_ref` option of `gs3dx_build_fit_balance`
      (World path, exclusive with `com_offset`) + `gs3dx_capture_com_reference` (capture COM from
      `gs3dx_kinematic_grf`, translated to the model's at address). Trial, not saved: support peak
      1.44 BW (was 2.63; capture 1.27), slip 44/53 mm (58/88), pelvis 42/72 mm (27/73) - the spike
      is the reference, confirmed. `gs3dx_inertia_audit` + `gs3dx_segment_inertia` + de Leva
      `.gyration`/`.length` in `gs3dx_anthropometry`; `docs/INERTIA.md`: limb longitudinal inertia
      0.43-0.65 of de Leva, lower trunk 0.53, foot close (1.08). Agents (agy, uncommitted worktrees
      `agy-gs3dx-{shape,render}`): GS3DX_Shape (ellipsoid solids with custom de Leva inertia/COM)
      and `gs3dx_render` (headless KinematicsSolver + MATLAB graphics) in progress.
  22. #10979 shape and rendering (2026-09-28): `gs3dx_render` (headless stills/video from a
      KinematicsSolver pose of every Solid; `docs/RENDERING.md`; stills in
      `docs/screenshots/GS3DX_{FitBalance,Shape}_*.png`). `gs3dx_build_shape` -> `GS3DX_Shape`:
      custom de Leva moments on limbs and head (audit 1.00), de Leva centres of mass on thighs,
      shanks, upper arms and forearm halves (joints measured on each solid's z axis, proximal +z),
      ellipsoids for thighs/shanks/hands/head via `PortConnectivity` rewiring; 973 compiled (one
      extra visual solid costs 7). `gs3dx_balance_reference` shared with FitBalance. Finding: the
      Shape COM reference is within 1 mm RMS of FitBalance's (12.4 vs 12.9 mm from the capture's)
      and balance runs repeat it (2.64 BW offset, 1.44 BW capture COM): segment inertia does not
      cause the impact spike (`docs/SHAPE.md`). The earlier lower-trunk 0.53 ratio was an audit
      length artifact (`docs/INERTIA.md`).
  23. #10979 render views (2026-09-28): `gs3dx_render`'s "face-on" drew down the line and
      "down-the-line" drew face-on from behind; now face-on = camera on +X (az 90), down-the-line =
      camera on -Y (az 0), unknown views error, `out.view` reported and tested. All
      `docs/screenshots/GS3DX_*` stills re-rendered.
  24. #10979 impact spike resolved (2026-09-28): per-segment vertical COM, model (regularised IK,
      `gs3dx_render` poses) vs capture, agrees to 1 mm for legs/arms/club; the 12.9 mm RMS is the
      trunk (10.3) and head (3.5). The trunk part is the capture's C7-skin proxy: with the
      shoulder-centre -> hip-centre trunk (`gs3dx_kinematic_grf(trunk="joint_centres")`) the
      capture matches the model to 4.0 mm RMS. As balance reference it gives 1.77 BW peak (vs
      2.64 offset / 1.44 C7), pelvis 22 mm RMS, COM vert 2.9 mm, slip 38/16 mm: best run, and
      `GS3DX_Shape` is saved on it (`docs/SHAPE.md`). The head is rigid with the upper trunk
      (no neck joint, no head IK target) and rises ~50 mm in the downswing; a driven neck needs
      ~10 compiled blocks and Shape has 2 under the 975 cap.
  25. #10979 neck (2026-09-28): `GS3DX_Neck` (`gs3dx_build_neck`, `docs/NECK.md`) = Shape + a
      Universal Joint between Rigid Transform5 and the Neck's "Bottom of Neck" frame (the neck
      base), both axes input motion from `NeckReference` (capture head frame relative to the model
      upper trunk; axial turn dropped). 6 blocks, paid by deleting the 4 massless elbow/shoulder
      spheres (1 compiled block each, not ~7): 975. Head vertical travel to impact 213 -> 113 mm
      (capture 55), vertical error 52 -> 37 mm RMS; facing error is the trunk's. Adding a joint
      renumbers KinematicsSolver IDs (block-path order), so `gs3dx_render` now matches joints by
      `gs3dx_joint_keys` (block path + primitive); IK/fit helpers still hard-code j15/j18/j19 and
      are only used on neck-free variants. agy's `gs3dx_capture_head_frame` +
      `gs3dx_capture_address_transform` ported (reviewed).
  26. #10979 human shape (2026-09-28/29): `GS3DX_Human` (`gs3dx_build_human`, `docs/HUMAN.md`,
      965 compiled; current state in Next Steps item 2) = Neck with: the unused "Inertia Sensor" subsystem removed (-75); trunk,
      neck, shoulder, arm, forearm cylinders and the misaligned `ZeroMassHipReference` bar hidden
      (inertia kept) and massless ellipsoids drawn on each parent's exposed reference frame (a
      cylinder with custom frames shows only its end frames); head radii 90x90x105 mm;
      `NeckAddress` (-24.2, -7.0 deg) aims the neck at the capture head (address error 126 ->
      40 mm); `FaceSquareRoll` -20.73 deg squares the face at address (driver-head visual,
      10.5 deg loft; still 14 deg open at impact from the posed hands); sprung revolute
      midfoot (MTP) joints with a 0.25 kg forefoot carrying the toe contact spheres, foot COM
      preserved. `gs3dx_reference_pose` renders with the servo leg reference (IK feet were
      reversed: no toe target). Tests: human 5/5, neck 5/5, render 4/4.

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
  Plus (2026-09-28, one MATLAB process) `test_gs3dx_fit_balance` 6/6 (leak max 0.975 / median
  0.303, bounds 0.98 / 0.31; 0.3 s COM error 14.6 RMS / 24.9 max mm, bound 26 mm),
  `test_gs3dx_fit_track` 5/5, `test_gs3dx_leg_kinematics` 4/4, `test_gs3dx_contact` 7/7.
  Plus `runtests('test_gs3dx_fit_track')` -> 5 passed / 0 failed (2026-09-27; 0.3 s replay 0.283 deg,
  worst joint 0.491 deg, PD 16.7 N\*m) and, after the FitLegs rebuild, `test_gs3dx_fit_legs` 3/4 + the
  standing test rerun alone 1/1 (slip 1.4/0.9 mm, lift 0, pelvis 28 mm). Run alongside another
  MATLAB simulation the standing test errored (zero logged samples in `gs3dx_stance_frames`):
  do not run GS3DX simulations in parallel processes (shared cache). test_gs3dx_fit 7/7 after
  the IK options. The full suite was not rerun.
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
2. #10979 human, in progress (2026-09-29, owner reviews of the renders). State:

   - **Done, tested (human 9/9, render 5/5):** five
     contacts per foot; fixed "Neck Address" transform; neck pivot at C7 (`NeckLength`
     7.717 in; lift is `-lift*Q0(:,3)`); driver head = massless File Solid of
     `models/gs3dx_driver_head.stl` (Tools `rate_of_closure` Driver 10.5,
     `models/README_DRIVER_HEAD.md`; File Solid needs `UnitType Custom`); neck shape now
     runs head centre -> 50 mm past C7 and a "Trapezius" ellipsoid on UpperTorsoTop joins
     it to the shoulders (owner: neck gap; test `neck_joins_the_head_and_the_trunk`).
     965 compiled / 788 uncompiled. Renderer returns `out.focus` (bug found by the mesh
     test; fixed).
   - **Start kick FIXED:** bisected with scratch `human/bisect/bisect_human.m` (Neck copy +
     builder stages). Not the feet/midfoot/foot inertia: the neck address moved the address
     COM (-4.7, -0.9, +5.2) mm while `BalanceCOMRef` was anchored at Neck's COM, so the
     balance loop kicked both feet off the ground at t = 0. `gs3dx_build_human` now
     re-anchors `BalanceCOMRef` after saving (1 ms balance-off `gs3dx_contact_check`;
     `report.com_shift`; test `balance_reference_is_anchored_at_this_body`). Support now
     0.46-1.97 BW (was 0-10.3).
   - **Balance drift FIXED (2026-09-29):** the foot layout, not the midfoot or the head. A
     Neck-feet variant with the same head + anchor drifts 18.0 mm; the first five-contact
     layout 42.9 mm even locked (159 mm at 100 N\*m/rad). COP (scratch `human/cop.m`): both
     models stand on the front contacts, lead foot on its inside front corner, and the
     Human's front support stopped short of the toe tip. Layout sweep (scratch
     `human/layout1.m`, `layout2.m`; table in `docs/HUMAN.md#why-the-human-drifted`) ->
     built layout: one heel on the axis, balls unchanged, Big Toe under the tip 50 mm
     inside, Lesser Toes at 0.945 of the foot length 50 mm outside, both on the forefoot.
     Built model (scratch `human/test9.m`): pelvis 20.2 mm RMS locked, 21.9 mm at 2,000
     N\*m/rad (COM 14.0, support 0.52-2.02 BW); `midfoot_stiffness` default now 2,000
     (100/800 fold). Human 9/9 + render 5/5 pass on the rebuilt model.
   - **Trap:** a failed `gs3dx_build_human` leaves `GS3DX_Human.slx` as a copy of
     `GS3DX_Neck`; check size/mtime before trusting a run. `gs3dx_render` and
     `gs3dx_contact_check` close the models they load: tests reload after them.
   - **Finish (2026-09-29, scratch `human/fin1.m`, `fin2.m`):** all workspace references
     span 654 frames (1.814 s), so `gs3dx_contact_check(..., stop_time=1.809)` runs through
     the finish. Trail heel up onto Big Toe then Lesser Toes (ankle rise 142 mm vs capture
     129), lead foot onto its outside edge: works. OPEN: at 1.55-1.6 s the trail toe carries 0.73-0.83 BW with the lead foot
     at 0, and the unloaded lead foot slides 234 mm (capture 36 mm, turns 30 deg in place);
     pelvis 56.6 mm RMS after impact. NOT the COM: along trail ankle (0) -> lead ankle (1)
     the model COM ends at 0.60, its reference 0.58, the capture's 0.54 (scratch
     `human/fin5.m`; TRAP: `gs3dx_kinematic_grf` .com is in target axes with the LAB
     origin: subtract `tf.S.' * tf.origin` (`gs3dx_capture_address_transform`) before
     comparing with joint centres, which use the address-waist origin). Neck and Shape do
     the same and worse (support 0 near 1.4 s, trail slide 0.55 m; scratch `fin4.m`). The
     trail foot runs 60-105 mm ahead of its reference from 1.45 s. Hypothesis: the stiff
     midfoot makes the foot pivot on the toe tip instead of hinging at the ball; running
     `human/fin6.m` (MidfootStiffness 300 vs 1e4 through the finish) to test it.

3. #10979 learning drift: per-joint PD torque over more iterations (`out.joint_pd`), then a
   forgetting factor or PD-only loop joints.
4. Optional owner inputs: the golfer's height/mass (mass is not identifiable from markers).
5. Reproducibility: three inputs exist only in model workspaces (FitBalance `com_offset`,
   Shape `com0`, Neck `NeckReference`); add a tool per input and save the values beside
   the models so `docs/DESIGN_REPORT.md#reproducing-the-model` runs end to end.
