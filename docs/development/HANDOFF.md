# Implementation Handoff — Exploratory GS3DX Simscape Model (#10950)

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-simscape-gs3dx`
- Branch: `feat/simscape-gs3dx-exploratory`
- Baseline commit: `2d5830d18` (origin/main)
- Implementation commit: `SELF`
- Pull request: #10963 (draft) https://github.com/D-sorganization/UpstreamDrift/pull/10963
- Governing issue/epic: #10950 (children #10951–#10959)
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
  follow-up GS3DX-10 (#10979) drives the legs and refits the pelvis inputs.
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

## Validation

- From an empty cwd: `matlab.exe -batch "addpath('<exploratory_gs3dx>'); info=gs3dx_setup(); runtests(fullfile(info.root,'tests'))"`
  → 55 passed / 0 failed in 403 s (2026-09-26, after the #10959 layout fixes and a rebuild of
  Quat/KDS_Spherical/FullBody; the Simulation-tagged tests take most of the time; select
  `~HasTag('Simulation')` for the structural tests).
- `gs3dx_layout_qa` on the rebuilt diagrams: 0 overlapping blocks.
- The original model simulates headlessly (0.3 s of swing takes about 223 s cold); the model workspace is embedded (676 vars).

## Blockers and Risks

- MATLAB tests need a licensed R2025b; they are not part of the Python CI.

## Next Steps

1. Owner review of draft PR #10963; mark it ready once reviewed (a GUI open-check via
   computer use needs the owner to grant app access interactively).
2. #10979: fit leg torques, refit the pelvis inputs with legs attached, and add an
   energy audit (`docs/FULL_BODY.md` next steps).
