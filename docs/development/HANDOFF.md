# Implementation Handoff — Exploratory GS3DX Simscape Model (#10950)

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-simscape-gs3dx`
- Branch: `feat/simscape-gs3dx-exploratory`
- Baseline commit: `2d5830d18` (origin/main)
- Implementation commit: `SELF`
- Pull request: not created
- Governing issue/epic: #10950 (children #10951–#10959)
- Lease session: `claude-deskcomputer-20260926-gs3dx`
- Development log: `DL-#10950`

## Objective and Status

- Objective: build agent-editable `GS3DX_` clones of `GolfSwing3D_Kinetic.slx` in
  `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/`,
  move shoulders and hip to quaternion joints, and grow to a full-body model under
  the 1,000 non-virtual-block Home-license limit, without touching the originals.
- Status: GS3DX-1 (#10951) scaffold done.
- Completed:
  1. `gs3dx_setup` (session path, cache redirect, no `savepath`), `gs3dx_names`,
     `gs3dx_assert_no_shadowing`, `gs3dx_save_model` (write guard),
     `gs3dx_original_manifest`, `gs3dx_clone_baseline`.
  2. `models/GS3DX_Baseline.slx` + `GS3DX_KD_{Gimbal,Revolute,Universal}.slx` generated
     headlessly in R2025b; 12 `ReferencedSubsystem` blocks re-pointed.
  3. `tests/test_gs3dx_safety.m` 5/5 pass.

## Files and Decisions

- The originals are only `copyfile`d, never loaded or saved, so R2025b cannot re-save them
  (the original .slx was last saved in R2025a).
- The exploratory folder sits outside `matlab/src/`, so `setup_matlab_environment`'s
  `genpath(src)` never adds it to the path.
- Shadowing risk is real: during the probe, a same-named scratch copy won `which()`.

## Validation

- `matlab.exe -batch "cd('.../exploratory_gs3dx'); info=gs3dx_setup(); runtests('tests/test_gs3dx_safety.m')"`
  → 5 passed, 0 failed.
- The original model simulates headlessly (0.3 s of swing takes about 223 s cold); the model workspace is embedded (676 vars).

## Blockers and Risks

- MATLAB tests need a licensed R2025b; they are not part of the Python CI.

## Next Steps

1. #10952 regression harness: `gs3dx_simulate` + `gs3dx_compare`, baseline capture, clone equivalence.
2. #10953 block budget + license probe (agy).
3. #10954 slim, then #10955/#10956 quaternion joints, then #10957/#10958 lower body.
