# Regression Drive Sensitivity Findings

Epic [#10950](https://github.com/D-sorganization/UpstreamDrift/issues/10950),
found while proving `GS3DX_Slim` (#10954). MATLAB R2025b, `ode23t`,
RelTol 1e-3, AbsTol 1e-5, MaxStep 1e-3, 0.3 s window.

## Summary

The inputs saved in `GolfSwing3D_Kinetic.slx` (identical to
`src/model/PolynomialInputValues.mat`) drive an ill-conditioned run.
Rounding-level changes grow into metre-scale differences. That run cannot
tell a structural change apart from rounding noise, so it is not usable as an
equivalence reference for any model restructuring.

`src/model/inputs/3DModelInputs_Impact.mat` is well conditioned and is now
the regression drive (`gs3dx_drive(info, "impact", mdl)`).

## Evidence

| Drive     | Steps | Clubhead speed at 0.1 / 0.2 / 0.3 s | Effect of RelTol x (1 + 1e-9) on clubhead |
| --------- | ----- | ----------------------------------- | ----------------------------------------- |
| persisted | 4,236 | 24 / 94 / 4,260 m/s                 | 2.43 m max; > 1 um from 0.059 s           |
| impact    | 344   | 18 / 9.3 / 11 m/s                   | 7.2e-11 m max                             |

- In the persisted drive, the left-shoulder angular acceleration reaches
  8e4 rad/s² at 0.05 s and 4e9 rad/s² at 0.3 s. The swing blows up, so the
  last part of the window is not physical.
- A RelTol change of one part in a million gives 2.58 m and 3,781 steps.
  Rounding differences grow about 1e6-fold within 10 ms.
- Zeroing the 225 polynomial coefficients in `PolynomialInputValues.mat`
  through the model workspace does not change the persisted run at all. Those
  variables are not what drives this run. Not investigated further, because
  it is outside this epic.

## Consequence For `GS3DX_Slim`

Direct `InputTorque` drive reorders the Simscape equations. It is exact in
exact arithmetic but not bit-identical in floating point.

- **Persisted drive:** Gimbal and Revolute rewires happen to stay
  bit-identical (0 failing signals). The Universal rewire perturbs rounding
  at 1e-9 by t = 8 ms, and that grows to a 2.57 m clubhead difference, the
  same size as the original's own 1e-9 RelTol perturbation (2.43 m).
- **Impact drive:** `GS3DX_Slim` matches the original on all 413 signals,
  with the same 344 steps and a 6e-14 m maximum clubhead difference.

The unchanged `GS3DX_Baseline` clone still matches the original on both
drives (`test_gs3dx_harness/clone_reproduces_original`). The comparison
tolerances (atol 1e-6, rtol 1e-3) were not changed.

## Follow-Up for the Model Owner

The persisted input set produces a non-physical swing (clubhead above
4 km/s). Anyone using the default model inputs for analysis or motion
matching should check that this is the intended input set.
