# 3D Golf Swing Surrogate Control Strategy

This document describes the machine-learning control plan for the MATLAB/Simscape 3D golf swing model.

The project now supports three related surrogate-learning paths:

1. Body dynamics surrogate.
2. Direct torque-to-club surrogate.
3. Two-stage club-to-body-to-torque surrogate workflow.

All three paths treat rows from the simulator parquet as independent instantaneous samples. Time is not a training input. Time is used only where needed to differentiate measured or simulated club velocity into club acceleration.

## Source Data

Simulation source:

```text
C:\Users\diete\Repositories\data\TenThousandFiles.parquet
```

Observed club target source:

```text
src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Motion Capture Plotter\Wiffle_ProV1_club_3D_data.xlsx
```

The observed workbook contains these sheets:

```text
Definitions
TW_wiffle
TW_ProV1
GW_wiffle
GW_ProV11
```

The first prepared target trajectory uses `TW_ProV1`.

## Model A: Body Dynamics Surrogate

Purpose:

```text
f_body(q, qdot, tau) -> q, qdot, qddot
```

Where:

- `q` is the model body/joint position vector.
- `qdot` is the model body/joint velocity vector.
- `tau` is the applied torque/force control vector.
- `qddot` is the resulting acceleration vector.

This model is useful when a full desired body motion is available or when another model has produced a body target motion from a desired club trajectory.

Extraction command:

```powershell
py -3.12 src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\extract_dynamics_dataset.py `
  --source C:\Users\diete\Repositories\data\TenThousandFiles.parquet `
  --manifest src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\column_manifest_inverse_ready.json `
  --output src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\golf_inverse_ready.parquet
```

Training command:

```powershell
py -3.12 src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\train_dynamics_surrogate.py `
  --dataset src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\golf_inverse_ready.parquet `
  --output-dir src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\runs\inverse_ready_10_cpu `
  --epochs 10 `
  --batch-size 8192
```

10-epoch CPU test result:

```text
rows: 310000
input_dim: 114
target_dim: 145
best scaled validation loss: 5.551402136916295e-05
mean position normalized RMSE: 0.007018344573831807
mean velocity normalized RMSE: 0.007607688433684719
mean acceleration normalized RMSE: 0.005450641655610228
```

Position and velocity are included in both the input and target vector for this first body model. That makes the position and velocity outputs mostly an identity check. The acceleration outputs are the more meaningful dynamics signal.

## Model B: Direct Torque-To-Club Surrogate

Purpose:

```text
f_club_direct(q, qdot, tau) -> club_position, club_velocity, club_acceleration
```

This is the one-model control path. Given a desired club trajectory, hold the current body state fixed and optimize the applied torque/force vector directly against the predicted club position, velocity, and acceleration.

Dataset extraction:

```powershell
py -3.12 src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\extract_club_datasets.py `
  --mode direct
```

Training command:

```powershell
py -3.12 src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\train_dynamics_surrogate.py `
  --dataset src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\club_direct_dynamics.parquet `
  --output-dir src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\runs\club_direct_10_cpu `
  --epochs 10 `
  --batch-size 8192
```

10-epoch CPU test result:

```text
rows: 310000
input_dim: 114
target_dim: 9
best scaled validation loss: 9.681552910478786e-06
mean unscaled RMSE across all club targets: 34.568023681640625
club acceleration RMSE: [31.5627, 95.0317, 26.9487]
club acceleration normalized RMSE: [0.001032, 0.003600, 0.002248]
```

Use this path when the only desired motion available is club motion and the goal is to solve directly for torques. It is simple and practical, but the optimization may discover torque solutions that match the club target while producing body motion that is undesirable unless regularization and actuator limits are added.

## Model C: Body-To-Club Kinematics Surrogate

Purpose:

```text
f_body_to_club(q, qdot, qddot) -> club_position, club_velocity, club_acceleration
```

This is the first model in the two-stage control path. It maps a full body kinematic state to the club state. It can be inverted by optimization:

```text
argmin_(q, qdot, qddot) || f_body_to_club(q, qdot, qddot) - desired_club ||^2
                    + motion_regularization
                    + acceleration_regularization
```

The resulting body kinematic target can then be passed to the body dynamics surrogate to solve for torques.

Dataset extraction:

```powershell
py -3.12 src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\extract_club_datasets.py `
  --mode body-to-club
```

Training command:

```powershell
py -3.12 src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\train_dynamics_surrogate.py `
  --dataset src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\body_to_club_kinematics.parquet `
  --output-dir src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\runs\body_to_club_10_cpu `
  --epochs 10 `
  --batch-size 8192
```

10-epoch CPU test result:

```text
rows: 310000
input_dim: 145
target_dim: 9
best scaled validation loss: 1.9082865037489682e-05
mean unscaled RMSE across all club targets: 30.222551345825195
club acceleration RMSE: [53.4838, 64.2922, 35.0145]
club acceleration normalized RMSE: [0.001749, 0.002436, 0.002921]
```

This path is more controllable than the direct torque-to-club path because it can impose body-motion efficiency terms before solving for torques.

## Preparing A Measured Club Target

The workbook target preparation command is:

```powershell
py -3.12 src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\prepare_club_target_trajectory.py `
  --sheet TW_ProV1 `
  --output src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\TW_ProV1_club_target.csv
```

Current `TW_ProV1` preparation result:

```text
rows: 775
time_min: -2.183333333333333
time_max: 1.0416666666666667
columns: sample, time, clubface_x, clubface_y, clubface_z, clubface_vx, clubface_vy, clubface_vz, clubface_ax, clubface_ay, clubface_az
```

The workbook coordinates use the documented global coordinate system in the `Definitions` sheet. The current exporter keeps those measured coordinates as-is. Alignment from measured clubface coordinates to Simscape `ClubLogs_CHGlobal*` coordinates should be verified before using the target in closed-loop optimization.

## One-Model Control Workflow

Use the direct torque-to-club model:

```text
desired club trajectory
    -> optimize tau in f_club_direct(q, qdot, tau)
    -> fit sixth-order torque polynomials
    -> write MATLAB coefficient MAT file
    -> torque timeseries
    -> MATLAB/Simscape replay
```

Advantages:

- Shortest path from measured club motion to torques.
- Does not require full-body motion capture.
- Uses the columns that are already available from the simulator parquet.

Risks:

- Many body/torque combinations can produce similar club states.
- Without torque limits and motion regularization, the optimizer can find non-human or numerically unstable solutions.
- It does not explicitly produce a body motion plan unless body state is also optimized.

## Two-Model Control Workflow

Use body-to-club, then body dynamics:

```text
desired club trajectory
    -> optimize q, qdot, qddot in f_body_to_club(q, qdot, qddot)
    -> desired body kinematic trajectory
    -> optimize tau in f_body(q, qdot, tau) to match desired q, qdot, qddot
    -> fit sixth-order torque polynomials
    -> write MATLAB coefficient MAT file
    -> torque timeseries
    -> MATLAB/Simscape replay
```

Advantages:

- Produces an interpretable body-motion target.
- Can minimize required body motion while matching the club.
- Can add joint-specific penalties, acceleration penalties, and feasibility constraints before solving for torques.

Risks:

- More moving pieces.
- Requires careful scaling between measured club coordinates and simulated club coordinates.
- Needs smoothing across the full trajectory; per-sample optimization can create discontinuous body targets or torques.

## Next Required Refinements

Before applying optimized torques back into the MATLAB model, add these controls:

1. Coordinate alignment between the measured workbook target and Simscape club-head coordinates.
2. Bounds for actuator torque/force columns.
3. Smoothness penalties across adjacent samples.
4. A held-out swing split in addition to the current random row split.
5. Closed-loop replay tests in MATLAB/Simscape.
6. A sequence-level optimizer that solves the whole club trajectory instead of one timestep at a time.

## MATLAB Polynomial Input Bridge

The kinetic Simscape model already has a polynomial input path. The relevant
MATLAB pieces are:

```text
matlab/src/functions/HexPolyInputFunction.m
matlab/src/functions/dataset_generator/getPolynomialParameterInfo.m
matlab/src/functions/dataset_generator/setPolynomialCoefficients.m
matlab/src/functions/dataset_generator/loadInputFile.m
matlab/src/model/PolynomialInputValues.mat
matlab/src/model/Inputs_GolfSwing3D_Kinetic.mat
```

`HexPolyInputFunction.m` evaluates:

```text
A*x^6 + B*x^5 + C*x^4 + D*x^3 + E*x^2 + F*x + G
```

`PolynomialInputValues.mat` contains coefficient variables such as:

```text
LScapInputXA ... LScapInputXG
LScapInputYA ... LScapInputYG
LSInputXA ... LSInputXG
LSInputYA ... LSInputYG
LSInputZA ... LSInputZG
RSInputXA ... RSInputXG
RSInputYA ... RSInputYG
RSInputZA ... RSInputZG
HipInputXA ... HipInputXG
HipInputYA ... HipInputYG
HipInputZA ... HipInputZG
TranslationInputXA ... TranslationInputXG
TranslationInputYA ... TranslationInputYG
TranslationInputZA ... TranslationInputZG
SpineInputXA ... SpineInputXG
SpineInputYA ... SpineInputYG
```

The ML bridge maps optimized torque columns to those coefficient bases:

```text
LScapLogs_ActuatorTorqueX -> LScapInputX
LScapLogs_ActuatorTorqueY -> LScapInputY
RScapLogs_ActuatorTorqueX -> RScapInputX
RScapLogs_ActuatorTorqueY -> RScapInputY
LSLogs_ActuatorTorqueX -> LSInputX
LSLogs_ActuatorTorqueY -> LSInputY
LSLogs_ActuatorTorqueZ -> LSInputZ
RSLogs_ActuatorTorqueX -> RSInputX
RSLogs_ActuatorTorqueY -> RSInputY
RSLogs_ActuatorTorqueZ -> RSInputZ
SpineLogs_ActuatorTorqueX -> SpineInputX
SpineLogs_ActuatorTorqueY -> SpineInputY
HipLogs_TranslationForceXInput -> TranslationInputX
HipLogs_TranslationForceYInput -> TranslationInputY
HipLogs_TranslationForceZInput -> TranslationInputZ
HipLogs_HipTorqueXInput -> HipInputX
HipLogs_HipTorqueYInput -> HipInputY
HipLogs_HipTorqueZInput -> HipInputZ
```

End-to-end one-model command sequence:

```powershell
py -3.12 src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\create_reference_body_state.py `
  --dataset src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\club_direct_dynamics.parquet `
  --output src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\reference_body_state.csv

py -3.12 src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\optimize_torque_sequence_for_club.py `
  --checkpoint src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\runs\club_direct_10_cpu\best_model.pt `
  --desired-club-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\TW_ProV1_club_target.csv `
  --reference-body-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\reference_body_state.csv `
  --output-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\optimized_club_torques.csv

py -3.12 src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\export_torque_polynomials.py `
  --torque-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\optimized_club_torques.csv `
  --output-mat src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\ml_torque_polynomial_inputs.mat
```

Then run in MATLAB:

```matlab
cd('C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning')
simOut = matlab.run_ml_polynomial_input_swing( ...
    fullfile(pwd, 'data', 'processed', 'ml_torque_polynomial_inputs.mat'), ...
    'GolfSwing3D_Kinetic');
```

This is now enough to generate model-importable polynomial coefficients. The
remaining scientific issue is not the file format; it is closing the loop by
running Simscape, comparing the resulting club trajectory to the target, and
iterating with coordinate alignment, torque bounds, and sequence smoothness.
