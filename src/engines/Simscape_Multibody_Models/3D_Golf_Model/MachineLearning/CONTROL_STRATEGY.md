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

## Matching Objective And Diagnostics

The redundant-control problem should be optimized as a weighted trajectory
objective, not as a single inverse model lookup. A practical objective is:

```text
minimize J =
  || club(q, qdot, qddot) - club_target ||_W^2
  + lambda_work * integral(sum(max(tau_i * qdot_i, 0)), dt)
  + lambda_tau * integral(||tau||_2^2, dt)
  + lambda_smooth * integral(||d tau / dt||_2^2, dt)
  + lambda_motion * integral(||q - q_reference||_2^2, dt)
  + lambda_limits * joint_velocity_torque_limit_penalties
```

The first term is the club-tracking target. The other terms choose among the
many torque and body-motion solutions that can produce similar club motion.
Positive mechanical work is the physically meaningful energy term, because net
work can cancel across accelerating and braking phases. When paired joint
velocities are not exported with the torque sequence, use squared torque,
absolute torque impulse, peak control magnitude, and torque-rate smoothness as
lower-fidelity effort proxies.

`evaluate_matching_workflow.py` is the current non-blocking feedback harness. It
does not decide whether a run is acceptable. It writes repeatable metrics and
plots so each optimization attempt can be compared against previous runs:

```powershell
py -3.12 src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\evaluate_matching_workflow.py `
  --target-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\TW_ProV1_downswing_club_target_calibrated.csv `
  --sim-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\simulated_club_motion.csv `
  --torque-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\optimized_club_torques.csv `
  --joint-velocity-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\simulated_body_state.csv `
  --scenario downswing `
  --run-label downswing_trial_001
```

Use the resulting JSON/Markdown report to compare:

- whole-trajectory normalized RMSE for club position, velocity, and acceleration
- impact-window RMSE near ball contact
- positive mechanical work when paired qdot is available
- L2 torque effort, L1 torque impulse, peak absolute torque, and torque smoothness
- weighted objective changes as `lambda_work`, `lambda_tau`, and
  `lambda_smooth` are swept

This gives a practical way to minimize work required to match the club motion:
run a Pareto sweep over tracking, effort, and smoothness weights, replay the
best candidates in MATLAB, then select the lowest-effort candidate whose
impact-window club error is still acceptable.

The Pareto sweep runner automates the first filtering pass:

```powershell
py -3.12 src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\run_matching_pareto_sweep.py `
  --checkpoint src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\runs\club_direct_10_cpu\best_model.pt `
  --desired-club-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\TW_ProV1_downswing_club_target_calibrated.csv `
  --reference-body-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\reference_body_state.csv `
  --output-dir src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\pareto_sweep `
  --effort-weights 1e-8,1e-7,1e-6 `
  --smoothness-weights 1e-10,1e-9,1e-8 `
  --steps 500 `
  --scenario downswing
```

Replay the sweep's `best_low_error`, `best_low_effort`, and `knee_point`
candidates first. This is a practical way to manage redundant torques: search
the regularization surface, then spend MATLAB time only on candidates that are
clearly distinct in tracking/effort tradeoff.

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

## Starting State Strategy

The first smoke workflow used row 0 from `club_direct_dynamics.parquet` as the
reference body state for torque optimization and otherwise relied on the model
input file already loaded by the MATLAB model. That was enough to prove the
polynomial input bridge, but it did not explicitly choose an address position.

The start-state workflow is now explicit and separate from the polynomial torque
inputs:

```text
start-state MAT file -> StartPosition/StartVelocity variables
polynomial MAT file  -> sixth-order torque coefficient variables
```

Two scenarios are supported:

- `full-swing`: exports start positions and velocities from `3DModelInputs.mat`.
  Use this for an address-position workflow. This should be reviewed visually
  against the measured motion-capture address before trusting the target match.
- `downswing`: exports start positions and velocities from
  `3DModelInputs_TopofBackswing.mat`. Use this when the current model start is
  already a good end-of-backswing pose and the target is sliced to the downswing.

Export a start state:

```matlab
cd('C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning')
addpath(fullfile(pwd, 'matlab'))
export_start_state_from_input_file( ...
    "downswing", ...
    fullfile(pwd, 'data', 'processed', 'ml_downswing_start_state.mat'));
```

Run the model with both the start-state MAT and polynomial MAT:

```matlab
simOut = run_ml_polynomial_input_swing( ...
    fullfile(pwd, 'data', 'processed', 'ml_torque_polynomial_inputs.mat'), ...
    'GolfSwing3D_Kinetic', ...
    fullfile(pwd, 'data', 'processed', 'ml_downswing_start_state.mat'));
```

This is deliberately not blended into the polynomial coefficient file. Starting
pose and actuator forcing are different control surfaces in the Simscape model
and should remain separately inspectable.

## Coordinate Calibration

The measured workbook club coordinates and the Simscape club-head logs can have
different origin, scale, and axis orientation. The calibration script fits either
a similarity transform or a full affine transform from measured clubface
positions to Simscape `ClubLogs_CHGlobalPosition_*` positions:

```powershell
py -3.12 src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\calibrate_club_target_to_sim.py `
  --target-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\TW_ProV1_downswing_club_target.csv `
  --sim-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\simulated_club_motion.csv `
  --output-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\TW_ProV1_downswing_club_target_calibrated.csv `
  --output-json src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\club_target_calibration.json
```

Default `similarity` mode applies:

```text
club_sim = scale * club_measured * rotation + translation
```

Velocities and accelerations receive the linear part of the transform without
the translation. The direct torque optimizer can now consume either workbook
column names (`clubface_x`, `clubface_vx`, `clubface_ax`) or calibrated
Simscape target column names (`ClubLogs_CHGlobalPosition_1`, etc.).

Run the calibration validator before using the calibrated target as the
optimization reference:

```powershell
py -3.12 src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\validate_club_calibration.py `
  --measured-target-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\TW_ProV1_downswing_club_target.csv `
  --calibrated-target-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\TW_ProV1_downswing_club_target_calibrated.csv `
  --sim-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\simulated_club_motion.csv `
  --transform-json src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\club_target_calibration.json `
  --output-dir src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\calibration_validation
```

The JSON/Markdown report and optional plots make frame mistakes visible before
optimization. Negative determinant, extreme scale, large impact-window
residuals, and axis-skewed residuals should be treated as model setup issues,
not tuning problems.

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

## Sequential Frame-By-Frame Fallback

The neural-network path is still the preferred fast iteration loop, but the
project now also has a deterministic fallback for overnight experiments:

```text
current Simscape state
    -> evaluate short-horizon constant-torque candidates
    -> commit the best candidate for the next target frame
    -> advance state and repeat
    -> smooth the piecewise torque profile
    -> export polynomial inputs
    -> replay and evaluate target-vs-simulated club motion
```

This is a sequential optimal-control approximation. It is compute intensive,
but it gives a concrete way to make progress when the surrogate optimizer is
not yet faithful enough. The default `coordinate` candidate strategy evaluates
the current torque plus one-axis perturbations, which scales as:

```text
1 + control_count * non_zero_candidate_levels
```

Use full Cartesian candidate sets only for very small control subsets. MATLAB
candidate evaluation can use `parfor` when Parallel Computing Toolbox is
available.

The implementation currently includes the manifest builder, GUI tab, MATLAB
candidate-loop skeleton, smoothing, and polynomial export. The remaining
model-specific work is to implement the Simscape stepping hooks that restore
state, apply a constant-torque candidate over the horizon, and extract the next
state/club target values from `GolfSwing3D_Kinetic`.

## Next Required Refinements

Before relying on optimized torques in production model studies, add these controls:

1. Visual address-pose validation for `3DModelInputs.mat` against the motion-capture address frame.
2. Bounds for actuator torque/force columns.
3. Joint-specific smoothness and effort penalties across adjacent samples.
4. A held-out swing split in addition to the current random row split.
5. Model-specific frame-by-frame Simscape stepping hooks for the sequential fallback.
6. Closed-loop replay tests in MATLAB/Simscape for both full-swing and downswing scenarios.
7. A sequence-level optimizer that solves the whole club trajectory instead of one timestep at a time.

## MATLAB GUI

`matlab/ml_workflow_gui.m` provides a step-by-step UI for the workflow:

```matlab
cd('C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning')
addpath(fullfile(pwd, 'matlab'))
ml_workflow_gui
```

The GUI is intentionally thin. It calls the same Python and MATLAB scripts
documented here so that every button has a reproducible command-line equivalent.
The tabs separate target/replay, surrogate sweeps, frame-by-frame fallback
development, and diagnostics.

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
