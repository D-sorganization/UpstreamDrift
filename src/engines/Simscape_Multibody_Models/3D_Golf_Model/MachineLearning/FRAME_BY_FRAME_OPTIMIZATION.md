# Frame-By-Frame Torque Search

This workflow is an overnight fallback for cases where the neural-network torque
optimization path does not produce a usable replay. It searches the actual
Simscape model one frame at a time instead of trusting a learned surrogate.

## Concept

For each target frame:

1. Start from the current simulated state.
2. Build a small deterministic set of constant-torque candidates.
3. Simulate each candidate over a short horizon.
4. Score the resulting club/body state against the next desired frame.
5. Commit the best candidate, advance the state, and repeat.
6. Smooth the piecewise-constant torque sequence.
7. Export polynomial coefficients for `run_ml_polynomial_input_swing.m`.

The MATLAB runner supports `parfor` candidate evaluation when Parallel
Computing Toolbox is available. It falls back to serial evaluation when the
toolbox or worker pool is unavailable.

## Prepare A Manifest

```powershell
py -3.12 src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\prepare_frame_by_frame_search.py `
  --desired-target-csv src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\desired_club_target.csv `
  --output-json src\engines\Simscape_Multibody_Models\3D_Golf_Model\MachineLearning\data\processed\frame_by_frame_search.json `
  --candidate-step 5.0 `
  --candidate-levels "-1,0,1" `
  --candidate-strategy coordinate `
  --horizon-frames 1 `
  --use-parallel auto
```

The manifest records target columns, control columns, candidate count, output
paths, tracking weights, smoothing parameters, and the required MATLAB hook
names. The helper validates that target time is strictly increasing and that
requested control columns exist in the column manifest.

## Run MATLAB

```matlab
summary = run_frame_by_frame_torque_search( ...
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/MachineLearning/data/processed/frame_by_frame_search.json");
```

The skeleton intentionally does not claim a complete Simscape stepping
implementation. The runner calls these extension points:

- `evaluateFrameByFrameTorqueCandidate(modelName, state, candidate, target, config)`
- `extractFrameByFrameState(simOut, previousState, config)`

Those hooks must be implemented against the concrete `GolfSwing3D_Kinetic`
state save/restore mechanics before production runs.

## Outputs

The runner writes:

- `frame_by_frame_torque_sequence.csv`: committed piecewise-constant controls.
- `frame_by_frame_torque_sequence_smoothed.csv`: moving-average smoothed controls.
- `frame_by_frame_torque_polynomials.mat`: polynomial coefficient variables for
  the existing polynomial-input replay.
- `frame_by_frame_torque_polynomials.summary.json`: coefficient and RMSE summary.

The polynomial MAT output is compatible with:

```matlab
run_ml_polynomial_input_swing(".../frame_by_frame_torque_polynomials.mat")
```

## Candidate Strategy

The default `coordinate` strategy evaluates hold-current plus one-axis
perturbations. That keeps the per-frame search bounded:

```text
1 + control_count * non_zero_candidate_levels
```

Use `cartesian` only for a deliberately small control subset because it expands
as:

```text
candidate_level_count ^ control_count
```

## Current Blocker

The workflow contract and deterministic planning are in place, but real
stateful Simscape stepping still requires model-specific hook implementations.
Until those hooks exist, the MATLAB runner raises a clear error at the candidate
evaluation extension point.
