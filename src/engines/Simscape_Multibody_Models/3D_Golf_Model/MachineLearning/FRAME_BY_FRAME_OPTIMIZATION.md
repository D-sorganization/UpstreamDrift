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

The runner calls three extension points, all implemented in the
`+frame_search` package alongside `run_frame_by_frame_torque_search.m`:

- `evaluateFrameByFrameTorqueCandidate` →
  `frame_search.evaluate_candidate_step`. Restores the previous frame's
  `xFinal` (or a starting-state MAT on the first frame), applies the
  candidate as a constant polynomial torque (constant term `<base>G`,
  higher terms zero), and runs the model from `currentState.time` to the
  next manifest target time with `SaveFinalState='on'`.
- `extractFrameByFrameState` → `frame_search.extract_state`. Pulls the
  Simulink final-state struct, the last-sample time, and (when manifest
  joint columns are present) the joint position/velocity vectors.
- `extractPredictedTarget` → `frame_search.extract_predicted`. Resolves
  each manifest target column (e.g. `ClubLogs_CHGlobalPosition_1`) via
  `CombinedSignalBus` or `logsout` and returns the value at the final
  sample. Missing columns raise an actionable MATLAB error.

Pure helpers in the package
(`parse_target_column`, `control_column_to_polynomial_base`,
`apply_constant_torque`, `frame_horizon`, `lookup_signal_value`) are
unit-tested in `matlab/tests/test_frame_by_frame_hooks.m`.

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

## Implementation Status

### Complete (Epic #3976)

**#3977 — Simscape Stepping Hooks**

- `+frame_search/evaluate_candidate_step.m`: Restores previous-frame `xFinal` or
  starting-state MAT, applies candidate as a flat polynomial torque (constant term
  G = torque, A..F = 0), runs the model from current time to target time.
- `+frame_search/extract_state.m`: Harvests final-state struct and (q, qd) vectors
  from the simulation output.
- `+frame_search/extract_predicted.m`: Resolves target columns (e.g.
  `ClubLogs_CHGlobalPosition_1`) via `CombinedSignalBus` or `logsout`.
- Pure helpers (`parse_target_column`, `control_column_to_polynomial_base`,
  `apply_constant_torque`, `frame_horizon`, `lookup_signal_value`) are unit-tested
  in `matlab/tests/test_frame_by_frame_hooks.m`.

**#3978 — Checkpoint/Resume + Progress Artifacts**

- `frame_search.checkpoint()`: Atomically writes the run state to
  `<run_dir>/checkpoint.mat` with manifest SHA-256 validation.
- `frame_search.resume()`: Reads the checkpoint, validates manifest hash, and detects
  stale locks (progress CSV not updated for >2x expected frame time).
- `frame_search_artifacts.py`: Python reader for progress CSV and run status.
  Exposes `ProgressRow`, `RunStatus`, and helpers for the GUI and analysis tools.
- `run_frame_by_frame_torque_search.m` increments progress.csv after each frame,
  snapshots checkpoint.mat every K frames, and resumes from the last committed frame
  when manifest hash matches.

**#3979-#3980 — Replay Diagnostics + Torque Smoothing**

- `torque_smoothing.py`: Moving-average, Savitzky-Golay, Butterworth lowpass, and
  spline smoothing methods. Polynomial residual diagnostic flags fits exceeding a
  configurable threshold.
- `frame_search_replay_diagnostics.py`: Drives polynomial replay (via
  `replay_matching_workflow.py`), computes trajectory residuals including impact-window
  analysis, torque effort, and emits a canonical Metrics record (or JSON fallback).
- `export_torque_polynomials.py`: Accepts smoothing configuration, optionally writes
  smoothed CSV, and flags polynomial fits with excessive residual.

### Test Coverage

- **Python**: 19 unit tests covering artifacts I/O, checkpoint manifest validation,
  stale-lock detection, replay subprocess mocking, smoothing methods, and metrics
  module discovery.
- **MATLAB**: 6 unit tests in `test_frame_search_checkpoint.m` for checkpoint/resume
  atomicity, manifest validation, and stale-lock warnings.
- All tests passing. Ready for overnight runs with trajectory slices.
