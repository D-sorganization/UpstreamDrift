# NM-11: Integrate Model-Specific Training and Inference With Existing Tools

Governing issue: #10626 (epic #10603).
Schema: `neural-tools-integration/1.0.0`

## What Landed

### 1. Multi-Runner Framework Dispatch (`src/shared/python/training/runtime/runner_registry.py`)
- Registered `neural_motion` framework alongside existing classical runners (`classical_motion_matching`, `surrogate_training`, `direct_collocation`).
- `get_runner_for_framework("neural_motion")` dynamically dispatches to `NeuralMotionRunner`.
- `available_frameworks()` returns the full immutable list of supported training framework keys.

### 2. Neural Motion Training Runner (`src/shared/python/training/runtime/adapters/neural_motion.py`)
- `NeuralMotionRunner` implements the standard `TrainingJobRunner` protocol.
- Emits fine-grained `ProgressSink` events (`STATUS_CHANGED`, `METRIC_RECORDED`) for real-time dashboard telemetry.
- Produces verifiable `ModelCheckpointCard` outputs with SHA-256 digests and model contracts.
- Respects cooperative cancellation via `CancelToken`, ensuring graceful early exit without resource leaks or corrupted checkpoints.

### 3. Portable Job Package Packaging (`src/shared/python/training/portable.py`)
- `export_job_package`: bundles job configuration, manifest, and checkpoint artifacts into portable `.zip` archives with deterministic SHA-256 manifest verification.
- `import_job_package`: extracts and verifies portable packages, strictly guarding against zip-slip directory traversal vulnerabilities (`ValueError: Potential directory traversal`).
- Safely serializes mappingproxy hyperparameters via `dict()` wrapping.

### 4. Training Controller View Models (`src/tools/training_controller/view_model.py`, `controller.py`)
- View models `ModelTopologyItem` and `DatasetSchemaItem` representing model architectures and dataset expectations.
- `default_neural_motion_topologies()` provides standard neural topologies (`mlp_residual`, `transformer_sequence`, `diffusion_policy`).
- `TrainingDashboardController.available_model_topologies()` and `available_dataset_schemas()` allow dynamic GUI inspection of model requirements and dataset formats.

### 5. Motion Matching GUI Integration (`src/tools/motion_matching/gui.py`)
- Added `neural_group` controls:
  - Mode selector: `Classical Only`, `Neural Preview`, `Neural Verified`.
  - Topology / Checkpoint selector: dynamically populates available neural motion architectures.
  - Classical fallback toggle: `Allow classical fallback if neural check fails`.
- Added results inspection badges:
  - `neural_status_badge`: visual indicator of solver outcome (`Neural Accepted`, `Classical Fallback`, `Rejected`, `Classical Solution`).
  - `metric_neural_confidence`: empirical confidence readout $[0.0, 1.0]$.
  - `metric_time_breakdown`: visual latency split between neural proposal and verification/polish phases.
  - `update_neural_metrics(...)` method for updating readouts upon job completion.

## Evidence

- `docs/plans/neural_motion_matching/evidence/nm11_tools_integration_receipt.json`
- Tests:
  - `tests/unit/training/test_neural_motion_runner_nm11.py` (8 passing)
  - `tests/unit/training/test_portable_packaging_nm11.py` (4 passing)
  - `tests/unit/training/test_view_model_nm11.py` (4 passing)
  - `tests/tools/motion_matching/test_motion_matching_gui.py` (12 passing)

## Limitations

- GUI components operate offscreen in automated CI; interactive display relies on PyQt6 desktop environment.
- Neural proposal inference requires trained model weights; untrained / missing checkpoints fail closed or trigger fallback according to user configuration.

## Next Action

NM-12 (#10627): Publish model cards, reproduction commands, and final turnover under epic #10603.
