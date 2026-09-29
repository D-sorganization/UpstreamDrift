# NM-09 Checkpoint Matrix and Qualification

Governing issue: #10624 (epic #10603).
Schema: `neural-checkpoint-matrix/1.0.0`
Card Schema: `neural-model-checkpoint/1.0.0`

## What Landed

Primary package under `src/shared/python/neural_motion/matrix/`:

- `NeuralCheckpointMatrix` and `ModelCheckpointCard`:
  - Complete per-model matrix covering all 20 models across the #10585 roster.
  - Variable tensor dimensions ($n_q, n_v, n_u, n_c$) strictly bound to authoritative `GolfModelIdentity`.
  - Cryptographic hash chain (`dataset_hash` -> `split_hash` -> `weight_digest` -> `checkpoint_hash` -> `matrix_digest`).
  - Precondition assertion `assert_model_checkpoint_compatible` preventing accidental cross-model loading.
- Model qualification and adapters:
  - Planar driven pendulums (`driven_double_pendulum`, `driven_triple_pendulum`): qualified native with forward ODE replay and 3-seed evidence.
  - Constrained upper-body golfer (`constrained_upper_body_golfer`): 8 coordinates, 5 independent DOFs, with 4 holonomic bilateral loop closure constraints verified to $< 10^{-4}$ residual norm.
  - Reconstruction models (`reconstruction_golfer`, `reconstruction_double_pendulum`, `reconstruction_triple_pendulum`): kinematic proposal networks with kinematic joint angle control basis; torque fabrication is strictly forbidden.
  - Full-body engine models: fail closed with explicit named blockers where optional runtimes are uninstalled (MATLAB R2025b for Simscape, Drake SDK, OpenSim Moco, MyoSuite MS-50).
- Independent native replay:
  - `verify_checkpoint_native_replay`: validates time horizon, dynamic consistency, and holonomic loop constraints, failing closed on non-finite values or constraint violations.

## Evidence

- `docs/plans/neural_motion_matching/evidence/nm09_checkpoint_matrix_receipt.json`
- Matrix digest: `a73d0303e16aec98b68cd0bbd69fe2b0fe337474ea0b37ca3830e9a8b6e93b72`

## Limitations

- Native ODE replay verified on reduced planar mechanisms and constrained bilateral loop.
- Kinematic models evaluate kinematic joint trajectories only and do not produce physical joint torques.
- Full-body models with uninstalled runtimes remain blocked rather than claiming false native acceleration.

## Next Action

NM-10 (#10625): Benchmark accepted-match speed, data efficiency and break-even under epic #10603.
