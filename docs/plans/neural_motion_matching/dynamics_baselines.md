# NM-05 Classical and Small Neural Dynamics Baselines

Governing issue: #10620 (epic #10603).

Schema: `neural-dynamics-baselines/1.0.0`

## What Landed

Pilot training under `src/shared/python/neural_motion/baselines/`:

- Trial-level matrices from NM-03 `EpisodeStore` / `FamilySplitPlan` (no row shuffle)
- Separate forward acceleration / next-state and inverse control tasks
- Classical baselines: analytical 1/2-DOF fixture map, ridge, nearest neighbor
- Optional small MLP with clean torch-absence skip
- Three seeds, validation checkpointing, test split untouched during selection
- Inverse labels require explicit `InverseLabelConditioning` (contact/actuation/allocation)
- Identity-channel leakage guards; unavailable torque fails closed

Reuses per-step normalization patterns from
`motion_matching/surrogate/perstep/train.py`. Pilot orchestration lives in
`neural_motion/baselines/train.py` and `neural_motion/baselines/dataset.py`.
Scheduler lookup remains `training/runtime/runner_registry.py`.

## Evidence

`evidence/nm05_dynamics_baselines_receipt.json`

## Limitations

Software-contract fixtures only. No native training success, acceleration claim,
or motion-matching certification from low MSE alone.
