# Neural Learning Tasks and Benefit Experiment (NM-01)

Governing issue: [#10616](https://github.com/D-sorganization/UpstreamDrift/issues/10616).
Parent epic: [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603).
Schema: `neural-benefit-experiment/1.0.0`.
Prerequisite: [#10615](https://github.com/D-sorganization/UpstreamDrift/issues/10615) (NM-00) merged.

## Purpose

Freeze reviewed learning-task contracts, the TB-00 model roster binding for
checkpoint dimensions, and the benefit-experiment registration before any
production generation or training.

## Learning Tasks

| Task                          | Inputs (conditioning)                                                          | Output                                         | Nonuniqueness Policy                                                     |
| ----------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------------ |
| Forward dynamics              | q, v, u, geometry, contact, dt (optional)                                      | acceleration or next state                     | N/A                                                                      |
| Inverse dynamics              | q, v, a, geometry, contact/actuation                                           | feasible controls/reactions                    | Selection objective **or** multimodal target; **no** physical uniqueness |
| Masked trajectory-to-controls | masked observation history, timestamps/horizon, q0/v0, geometry, model/profile | candidate trajectories or control coefficients | Proposal distribution + native polish                                    |

Shared conditioning always declares geometry, q0/v0 dimensions, horizon and
step count, contact regime, constraint set id, and a non-empty observation
mask. Missing mask, time or geometry fails closed.

## Model Roster and Dimensions

Model IDs and `n_q` / `n_v` come from TB-00
([#10585](https://github.com/D-sorganization/UpstreamDrift/issues/10585))
`GolfModelIdentity` entries via `dimensions_for_model`. Control dimension
`n_u` is supplied per layout and is never assumed to be a fixed 27×7 shape.

## Benefit Experiment Freeze

- Baselines: cold classical; retrieval+solve; existing networks+solve;
  forward-surrogate+polish; proposed inverse+polish.
- Latency phases: startup, preprocessing, candidate generation, neural
  inference, physical refinement, rejected attempts, independent replay.
- Pilot: nested 100 / 500 / 2,000 episodes; three seeds; ≥30 synthetic
  queries per stratum; four workbook trials; two C3D tests; statistical
  limitations recorded on the tiny empirical set.
- Promotion gates (proposed, not measured): ≥2× median speedup, non-worse
  p95, non-worse accepted-quality rate at frozen TB-02/#10587-aligned gates.
- Break-even: offline cost ÷ per-query savings; no positive break-even when
  savings ≤ 0. Negative benefit retains research checkpoints but blocks
  accelerated-product promotion.

Compute/storage/wall-time caps remain pending a short timing probe before
training dispatch (`compute_caps_pending_timing_probe=true`).

## How to Reproduce

```powershell
python -c "from src.shared.python.neural_motion import default_benefit_experiment; s=default_benefit_experiment(); print(s.schema, s.split_digest[:12], len(s.baselines))"
python -m pytest tests/unit/neural_motion/test_learning_tasks.py tests/unit/neural_motion/test_benefit_experiment.py -q -n 0 --no-cov --timeout=60
```

Evidence receipt:
`docs/plans/neural_motion_matching/evidence/nm01_benefit_experiment.json`.

## Next Action

Dispatch [#10617](https://github.com/D-sorganization/UpstreamDrift/issues/10617)
(NM-02): make native dataset labels complete and semantically correct under
these frozen contracts.
