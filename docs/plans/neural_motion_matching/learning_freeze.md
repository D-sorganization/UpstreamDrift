# Neural Motion Matching Learning Freeze (NM-01)

Governing issue: [#10616](https://github.com/D-sorganization/UpstreamDrift/issues/10616).
Parent epic: [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603).
Prerequisite: [#10615](https://github.com/D-sorganization/UpstreamDrift/issues/10615) (NM-00).

Schemas:

- `neural-learning-tasks/1.0.0`
- `neural-model-roster/1.0.0`
- `neural-benefit-experiment/1.0.0`

## Purpose

Freeze learning-task contracts, the TB-00-keyed model roster and the benefit
experiment (baselines, splits, latency phases, promotion gates, compute caps,
break-even formula) before any production generation or training.

## Contracts

| Surface                          | Module                     | Notes                                                                                                           |
| -------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Forward / inverse / masked tasks | `neural_motion.tasks`      | Dimensions from `GolfModelIdentity`; inverse labels require selection or multimodal policy; no uniqueness claim |
| Model roster                     | `neural_motion.roster`     | One entry per registered model; full-body deferred pending benefit review                                       |
| Benefit experiment               | `neural_motion.experiment` | Nested 100/500/2000 episodes, three seeds, ≥30 synthetic queries/stratum, four workbook + two C3D targets       |

Promotion gates (proposed, not measured): ≥2× median speedup, non-worse p95,
non-degraded accepted-quality rate. Negative benefit retains research
checkpoints and blocks accelerated-product promotion. Break-even is undefined
when per-query savings ≤ 0.

## How to Reproduce

```powershell
python -c "from src.shared.python.neural_motion import build_neural_model_roster, freeze_benefit_experiment; r=build_neural_model_roster(); print(len(r.entries), r.content_digest()); print(freeze_benefit_experiment().split_digest)"
```

```powershell
python -m pytest tests/unit/neural_motion/test_learning_tasks.py tests/unit/neural_motion/test_model_roster.py tests/unit/neural_motion/test_benefit_experiment.py tests/unit/neural_motion/test_nm01_dbc_optimize.py -q -n 0 --no-cov --timeout=60
```

## Evidence

- `docs/plans/neural_motion_matching/evidence/nm01_learning_tasks_pilot.json`
- `docs/plans/neural_motion_matching/evidence/nm01_model_roster.json`
- `docs/plans/neural_motion_matching/evidence/nm01_benefit_experiment_receipt.json`

## Limitations

No training, native generation or speed measurement was performed. Synthetic
fixtures remain software-contract-only. Workbook/C3D slots are external targets,
not a population sample.

## Next Action

NM-02 ([#10617](https://github.com/D-sorganization/UpstreamDrift/issues/10617))
landed native label completeness under these frozen contracts. Do not start
[#10618](https://github.com/D-sorganization/UpstreamDrift/issues/10618) (NM-03)
until explicitly dispatched.
