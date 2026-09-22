# NM-04 Teacher Episodes and Active-Learning Candidates

Governing issue: [#10619](https://github.com/D-sorganization/UpstreamDrift/issues/10619)
(epic [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603)).

Prerequisites: NM-02 (#10617), NM-03 (#10618).

Schema: `neural-teacher-campaign/1.0.0`

## What Landed

Teacher generation under `src/shared/python/neural_motion/teacher/`:

- Structured perturbations from reviewed CO-03-style anchors (no workbook ancestry
  on generated episodes — teacher leakage prevention)
- Feasible episodes written to NM-03 `EpisodeStore`; infeasible rollouts in a
  separate rejected ledger with simulation-cost accounting
- Teacher ledger records objective, convergence iterations and independent replay
  digest (software-contract mock backend in unit tests; native adapters deferred)
- Resumable generation state keyed by attempt identity (duplicate-seed skip)
- Active-learning candidate selection (`uncertainty`, `disagreement`, random
  control) that never consumes `test`, `real_data_eval`, or `eval_held_out` labels
- Pilot spec for `driven_double_pendulum` aligned with NM-01 nested 100/500/2000
  stages

## Reproduce

```powershell
python -m pytest tests/unit/neural_motion/test_teacher_episodes_nm04.py -q -n 0 --no-cov --timeout=60
```

## Evidence

[`evidence/nm04_teacher_pilot_receipt.json`](evidence/nm04_teacher_pilot_receipt.json)

## Limitations

Mock backend and software-contract fixtures only. No native teacher generation,
training, speed, or acceleration claims. Expansion beyond NM-01 compute caps
requires a documented evidence-based decision.
