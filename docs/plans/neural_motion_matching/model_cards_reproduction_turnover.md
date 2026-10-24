# Neural Motion Matching: Model Cards, Reproduction Commands and Turnover (NM-12 #10627)

Governing Issue: [#10627](https://github.com/D-sorganization/UpstreamDrift/issues/10627)  
Parent Epic: [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603) (Neural Motion Matching)  
Receipt: [nm12_model_cards_turnover_receipt.json](evidence/nm12_model_cards_turnover_receipt.json)

> **DIAGNOSTIC ONLY.** The receipt has `status: DIAGNOSTIC`, an empty
> `promoted_models` list, and `end_to_end_verification.status: not_verified`.
> Earlier promotion, RMSE, speedup, and completion claims were retracted by the
> fabricated-evidence audit (#10960). No model promotion or end-to-end
> qualification is established by this record.

## Receipt-Supported Model Dispositions

| Model ID | Receipt disposition | Evidence status |
| --- | --- | --- |
| `driven_double_pendulum` | Unmeasured | Diagnostic; no promotion |
| `driven_triple_pendulum` | Unmeasured | Diagnostic; no promotion |
| `constrained_upper_body_golfer` | Unmeasured | Diagnostic; no promotion |
| `reconstruction_double_pendulum` | Research only | Diagnostic receipt classification |
| `reconstruction_triple_pendulum` | Research only | Diagnostic receipt classification |
| `reconstruction_golfer` | Research only | Diagnostic receipt classification |
| Full-body engine models | Blocked prerequisite | Diagnostic receipt classification |
| Reference catalog models | Reference only | Diagnostic receipt classification |

The receipt lists 20 catalog entries, with no promoted models. It does not
provide measured speedups, native replay RMSE, or verified reproduction
results. Numeric model performance claims previously published here are
withdrawn and must not be used as results.

## Reproduction Commands

The commands below describe proposed software entry points only. Their presence
does not establish successful execution, a trained checkpoint, or physical
forward replay:

```bash
python -m src.shared.python.neural_motion.episodes.storage --model <model_id> --episodes 100 --seed 42
python -m src.shared.python.training.cli --runner neural_motion --entry-point neural_motion.train --model-id <model_id> --epochs 10
python -m src.shared.python.neural_motion.benchmark.runner --model <model_id> --seeds 3
python -m src.shared.python.motion_matching.hybrid --neural-model <model_id> --fallback-classical
python -m src.shared.python.neural_motion.matrix.replay --model <model_id> --verify-native
```

## Turnover Status

The receipt records end-to-end verification as `not_verified` and
`all_issues_completed` as `false`. No roadmap completion or physical ODE replay
claim is made here. Reproduction and qualification require new, receipt-backed
evidence from real trained checkpoints and independent forward replay.
