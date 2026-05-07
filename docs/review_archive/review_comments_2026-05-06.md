# Review Comments Archive - 2026-05-06

Generated: 2026-05-06T19:08:11.283000

## Reviewer (chatgpt-codex-connector[bot]) (3 comments)

### PR #4227: src/shared/python/motion_matching/inverse/train_option3_cvae.py:251

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Call dataset_coverage_map with its real parameters**

This call uses `kinematics=` and `forward_fn=`, but `dataset_coverage_map` expects `trials` and `sim_fn` (see `inverse/diagnostics.py`). That mismatch raises a `TypeError` immediately, so `train_option3_inverse_cvae` fails during evaluation before it can produce metrics/artifacts on any dataset.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4227#discussion_r3198553514)

---

### PR #4227: src/shared/python/motion_matching/inverse/train_option3_cvae.py:269

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Pass a single trajectory into sample_diversity**

`sample_diversity` only accepts one conditioning trajectory (`(T,F)` or `(1,T,F)`), but this slice passes up to 5 test trajectories at once. For typical test splits (`len(test_idx) > 1`) this triggers a `ValueError`, and since the code only catches `RuntimeError`, evaluation aborts and training cannot complete.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4227#discussion_r3198553518)

---

### PR #4227: tests/unit/motion_matching/test_train_option3_cvae.py:113

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Use the dataset directory path in end-to-end test**

Here `dataset_path=str(dataset)` stringifies a `SweepDataset` object instead of passing the parquet folder path. `train_option3_inverse_cvae` then calls `load_sweep_dataset` on that non-path string, causing `FileNotFoundError` and preventing this end-to-end test from exercising the training pipeline.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4227#discussion_r3198553520)

---

