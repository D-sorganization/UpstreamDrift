# Review Comments Archive - 2026-05-06

Generated: 2026-05-06T19:10:59.150452

## Reviewer (chatgpt-codex-connector[bot]) (3 comments)

### PR #4228: src/shared/python/motion_matching/inverse/train_option3_cvae.py:251

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P0 Badge](https://img.shields.io/badge/P0-red?style=flat)</sub></sub>  Call dataset_coverage_map with its supported parameters**

`dataset_coverage_map` in `inverse/diagnostics.py` accepts `trials` and `sim_fn`, but this call passes `kinematics=` and `forward_fn=`. That raises a `TypeError` at runtime before any metrics are produced, so `train_option3_inverse_cvae` fails every time after training completes instead of returning a result bundle.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4228#discussion_r3198560364)

---

### PR #4228: src/shared/python/motion_matching/inverse/train_option3_cvae.py:274

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Pass exactly one trial to sample_diversity**

`sample_diversity` requires conditioning input that reduces to shape `(1, T, F)`, but this code passes up to 5 test trials at once (`test_idx[:min(5, ...)]`). For typical test splits this triggers a `ValueError` (not `RuntimeError`), so the current `except RuntimeError` block will not handle it and evaluation will abort once the coverage call is fixed.

Useful? Re...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4228#discussion_r3198560367)

---

### PR #4228: tests/unit/motion_matching/test_train_option3_cvae.py:118

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Use the dataset folder path in Option-3 training tests**

These tests build a `SweepDataset` object and then pass `dataset_path=str(dataset)` into `Option3TrainConfig`; `train_option3_inverse_cvae` expects a filesystem folder containing parquet files, so this stringified object path is invalid and causes `load_sweep_dataset` to fail. As written, the new end-to-end training tests cannot exercise the intended w...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4228#discussion_r3198560369)

---

