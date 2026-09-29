---
title: Training Controller
tile_id: training_controller
status: complete
---

# Training Controller

## Purpose

Training Controller is the dashboard for model-training jobs. You submit
a job with a framework, entry point, dataset, and resource ceiling; the
tile queues it behind a compatibility check, lists every job with its
status and elapsed time, plots live metrics for the selected job, and
gives you Cancel, Pause, and Resume. It manages jobs; the training
itself runs in the framework adapter.

## Inputs

Submitting a job builds a `TrainingConfig`
([`config.py`](../../src/shared/python/training/config.py)):

| Field | Type / unit |
| --- | --- |
| `framework` | `TrainingFramework.PYTORCH` (`"pytorch"`) or `TrainingFramework.GYMNASIUM` (`"gymnasium"`) |
| `entry_point` | non-empty string, `"module.path:callable"` or a script path |
| `output_dir` | `pathlib.Path` for checkpoints, metrics, and logs; created by the worker |
| `hyperparameters` | free-form mapping, passed verbatim to the adapter |
| `dataset_id` | id of a registered dataset, or `None` for self-generating workloads such as RL envs |
| `resources` | `ResourceRequest`; defaults to 1 CPU core and 1024 MiB RAM |
| `max_epochs` | positive int or `None` (supervised) |
| `max_steps` | positive int or `None` (RL / per-iteration) |
| `seed` | non-negative int or `None` |
| `tags` | string-to-string mapping for dashboard filtering |
| `schema_version` | int >= 1 |

`ResourceRequest` fields: `cpu_cores` (count, >= 1), `gpu_count`
(count, >= 0 where 0 means CPU-only), `memory_mb` (MiB resident-set
ceiling), and `gpu_memory_mb` (MiB per GPU, or `None` for no explicit
limit; must be `None` when `gpu_count` is 0).

Live input arrives as `TrainingMetric` events over
`training/<job_id>/progress`, plus `(TrainingStatus, message)` status
events. Datasets come from a `DatasetRegistry`.

## Outputs

The read-model in
[`view_model.py`](../../src/tools/training_controller/view_model.py):

- `JobRow` per job: `job_id`, `framework`, `status`, `dataset_id`,
  `elapsed_s` (seconds, non-negative; `now - started_at` while running,
  `completed_at - started_at` when terminal, `0.0` before start), and
  `error_message`.
- `MetricSeries` per metric: `name`, a `MetricKind` of `loss`, `reward`,
  `accuracy`, `scalar`, `learning_rate`, or `grad_norm`, plus equal-length
  `steps` (non-negative ints, the plot x-axis) and `values` (floats, the
  y-axis), with an optional `smoothed` rolling-mean overlay for noisy RL
  rewards. `MetricKind.lower_is_better` is `True` for `loss` and
  `grad_norm`, so the dashboard can pick a best record without
  per-metric configuration.
- `ResourceSnapshot`: `cpu_percent` and `memory_percent` (both percent
  in `[0, 100]`, or `None`), a tuple of `GpuSnapshot`, and an
  `available` flag that is `False` with all-`None` numerics when
  `psutil` is missing.
- `GpuSnapshot` per device: `index`, `name`, `utilization_percent`
  (percent in `[0, 100]` or `None`), `memory_used_mb` and
  `memory_total_mb` (MiB, used <= total).
- On disk: whatever the worker writes into `output_dir`.

## Method

The tile is a Model-View-Controller split. The headless controller is
`TrainingDashboardController`
([`controller.py`](../../src/tools/training_controller/controller.py)),
constructed with a `Scheduler`, a `DatasetRegistry`, and a
`CompatibilityChecker`. `submit_job` runs the compatibility check
**before** queueing, so the gate is enforced no matter which widget calls
it; `cancel_job`, `pause_job`, and `resume_job` act on the scheduler.
`ingest_metric` folds incoming metrics into the read-model, and
`on_model_change(callback)` registers a no-arg observer that fires on any
read-model change - scheduler status update, new metric for the selected
job, or selection change. All view-model dataclasses are frozen with
slots and validate their preconditions in `__post_init__`.

`live_subscriber.py` provides `TrainingJobLiveSubscriber(job_id,
on_metric=..., on_status=...)`, which subscribes to
`training/<job_id>/progress` via `src.shared.python.realtime` and decodes
each payload into a typed event. `start()` is idempotent; `stop()` is
idempotent and safe from any thread.

The PyQt6 surface is `gui.py`, providing `MainWindow`, `MainWidget`, and
`SubmitDialog` over that controller, with a hard-coded dark QSS theme.
`__main__.py` is the standalone entry point: it imports `gui`, and on
`ImportError` or `OSError` writes "Training Controller GUI unavailable"
to stderr and returns exit code 1. `_embed_adapter.py` exposes the same
widget to the launcher.

## Limitations

- Two frameworks only: PyTorch and Gymnasium. The enum docstring names
  TensorFlow, JAX, stable-baselines3, and RLlib as future work requiring
  a new enum member plus a `TrainingJobRunner` adapter.
- **It does not train anything itself.** It schedules, gates, monitors,
  and plots. Whatever `entry_point` names does the work.
- `entry_point` is only checked for being a non-empty string here;
  resolving it is the worker's job, so a typo surfaces at run time.
- `output_dir` is not verified to exist or be writable at construction.
- Resource fields are a declared ceiling passed to the scheduler, not an
  enforced sandbox.
- Metric plots need matplotlib. `gui.py` imports it behind a
  `HAS_MATPLOTLIB` guard and degrades when absent.
- Host resource readings need `psutil`. Without it `ResourceSnapshot`
  reports `available=False` and the strip shows nothing.
- The dark theme is hard-coded QSS in `gui.py`, not the shared theme
  system, so it will not follow a light appearance setting.
- Maturity is **beta**.
- `src/tools/training_controller/README.md` is stale: it states the
  package "currently ships only the headless portion" and that `gui.py`,
  `__main__.py`, and `_embed_adapter.py` are "deferred to a follow-up
  PR". All three now exist. Treat this page as current.

## See Also
- [Training Controller package README](../../src/tools/training_controller/README.md)
  (stale in the way noted above)
- [Dataset Generator](dataset_generator.md) - produces the datasets jobs consume
- [Engine selection](engine_selection.md)
