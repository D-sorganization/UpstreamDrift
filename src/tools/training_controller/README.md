# Training Controller — dashboard tab

The training-controller dashboard tab lets users schedule, monitor,
and manage model-training jobs across the engines supported by
UpstreamDrift. It is the GUI counterpart to the backend that lives in
`src/shared/python/training/` (see PR #6008).

## Status

This package currently ships only the **headless** portion of the
dashboard tab — the MVC controller, realtime subscriber, and read-model
dataclasses. The PyQt6 widget surface (`gui.py`, `__main__.py`,
`_embed_adapter.py`, and the `src/config/models.yaml` tile entry) is
deferred to a follow-up PR, because the environment that authored this
code has no display and cannot validate Qt widgets.

The follow-up PR will bind the read-model defined here onto:

- a job-list table,
- a per-job detail pane with live-metric plot + summary card,
- a Submit / Cancel / Pause / Resume action toolbar that runs the
  controller's compatibility-check gate before submit,
- a host-resource strip fed by `training.resource_monitor`,
- a dataset-library dock.

## Public surface (today)

```python
from training_controller import (
    TrainingDashboardController,   # MVC controller
    TrainingJobLiveSubscriber,     # realtime subscription wrapper
    DashboardModel,                # read-model
    JobRow,                        # one row of the job list
    MetricSeries,                  # plot-ready metric series
    ResourceSnapshot,              # host resource snapshot
    GpuSnapshot,                   # per-GPU snapshot
    job_row_from_training_job,     # TrainingJob -> JobRow projector
)
```

`TrainingDashboardController.on_model_change(callback)` registers a
no-arg callback that fires whenever the read-model changes (scheduler
status update, new metric ingested for the selected job, selection
change). The follow-up GUI layer subscribes once at construction and
re-renders from `controller.current_model()`.

`TrainingJobLiveSubscriber(job_id, on_metric=..., on_status=...)`
subscribes to `training/<job_id>/progress` via
`src.shared.python.realtime` and decodes each payload into a typed
`TrainingMetric` / `(TrainingStatus, message)` event. `start()` is
idempotent; `stop()` is idempotent and safe from any thread.

## Tests

Unit tests live under `tests/tools/training_controller/`:

- `test_view_model.py` — Design-by-Contract checks for every dataclass.
- `test_controller.py` — exercises the controller against a real
  `Scheduler` (the backend is headless) including submit/cancel/
  pause/resume, the compatibility-check gate, observer fan-out, and
  the read-model projection.
- `test_live_subscriber.py` — patches `src.shared.python.realtime`
  with a stub transport and verifies callback dispatch on synthesized
  metric / status payloads.

All three suites run in CI without PyQt6 installed — no
`pytest.importorskip` guards required.

## Branch naming

This PR was authored on `feat/training-controller-headless` per the
CLAUDE.md branch-naming rule. The GUI follow-up should use
`feat/training-controller-gui` or a `claude/training-controller-gui-*`
branch.
