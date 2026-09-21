# Training Controller — dashboard tab

The training-controller dashboard tab lets users schedule, monitor,
and manage model-training jobs across the engines supported by
UpstreamDrift. It is the GUI counterpart to the backend that lives in
`src/shared/python/training/` (see PR #6008).

## Status

This package provides both the headless MVC core and the PyQt6 GUI dashboard
tab for model training orchestration:

- Headless architecture: `TrainingDashboardController`, `TrainingJobLiveSubscriber`,
  and typed read-model dataclasses (`DashboardModel`, `JobRow`, `MetricSeries`).
- PyQt6 widget surface: `MainWindow`, `MainWidget`, and `SubmitDialog` in `gui.py`,
  with `_embed_adapter.py` integrating the controller into the launcher tile
  ecosystem and `models.yaml`.

The GUI binds the read-model onto:

- a job-list table with status indicators and selection tracking,
- a per-job detail pane with live-metric visualization and resource telemetry,
- a Submit / Cancel / Pause / Resume action toolbar running compatibility gates,
- host-resource telemetry via `training.resource_monitor`,
- dataset selection and validation integration.

## Public Surface

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
change). The GUI layer subscribes at construction and re-renders from
`controller.current_model()`.

`TrainingJobLiveSubscriber(job_id, on_metric=..., on_status=...)`
subscribes to `training/<job_id>/progress` via
`src.shared.python.realtime` and decodes each payload into a typed
`TrainingMetric` / `(TrainingStatus, message)` event. `start()` is
idempotent; `stop()` is idempotent and safe from any thread.

## Tests

Unit and integration tests live under `tests/tools/training_controller/`:

- `test_view_model.py` — Design-by-Contract checks for every dataclass.
- `test_controller.py` — exercises the controller against a real `Scheduler`
  including submit/cancel/pause/resume, compatibility-check gates, observer
  fan-out, and read-model projection.
- `test_live_subscriber.py` — patches `src.shared.python.realtime` with a stub
  transport and verifies callback dispatch on synthesized metric/status payloads.
- `test_gui.py` — exercises the PyQt6 widget layer and embedded adapter.
