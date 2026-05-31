# Training Controller — dashboard tab

The training-controller dashboard tab lets users schedule, monitor,
and manage model-training jobs across the engines supported by
UpstreamDrift. It is the GUI counterpart to the backend that lives in
`src/shared/python/training/`.

## Status — complete (#6012)

The package ships both the **headless** triad (MVC controller, realtime
subscriber, read-model dataclasses) and the **PyQt6 widget surface**
(`gui.py`, `__main__.py`, `_embed_adapter.py`) plus the
`src/config/models.yaml` tile entry. The tile appears under the launcher
**Tools** category (`category: tool`, `default_launch: tab`).

Run standalone:

```bash
python -m src.tools.training_controller
```

## Widget surfaces

- **Job list** (left): sortable table of every `TrainingJob` — id,
  framework, status, dataset, elapsed, error — with a status-filter
  dropdown.
- **Job detail** (right, two tabs):
  - **Live metrics** — matplotlib plot grouped by `MetricKind` via
    `summarize_by_kind`; RL `REWARD` series get a `RollingMean`-smoothed
    overlay.
  - **Summary** — `best_per_metric` "best so far" card and a clickable
    output-directory artifact link (where `RunResult.artifacts` land).
- **Action toolbar**: Submit / Cancel / Pause / Resume / Open Output
  Dir. Submit opens a modal that builds a `TrainingConfig` (framework +
  target-engine dropdowns, dataset selector, JSON hyperparameter editor)
  and runs a `CompatibilityChecker` **preflight** — incompatible
  (config, engine) pairs surface in a non-dismissable banner and Submit
  stays disabled until the errors clear.
- **Resource strip** (bottom): latest `ResourceSample` (CPU%, mem%,
  per-GPU%); shows "monitoring unavailable" on
  `ResourceMonitorUnavailableError`.
- **Dataset library** (dock): `DatasetRegistry` contents with
  add-folder / remove / re-scan-folder controls.

## Backgrounding (stacked on Sub-PR A, #6013)

`_TrainingControllerEmbedAdapter` implements the optional
`BackgroundableTool` hooks the launcher host resolves structurally:

- `can_background()` → `True` (the scheduler runs independently of the
  GUI, so a hidden tab keeps training).
- `detach_to_window()` → `True`.
- `pause()` — no-op for the backend; the widget detaches its live
  realtime subscriptions so a hidden tab is cheap.
- `resume()` — re-renders from the still-bound controller and
  re-establishes a `TrainingJobLiveSubscriber` per non-terminal job.

The host surfaces a reopen affordance for backgrounded tabs via its
`backgrounded_tools()` API.

## Headless public surface

```python
from src.tools.training_controller import (
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

`controller.py`, `live_subscriber.py`, and `view_model.py` import with
no PyQt6 present — PyQt is imported lazily inside
`_embed_adapter.create_main_widget` and `gui.py`.

## Tests

Under `tests/tools/training_controller/`:

- `test_view_model.py` — Design-by-Contract checks for each dataclass.
- `test_controller.py` — controller bind/unbind against a real headless
  `Scheduler` (submit/cancel/pause/resume, compat gate, observer
  fan-out, read-model projection).
- `test_live_subscriber.py` — patches `src.shared.python.realtime` and
  asserts callback dispatch on synthesized payloads.
- `test_gui_smoke.py` — `pytest.importorskip`-style PyQt6 guard;
  instantiates the main window and exercises Submit / Cancel / status
  filter / preflight / dataset remove / pause-resume via `QTest`.
- `test_embed_adapter.py` — adapter capabilities + `BackgroundableTool`
  hooks + runtime-safe import.
