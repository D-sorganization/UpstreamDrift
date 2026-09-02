# Golf Modeling Suite Output Directory

This directory contains all simulation results, analysis outputs, and generated reports from the Golf Modeling Suite.

## REST API Integration (Issue #8871)

`src.api.services.simulation_service.SimulationService` persists every
successfully completed `/simulate` run here via `OutputManager` — the same
manager used by the desktop app and the video-pose pipeline, so files
written from any of these paths share one schema. Each run's simulation
data (and analysis results, if requested) is saved as JSON under
`simulations/<engine>/`, and the resulting file path is returned to the
caller in `SimulationResponse.export_paths`. A failed run persists nothing,
so `export_paths` is empty in that case — it is never fabricated.

## Choosing the Output Location (Issue #9220)

`OutputManager` resolves its base directory in this order:

1. an explicit `base_path` argument;
2. the `UPSTREAM_DRIFT_OUTPUT_DIR` environment variable;
3. this directory — `<repository root>/output`.

The test suite sets `UPSTREAM_DRIFT_OUTPUT_DIR` to a temporary directory in
`tests/conftest.py`, so a test run never writes generated files into the
checkout. Previously the root-detection heuristic stopped at `src/` (because
`src/engines` exists) and dropped untracked
`src/output/simulations/<engine>/simulation_*.json` files into the source
tree on every run.

## Directory Structure

`OutputManager.create_output_structure()` creates this tree under
`base_path` (the individual save methods create only the specific
sub-directory they need, on demand):

```
output/
├── simulations/          # Raw simulation results (save_simulation_results)
│   ├── mujoco/
│   ├── drake/
│   ├── pinocchio/
│   └── matlab/
├── analysis/             # Pre-created subdirectories (not yet wired to a save method)
│   ├── biomechanics/
│   ├── trajectories/
│   ├── optimization/
│   └── comparisons/
├── exports/              # Pre-created subdirectories (not yet wired to a save method)
│   ├── videos/
│   ├── images/
│   ├── data/
│   └── c3d/
├── reports/               # export_analysis_report() writes into reports/<format_type>/
│   ├── json/
│   └── html/
└── cache/                 # Reserved for cleanup_old_files(); no subdirectories are created
```

Only `simulations/<engine>/` and `reports/<format_type>/` are actually
populated by `OutputManager` today. The `analysis/` and `exports/`
subdirectories are created by `create_output_structure()` for future use,
but nothing currently writes into them through this class.

## File Naming

`save_simulation_results(results, filename, ...)` sanitizes `filename` via
`sanitize_filename()`: if the name has no digits in it, a timestamp suffix
is appended automatically so repeated saves never collide. The final name
is `<sanitized-filename>.<format>` under `simulations/<engine>/`.

`export_analysis_report(analysis_data, report_name, format_type="json")`
always appends its own timestamp suffix: `<report_name>_<timestamp>.<format_type>`
under `reports/<format_type>/`.

## Data Formats

`OutputFormat` (`src/shared/python/data_io/_format_handlers.py`) defines the
formats `save_simulation_results` / `load_simulation_results` understand:

- **CSV** (`OutputFormat.CSV`, default)
- **JSON** (`OutputFormat.JSON`)
- **HDF5** (`OutputFormat.HDF5`)
- **Pickle** (`OutputFormat.PICKLE`)
- **Parquet** (`OutputFormat.PARQUET`)

`export_analysis_report` only supports `format_type="json"` or `"html"`.
Every save also embeds a `ProvenanceInfo` record (timestamp, git SHA, model
file hash if given, and any explicit `parameters`) — a JSON-format save
nests it under `provenance`; a CSV-format save gets a commented provenance
header.

## Usage Examples

### Accessing Results Programmatically

```python
from src.shared.python.data_io.output_manager import OutputManager, OutputFormat

# Initialize output manager (base_path defaults to <repo root>/output,
# or $UPSTREAM_DRIFT_OUTPUT_DIR if set)
output = OutputManager()
output.create_output_structure()

# Save simulation results
path = output.save_simulation_results(
    results_df, "swing_001", format_type=OutputFormat.CSV, engine="mujoco"
)

# List available simulations
simulations = output.get_simulation_list(engine="mujoco")

# Load a specific simulation
results = output.load_simulation_results("swing_001", engine="mujoco")

# Export an analysis report
output.export_analysis_report(analysis_data, "swing_optimization")

# Clean up files older than 30 days
output.cleanup_old_files(max_age_days=30)
```

Module-level `save_results()` / `load_results()` convenience functions
(same module) wrap a default-constructed `OutputManager` for one-off calls.

### Command Line

There is currently no `upstream-drift output` subcommand — the
`upstream-drift` console script (`launch_upstream_drift.py`) only exposes
`--classic`, `--api-only`, `--engine`, `--port`, and `--no-browser` for
launching the application itself. Output files are produced as a side
effect of running simulations (via the REST API, the desktop app, or the
video-pose pipeline — see the REST API Integration section above), not
through a dedicated CLI. Use the `OutputManager` API above for
programmatic listing, loading, or cleanup.

## Cleanup and Maintenance

`OutputManager.cleanup_old_files(max_age_days=30)` removes files older than
`max_age_days` from the `cache/`, `simulations/`, and `analysis/`
directories (see `src/shared/python/data_io/_simulation_store.py`). There is
no automatic/scheduled cleanup and no configurable cache-size threshold or
archival step — those must be scripted or run manually by calling this
method.

```python
from src.shared.python.data_io.output_manager import OutputManager

output = OutputManager()
removed_count = output.cleanup_old_files(max_age_days=30)
```

## Best Practices

1. **Use Descriptive Filenames**: Include key parameters so saves stay identifiable even after the timestamp suffix is appended.
2. **Regular Cleanup**: Call `cleanup_old_files()` periodically to bound `output/` growth.
3. **Backup Important Results**: Keep copies of significant findings outside `output/`.
4. **Don't Rely on `analysis/`/`exports/` Subdirectories**: They are pre-created but not populated by `OutputManager` — check the producing tool's own docs for where it actually writes.

## Troubleshooting

**Disk Space**: Monitor output directory size and run `cleanup_old_files()` regularly.

```bash
du -sh output/
```

**Permission Errors**: Ensure write permissions to the output directory (the `base_path.exists()` invariant on `OutputManager` will raise if the directory is missing entirely).

**Missing Results**: Use `get_simulation_list(engine=...)` to confirm what was actually persisted, and check application logs — `OutputManager` logs `simulation_results_saved` / `simulation_save_failed` events via the standard `logging` module.
