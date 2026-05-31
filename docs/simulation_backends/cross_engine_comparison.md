# Cross-engine comparison reports

CC-27 adds a reusable comparison service over the shared `Trace` schema. It is
engine-agnostic: selected backends are reset to the same initial state, receive
the same control history, and produce a side-by-side report with provenance on
every panel.

## Command

Run a passive comparison with the always-available ODE backend twice:

```bash
python -m src.shared.python.simulation_backends.compare_cli \
  --engines ode,ode \
  --horizon 100 \
  --dt 0.005 \
  --output reports/cc27.md
```

When optional engines are installed, select them by name:

```bash
python -m src.shared.python.simulation_backends.compare_cli \
  --engines ode,mujoco \
  --controls swing_controls.csv \
  --horizon 200 \
  --dt 0.005 \
  --format json \
  --output reports/ode_vs_mujoco.json
```

Control input may be JSON or CSV. A one-column control file is repeated to the
configured `--control-dim` so simple torque profiles can drive two-DoF backends.

## Python API

```python
import numpy as np

from src.shared.python.simulation_backends import (
    ComparisonInput,
    GolfModelParams,
    SimState,
    compare,
    make_backend,
    write_report,
)

params = GolfModelParams.default()
engines = [make_backend("ode", params), make_backend("mujoco", params)]
report = compare(
    engines,
    ComparisonInput(
        horizon=200,
        dt=0.005,
        controls=np.zeros((200, 2)),
        initial_state=SimState(q=np.array([1.2, -0.6]), v=np.zeros(2)),
    ),
)
write_report(report, "reports/ode_vs_mujoco.md")
```

Use `compare_traces()` when engines have already written HDF5 traces and the
report should compare the data without rerunning simulations. Counterfactual
panels require live objects satisfying `DynamicsProvider`, so trace-only reports
omit ZTCF/ZVCF metrics.

## Panels

### Kinematics

The kinematics panel compares generalized positions `q` and velocities `v`.
Rows and columns are aligned to the common overlapping shape before deltas are
computed, which keeps reports usable when one backend records fewer samples or
coordinates.

### Kinetics

The kinetics panel compares recorded `torques` when present, otherwise applied
control history `u`. Missing channels are skipped instead of filled with
invented values.

### Counterfactuals

ZTCF and ZVCF panels are pointwise diagnostics computed only for backends that
implement `DynamicsProvider`. They reuse the canonical
`ztcf_acceleration()` / `zvcf_acceleration()` primitives and do not perform
forward-integrated counterfactual rollouts.

### Wrench

The wrench panel compares the optional six-axis `Trace.wrench` channel
`[fx, fy, fz, tx, ty, tz]`. It appears only when at least one trace carries
wrench data.

## Divergence annotations

Each non-baseline engine is compared to the first selected engine. An annotation
records max absolute delta, RMS delta, tolerance, severity, and a registry key.
The default registry keys are:

| Key                 | Meaning                                         |
| ------------------- | ----------------------------------------------- |
| `kinematics.q`      | Generalized position drift                      |
| `kinematics.v`      | Generalized velocity drift                      |
| `kinetics.control`  | Applied force / torque drift                    |
| `ztcf.acceleration` | Zero-torque counterfactual acceleration drift   |
| `zvcf.acceleration` | Zero-velocity counterfactual acceleration drift |
| `wrench.contact`    | Contact wrench drift                            |

Severity is `within_tolerance`, `minor`, or `major`. The top-level
`ComparisonReport.divergences` property includes only out-of-tolerance
annotations; each panel still retains all annotations for auditability.

## Provenance

Every panel carries `provenance_by_engine`. If a trace has flat
`provenance_*` metadata from `ProvenanceStamp`, those fields are grouped under
`stamp`. Otherwise, the report includes the trace backend, schema version, time
step, sample count, and any scalar trace metadata.
