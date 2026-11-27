# Iron Divot Diagnostic Reports

`src.tools.iron_divot_report.export_divot_report` presents the existing
`DivotResult` from `compute_divot` as a fresh JSON-compatible dictionary.
This reporting slice references #15523. It adds no force calculation,
flight penalty, engine integration, or sampled-contact classification.

## Contract and Units

The `format` is `upstreamdrift/iron-divot-report/1.0`.
Every report carries `model_status: uncalibrated_heuristic`; serialization
does not qualify the underlying parabolic sole-path/shear-work model.
The model source remains `src/shared/python/physics/divot_model.py`.

All eleven `DivotResult` fields retain their names and values except the
undefined entry/exit offsets for no contact, which become JSON `null`.
Offsets are metres along the target direction relative to the ball;
length and depth are metres, volume is cubic metres, energy loss is joules,
and speed/spin retentions are dimensionless. Flags remain JSON booleans.
No new numerical method, calibration, or equation is introduced.

The exporter rejects non-boolean flags, non-finite contact offsets or
magnitudes, negative magnitudes, inverted contact offsets, and retentions
outside `0 <= spin <= speed <= 1`. No-contact geometry/loss must be zero,
retentions must be one, and the fat-shot flag must be false. Only undefined
no-contact offsets are mapped to null; invalid losses are never silently
replaced with zeros. This is a serialization boundary, not a full validator
of physical consistency between independently fabricated model fields.

## Usage and Acceptance

```python
import json
from src.shared.python.physics.divot_model import SwingArc, compute_divot
from src.shared.python.physics.terrain.materials import MATERIALS
from src.tools.iron_divot_report import export_divot_report

result = compute_divot(SwingArc(-0.05, -0.01), 40.0, MATERIALS["fairway"])
payload = json.dumps(export_divot_report(result), allow_nan=False)
```

Acceptance checks use real model outputs for fat, ball-first and no-contact
cases, strict JSON round trips, field preservation, fresh output mappings,
and invalid-result refusal. They establish software behavior only.
Reproduce with `python3 -m pytest tests/unit/physics/test_iron_divot_report_15523.py
tests/unit/physics/test_divot_model_15523.py -q --no-cov`.
The supplementary editable [LaTeX reference](iron_divot_report.tex) records
the same reporting contract; it is separate from the governed design manual.
