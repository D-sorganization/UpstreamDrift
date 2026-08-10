# Tools Ground-Model Integration

UpstreamDrift consumes the ground model owned by the Tools repository through
one fail-closed boundary. The dependency arrow is one way:

```text
UpstreamDrift API / PyQt / React
              |
              v
src.shared.python.ground_model
              |
              v
shared.python.swing_sim.ground  (Tools authority)
```

UpstreamDrift does not copy the ground physics, wire records, schema builders,
or reference executor. The consumer gateway binds only the Tools public façade
and returns its records unchanged, preserving its validation and provenance.

## Supported contract

The first consumer slice requires these exact Tools exports:

| Export                                      | Required value or behavior           |
| ------------------------------------------- | ------------------------------------ |
| `REQUEST_SCHEMA_VERSION`                    | `flight-to-ground-request/v1`        |
| `RESULT_SCHEMA_VERSION`                     | `flight-to-ground-result/v1`         |
| `GROUND_REFERENCE_EXECUTION_SCHEMA_VERSION` | `ground-reference-execution/v1`      |
| `request_from_json`                         | strict request parser                |
| `result_from_json`                          | strict result parser                 |
| `run_ground_reference`                      | canonical bounded reference pipeline |

`probe_ground_contracts()` is safe for optional-install diagnostics. It reports
the feature unavailable when the Tools package is absent or incompatible.
`load_ground_contract_gateway()` is the strict execution boundary and raises a
typed import or compatibility failure instead of selecting another schema.

```python
from src.shared.python.ground_model import load_ground_contract_gateway

gateway = load_ground_contract_gateway()
request = gateway.parse_request(request_json)
result = gateway.run_reference(request)
response_json = gateway.serialize_result(result)
```

The gateway does not convert units, frames, enums, warnings, calibration, or
provenance. Those values remain governed by the Tools v1 contracts.

## Release boundary

This adapter is only a partial delivery of Tools issue #4276. It does not make
the ground model available in a clean UpstreamDrift install by itself.

As of 2026-08-10, UpstreamDrift `main` is `d8ac4651598f962a6f5bab11670800cf01f70a8a`
and has two older Tools authorities:

- `vendor/ud-tools` gitlink: `ff4240217005e1415ca409fd124e50b64ee642d2`
- Cargo `tools-core` revision: `ea2690362481379b94135894f9dfac2b70d1bc65`

Neither is claimed to contain the in-flight ground release. The exact pin may
change only after the Tools ground stack is reviewed, protected checks pass,
and its final merge commit is known. The same reviewed authority must then be
used for clean-clone Python and any compiled ground-result consumers.

UpstreamDrift PR #8369 is not a release parent. It was closed unmerged on
2026-08-10 as superseded by the clean, merged launch-monitor PR #8432.

## Remaining acceptance gates

1. Merge the Tools ground stack through ordinary protected behavior.
2. Repin `vendor/ud-tools` to the exact reviewed Tools merge; update the Cargo
   pin too if the released Upstream path consumes the ground wire/kernel there.
3. Add FastAPI, PyQt6, and React adapters that delegate to this gateway and
   register their real parity status without claiming absent surfaces.
4. Run clean recursive-clone and built-wheel smoke tests with no sibling Tools
   checkout on `PYTHONPATH`.
5. Verify canonical request/result round trips, cancellation and unavailable
   outcomes, UI error presentation, generated frontend contracts, and protected
   CI before release.

## Local verification

```bash
python3 -m pytest tests/unit/ground_model/test_consumer_gateway.py -q
python3 -m ruff check src/shared/python/ground_model \
  tests/unit/ground_model/test_consumer_gateway.py
python3 -m ruff format --check src/shared/python/ground_model \
  tests/unit/ground_model/test_consumer_gateway.py
```

For a pre-merge cross-repository smoke only, put one reviewed Tools checkout's
`src/` ahead of this repository on `PYTHONPATH`. Do not publish or encode that
branch-head SHA as the final dependency pin.
