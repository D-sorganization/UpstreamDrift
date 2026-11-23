# Bunker Flight Trace Contract

`src/tools/bunker_shot_gui/flight.py::BunkerFlight` is the common value
object consumed by bunker flight views and the workbench API (Refs #15497).
Construction rejects invalid traces with `ValueError`, independently of
`DBC_LEVEL`. This is a data-integrity boundary, not physical qualification.

## Coordinates and Invariants

- Positions are metres: x forward, y left, z up above the assumed flat green.
- The flight contains at least two finite xyz points; rollout contains at
  least its landing point. All heights are non-negative.
- The last flight point is the landing. Its height is zero, and carry is
  the non-negative horizontal distance `hypot(landing_x, landing_y)`.
- Rollout starts at that landing and ends on the green. Positive intermediate
  rollout heights are allowed because the existing green solver models bounce.
- The landing and rest heights, endpoint continuity, and carry comparison use
  a 1e-9 m absolute tolerance. Carry also uses 1e-9 relative tolerance,
  matching the existing `ShotOutcome` carry-pairing boundary. This permits
  floating-point residue without changing any physical acceptance threshold.

## Scope and Limits

The existing `bunker_flight` adapter still suppresses refused or invalid launch
states with `None`. Direct callers constructing a malformed `BunkerFlight`
receive an exception before a view can draw it. Coordinates are never repaired,
clamped, or silently connected. A one-point rollout is permitted for an
immediate stop; an empty rollout cannot describe a landing or rest point.

No flight, bounce, friction, or spin equations change. The existing
`BallFlightSimulator` and putting solver remain the calculation providers.
The uncalibrated launch and simplified flat 10 ft stimp green assumptions
remain attached to the trace. These structural checks do not establish
calibration, predictive accuracy, or a physically valid trajectory.

## Reproduction

```powershell
py -3.12 -m pytest -o addopts='' tests/unit/tools/bunker_shot_gui/test_flight_contract_15497.py -q --timeout=60
py -3.12 -m ruff check src/tools/bunker_shot_gui/flight.py tests/unit/tools/bunker_shot_gui/test_flight_contract_15497.py
py -3.12 -m ruff format --check src/tools/bunker_shot_gui/flight.py tests/unit/tools/bunker_shot_gui/test_flight_contract_15497.py
```
