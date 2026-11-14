---
issue: 15095
summary: "Fix SwingMetricsMixin test harness after velocity delegation and add unit marker (#15095)"
---

- Use concrete `SwingMetricsMixin` and `PhaseDetectionMixin` subclasses in `tests/unit/dbc/test_dbc_runtime_swing_phase.py` so `compute_x_factor_stretch` delegates cleanly to `compute_x_factor_velocity`, mark the file with `pytest.mark.unit` so CI `unit-test-gate` runs it, verify deprecation warning, and add direct `compute_x_factor_velocity` postcondition tests (#15095).
