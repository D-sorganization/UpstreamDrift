# Recorded Optimization Bundle Comparison

Epic #15493, Slice 5: compare stored baseline and candidate evidence using the
existing Solution Bundle codec. From the repository root:

```powershell
python3 -m scripts.compare_optimization_bundles --baseline PATH_A --candidate PATH_B
```

The command validates both JSON/NPZ bundles, including array hashes, before
reporting strict JSON to stdout. Model hash, asset hashes, frame, joint order
and environment must match. Controller, integrator, seed, source revision,
solution identity, parent identity and assumptions remain visible per bundle.
Different controller or integrator settings do not establish a controlled
experiment; inspect those settings before attributing a metric difference.

Each metric row gives baseline, candidate and candidate-minus-baseline delta.
`carry_m`, `ball_speed_ms`, `clubhead_speed_ms` and `lateral_dispersion_m` are
always shown; absent values are null, never zero. Other producer keys are
preserved without unit conversion or inferred aliases. Numeric values must be
finite and cannot be booleans. Feasibility margin keys are unioned; negative
values remain visible and missing margins stay null. No aggregate feasibility
verdict is inferred from an incomplete set of margins.

`rollout_m` and `total_distance_m` are separate from flight metrics: carry is
never computed by adding rollout. Rollout metrics are unvalidated, and rollout
optimization remains deferred. The report only compares recorded scalars; it
does not recompute physics, validate contact, or replay the controller. Forward
replay status is always unverified and drift is null. Use the independent
forward replay runner and per-quantity runtime-pinned comparison separately.

Validation with synthetic, non-capture bundles:

```powershell
python3 -m pytest tests/unit/scripts/test_compare_optimization_bundles.py tests/unit/optimization/test_solution_bundle.py -q --no-cov
python3 -m ruff check scripts/compare_optimization_bundles.py tests/unit/scripts/test_compare_optimization_bundles.py
```
