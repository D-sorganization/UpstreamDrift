# Moving-Horizon Estimator

The near-real-time estimator lives in
`src/shared/python/estimation/moving_horizon.py`. It is a bounded windowed
facade over the CC-19 single-trial MAP solver:

- callers append strictly increasing canonical-v2 sample times and state
  samples with `MovingHorizonEstimator.append_samples()`;
- only `MovingHorizonOptions.window_size` samples are retained;
- `solve_next()` advances deterministically when the retained window has at
  least `step_size` new samples since the previous solve;
- shared parameters are fixed for the window and passed into the same residual
  and Jacobian callables used by the batch MAP path;
- the first window initializes spline coefficients from samples, and later
  windows warm-start by evaluating the previous solved spline on the new window
  times and carrying q/v into the new spline;
- an optional callback receives each `MovingHorizonResult`, whose
  `callback_payload()` is JSON-serialisable for realtime bridge publishing.

The default latency budget is 50 ms per window, matching the local realtime IPC
p99 guidance. The current Python facade records achieved per-window latency and
flags `over_budget`; production Rust integration should drive the residual and
Jacobian hot path through `rust_core/upstream-realtime` while preserving this
API contract.

Focused coverage is in `tests/unit/estimation/test_moving_horizon_estimator.py`
for window advancement, state carryover, fixed-parameter objective construction,
and callback payloads.
