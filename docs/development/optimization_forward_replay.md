# Optimization Forward Replay Runner

Refs #15493, Slice 4. `scripts.replay_optimized_candidate.replay_candidate`
executes fresh closed-loop forward dynamics using the existing
`SimulationBackend`, `SimState` and `Trace` contracts. It contains no physics
model, optimizer, bundle codec or tolerance-comparison implementation.

## Integration Contract

After loading and validating a candidate bundle, the caller reconstructs two
factories from its pinned configuration: a fresh plant and a fresh controller.
The controller closure restores its complete initial internal state and owns
the reference trajectory, feedforward spline, PD gains and saturation rules.
It receives a copy of the live plant state at each control boundary. Initial
plant state is restored once; later states are obtained only by stepping the
plant. Recorded rollout arrays and recorded torques are never runner inputs.

`dt` is the control period in seconds. Each control command is held constant
for `substeps` equal integration steps (left-endpoint ZOH). This adapter supports
equal position, velocity and control dimensions and fixed-rate ZOH only.
Unsupported controller/hold/integrator semantics must be rejected by the caller
before invoking it. The generic backend interface cannot attest model hashes,
runtime pins, controller state or integrator selection; factories are trusted
configuration boundaries and must return fresh instances for each invocation.

The returned trace samples control boundaries, including the terminal state.
Its `dt` records the integration step; metadata records the control period.
`u[i]` is commanded control over interval i, with a declared unused zero final
row. It is not measured applied torque: `torques` remains absent. Applied-effort
audits require separate backend measurements. Invalid control, nonfinite state,
wrong state dimensions, failed initial restoration or clock drift raises an
error without returning a partial result. The clock guard is an execution
schedule check, not a physical replay tolerance.

## Verification and Limits

Run `python3 -m pytest tests/unit/scripts/test_replay_optimized_candidate.py -m unit --no-cov`.
Tests exercise live feedback on an explicit unit-mass fixture, reset isolation,
control timing, invalid output and repeatability through the existing analytical
ODE backend. No matched swing, contact, flight or optimization experiment is
claimed. A separate comparison gate must validate the fresh trace against the
recorded bundle under explicit per-quantity tolerances and matching runtime
pins. No physical acceptance or winning-candidate promotion follows merely from
successful execution. Bundle decoding, replay comparison, controller assembly,
step refinement and the optimization/flight loop remain separate integration work.
