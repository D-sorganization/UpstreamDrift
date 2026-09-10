# Model Parameters to Simulation Backends

## Responsibilities

`GolfModelParams` owns the double-pendulum model parameters. Its analytical
conversion and `simulation_backends.mjcf.params_to_mjcf` render that same model.
`make_backend(name, params, **kwargs)` constructs the selected backend lazily.

## Data Contract

Read units on each parameter field: lengths and masses use SI, while explicit
angle fields such as plane inclination retain their declared degrees.
Use the shared `SimulationBackend` protocol and `Trace`/`BatchTrace` outputs.
`DynamicsProvider` and `BatchedBackend` are optional capabilities and require
their own capability checks. Do not infer them from a backend name.

## Lifecycle and Failures

Construct parameters once and pass them into the factory. Unknown names raise
`UnknownBackendError`; optional backend dependencies are checked when requested.
Registered names do not establish installation or GPU readiness. Compare
cross-engine numerical outputs with the qualified tolerances, not exact equality.

## Evidence

- [Parameter Authority](../../../src/shared/python/simulation_backends/model_params.py)
- [Backend Factory](../../../src/shared/python/simulation_backends/factory.py)
- [Both Renderers Change Together](../../../tests/unit/simulation_backends/test_foundation.py)
- [Backend Usage Examples](../../simulation_backends/USER_GUIDE.md)

## Rationale

Changing a shared parameter must affect both analytical and rendered models.
Review renderer equivalence and capability behavior when changing this boundary;
do not maintain a second hand-edited model in a presenter.
