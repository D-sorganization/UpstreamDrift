# F01f Compiled Command Admission (#11955)

Branch: `feat/f01f-compiled-command-admission-11955`.

## Contract

The T02 `compiled-actuator-profile/1.0.0` row remains structural: it provides
an opaque artifact reference, implementation ID/version and digest. F09
resolves those exact immutable bytes and recomputes the expected canonical
profile from the loaded native MuJoCo model plus the T01 replay bundle before
the first step. The profile covers native model and provider/runtime
identities, resource closure, exact native actuator law and ordered channel
schema, complete initial state, applied input, time grid and execution policy.

F09 emits a frozen `native-actuator-command/1.0.0` receipt carrying the
separate inventory-to-execution correspondence, those native identities, the
submitted T01 horizon and output-state digest. It does not establish the
duration or completeness of any original capture. F01 accepts the new typed pathway under
`feedback-comparison/1.2.0`; the existing 1.1.0 comparison contract remains
supported for its prior evidence. The receipt is an integrity record, not a
cryptographic signature. A serialized receipt from an external caller remains
an assertion until its source/profile/initial/input/output lineage is
independently verified or native replay is repeated.

The coarse inventory drive row (`muscle_excitation`) is not the fine input
kind. Mixed compiled `ACTUATOR_COMMAND` may include motors, muscles and other
actuator laws. An actual motor-free command sequence still does not establish
muscle-only assistance, and this F01 pathway rejects that claim. Cross-engine
same-input comparison also fails closed without a separately reviewed
semantic mapping. Inventory package/provider identity is not overwritten by
the underlying native execution identity.

Runtime F09 consumers obtain T01 contract objects through
`native_replay_contract_types()`, which loads the pinned package under the
private `_pinned_tools__` namespace. This preserves one class/enum identity
without extending `sidekick.lab.__path__` or publishing a second
`sidekick.lab.mocap` module. Type-checking-only imports may use the package's
public source name; executable code and tests use the facade.

## Validation and Limits

The focused synthetic MuJoCo 3.8.0 fixture exercises the T02 opaque reference,
loaded-model profile recomputation, profile digest tampering, initial state,
input history, channel order, compiled law, provider and engine identity
mutations, and native output receipt binding. This is underlying MuJoCo
execution only; it does not qualify the official MyoSuite runtime, either
production golfer model, muscle-only assistance, cross-engine physiology,
private capture accuracy, or six-engine parity. Unsupported required rows
remain in the denominator.

Primary tests:

- `tests/unit/engines/myosuite/test_native_direct_model_replay.py`
- `tests/unit/engines/test_feedback_native_execution.py`
- `tests/unit/engines/test_feedback_comparison.py`

Run direct native fixture tests with the retained exact MuJoCo 3.8.0 runtime;
record any other environment separately. Do not substitute the official SDK
or production-asset gates with a synthetic pass.
