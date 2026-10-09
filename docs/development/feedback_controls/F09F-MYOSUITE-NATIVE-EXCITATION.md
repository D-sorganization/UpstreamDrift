# F09f MyoSuite Native Excitation Replay

Issue: [UpstreamDrift #11918](https://github.com/D-sorganization/UpstreamDrift/issues/11918)
Parent: [F09 #11793](https://github.com/D-sorganization/UpstreamDrift/issues/11793)
Dependencies: F09c strict native replay boundary, Tools T01
`experiment-replay/1.0.0`, installed MyoSuite 3.0.0 / MuJoCo 3.6.0.

## Delivered

`src/engines/physics_engines/myosuite/python/native_excitation_replay.py`
consumes an actual T01 bundle as direct normalized post-mapping excitation
through native MyoSuite/MuJoCo. It never treats a Gym `[-1, 1]` action as
excitation and does not invert the MyoSuite action transformation. The native
plant is advanced for the explicit frame skip with a fixed horizon and ZOH
input. No Gym `step`, task observation, reward, termination callback, tracking
or reset path is reachable through this provider.

The complete state binds separate q/v, activation, actuator control, full
MuJoCo integration state, and the supported Gym/MyoSuite wrapper state. The
provider digest binds runtime and provider sources, environment registration,
wrapper chain, actuator mapping, integration, solver, frame skip and direct
control boundary. Source bytes are hashed before/after environment creation;
loaded model, schema, actual initial payload, applied input and executed policy
are checked through T01/F09. Readback of `data.ctrl` must exactly equal the
frozen normalized excitation before each step.

F09 dispatch routes only an explicit MyoSuite binding to this adapter; it
rejects a torque inventory row. The output receipt stays unqualified, includes
no local paths, and cannot set F01 qualification. The two required MyoSuite
driver/iron rows and all six engine IDs remain in the report. The built-in
elbow task is a runtime test fixture only and does not bind either production
golf model.

## TDD and Validation

The contract tests cover valid open-loop excitation, torque-row rejection,
observation-enabled policy rejection, exact input digest and unqualified
receipt, actual F09 dispatch, all-engine/all-row blocker retention, and
wrapper-state snapshot/restore through the explicit supported wrapper chain.
The shared chain-walk helper keeps snapshot and restore traversal identical.

Local Windows command:

```powershell
ruff check src/engines/feedback_native_execution.py src/engines/physics_engines/myosuite/python/native_excitation_replay.py tests/unit/engines/myosuite
pytest -q tests/unit/engines/myosuite/test_native_excitation_replay_contract.py tests/unit/engines/myosuite/test_native_excitation_replay_runtime.py tests/unit/engines/test_feedback_native_execution.py
```

Before the wrapper-state refactor: Ruff passed; 17 tests passed and two
optional tests skipped locally because the Python 3.13 environment lacks
MyoSuite and the Drake Python package. An
isolated owned Python 3.12 environment on DeskComputer has MyoSuite 3.0.0,
myo-sim 0.2.3, Gymnasium 1.2.3, and MuJoCo 3.6.0. Its runtime test was run
after copying the final adapter source into a disposable task-owned test
overlay and passed: 1 passed, 2 unknown-marker warnings in 0.72 seconds.
The test uses `myoElbowPose1D6MFixed-v0`, six actual native muscle controls,
and a short native horizon. This validates the actual provider seam only; it
does not establish a MyoSuite golfer mapping, private-data result, physiology,
or scientific acceptance.

The installed MyoSuite 3.0.0 distribution metadata reports
`License-Expression: Apache-2.0` and homepage `https://www.myosuite.org`.
The upstream [MyoSuite source repository](https://github.com/MyoHub/myosuite)
publishes its Apache-2.0 license; the [3.0.0 PyPI release page](https://pypi.org/project/MyoSuite/3.0.0/)
identifies this release and documents the `myo-sim==0.2.3` model-package pin.
The installed `myo-sim` 0.2.3 metadata reports `License: Apache 2.0`; its
[PyPI release page](https://pypi.org/project/myo-sim/0.2.3/) identifies the
model package. This records the software and model-package provenance for the
public test lane; it makes no license or provenance claim about future golf
models or captured data.

The follow-up wrapper-state refactor adds one contract test; its focused suite
passes 7 tests, with the optional native MyoSuite runtime test skipped in the
local Python 3.13 environment.

## Follow-Up

The generic MyoSuite engine's legacy four-tuple `step()` interpretation remains
untouched because the native adapter never calls Gym `step`. If a supported
legacy Gym version is needed, fix and test that wrapper path separately. Add
driver/iron bindings only after reviewed model/provider identity, full-state
restore and native acceptance evidence are available; do not use the elbow
fixture to satisfy their inventory rows.

Canonical manual: `manuals/upstreamdrift/chapters/31-myosuite-native-excitation-replay.qmd`.
