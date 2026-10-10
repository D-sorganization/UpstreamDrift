# F09g Direct Model Replay Kernel

Issue: [UpstreamDrift #11945](https://github.com/D-sorganization/UpstreamDrift/issues/11945)
Parent: [F09 #11793](https://github.com/D-sorganization/UpstreamDrift/issues/11793)
Parent provider: [F09f MyoSuite excitation replay](F09F-MYOSUITE-NATIVE-EXCITATION.md)

## Scope and Boundary

`src/engines/physics_engines/myosuite/python/native_direct_model_replay.py`
adds a reusable direct MuJoCo model-path command replay kernel. It consumes the
existing Tools T01 `experiment-replay/1.0.0` contract and does not add a bundle
schema, F01 inventory, qualification authority, F09 receipt admission, or a
Gym task provider. The production MyoSuite `MujocoEnv` constructor was not
inspected or invoked for this change; no production MyoSuite binding is
registered, and the MyoSuite driver/iron rows remain unqualified.

The kernel accepts a source-bound factory and passes it the original model
path. It verifies each declared resource under the preserved root and hashes
the resource set without copying files to a temporary directory. This keeps
MJCF includes and mesh/texture paths resolvable. The compiled model and provider
implementation/runtime bytes are checked against the typed registration.
The kernel accepts only `engine_id="mujoco"`; every other engine ID is rejected
before the factory is called. The SDK constructor, source path handling, and
wrapped/unwrapped native plant boundary are not pinned here, so package
namespace or class-name checks would be insufficient. An explicit official
adapter can be added only after the exact constructor and runtime are
inspected. The generic MuJoCo kernel tests use a test-only registration; they
cannot claim MyoSuite execution.

The input kind is T01 `ACTUATOR_COMMAND`, distinct from actuator torque,
generalized effort, muscle excitation, or Gym actions. Each ordered channel
must target the compiled `actuator:<native-name>` exactly. The channel
manifest hashes the native actuator law, transmission, dynamics, gain, bias,
activation and limits. This supports a compiled mixed actuator model without
relabeling a motor command as muscle excitation. The final input row remains
integrity-bound at the final sample time; the kernel steps each interval before
that endpoint and does not inject an extra step or require the endpoint value
to repeat the preceding sample.

Initialization restores distinct native `qpos` and `qvel`, activation when
present, actuator controls, and the complete MuJoCo integration state. The
provider binds source/loaded model identity, provider/runtime source, state and
channel schema, actuator law, contact properties, solver/integrator, fixed
native clock, and initialization/input-player policy. It rejects unsupported
plugins, process-global callbacks, wrapper objects, feedback/observation/reset
policies, stale resource/model identities, command readback differences,
native warnings, off-step clocks outside a bounded ULP tolerance, non-unit
quaternion configurations, declared external-load payloads, external loads at
the frozen start, and nonfinite state. It never calls Gym registration, environment `step`, task
observations, reset, or tracking. Returned output remains a native diagnostic
trajectory; it cannot be promoted to F09 qualification by this module.

Resource admission is now bounded to the inspected MuJoCo 3.8.0 disk-MJCF
surface. A hardened XML walk follows native-tested include selection (entry
model directory first, then the including file's directory), and native
`MjSpec` supplies mesh/texture compiler directories and texture cube-face
assets. Every discovered source and asset must resolve inside the preserved
root and match an exact hash already present in the supplied inventory; extra
declared inventory rows remain allowed. When a registration supplies the
expected loaded-model digest, the preflight `MjSpec.compile()` output must
match it before the factory is invoked. Source/resource hashes are checked
again after compile and after the factory loads the native model.

This is not a general MJCF parser or a closed-world guarantee for all MuJoCo
loaders. The admitted disk route permits XML includes, STL/MSH meshes, and PNG
textures (including six-face textures). It rejects absolute/URI paths,
symlink escapes, URDF, `strippath`, content-type overrides, hfields, skins,
`attach`, `flexcomp`, plugins/extensions, VFS assets, and other file-bearing
elements before native parsing. MjSpec compilation and factory-loaded MJB
identity remain separate checks. These semantics were tested with MuJoCo
3.8.0 and the public driver/iron source closure; other runtimes and loaders
remain unsupported.

## TDD and Native Evidence

The synthetic MuJoCo fixture covers direct source-path loading, complete state
restore, exact applied controls, canonical channel targeting, changed terminal
row handling, missing/altered assets, undeclared and escaped includes,
model-root-first and nested include fallback, compiler-relative mesh assets,
texture cube-face inventory, unsupported loader refusal before factory/native
compile, callback refusal before resource loading, actuator-law mismatch,
wrapper rejection, native warning rejection, and the requirement that a
synthetic class cannot claim the official MyoSuite SDK. These tests use an
independently compiled synthetic MJCF model.

An opt-in integration test replays the frozen public MyoSim driver and iron
fixtures under the existing MuJoCo 3.8.0 environment. Configure
`UD_MYO_NATIVE_RESOURCE_ROOT` to the preserved `shared/models/myosuite` root
and `UD_MYO_NATIVE_ARTIFACT_ROOT` to the public
`myosuite_compiled_inventory` receipt directory, then run:

```powershell
python -m pytest tests/unit/engines/myosuite/test_native_direct_model_replay.py -q
python -m pytest tests/unit/engines/myosuite/test_native_model_resource_closure.py -q
```

The replay matches all 31 saved native integration-state samples exactly for
both variants, with 100 ordered actuator-command channels and the original
resource root. The discovered closure contains 49 files per model and is a
subset of the retained 310-file exact-hash manifest; MjSpec compilation matches
the saved driver/iron MJB identities. This corroborates the kernel against the
saved underlying MuJoCo 3.8.0 producer trajectories. It is not MyoSuite package execution, a
MyoSuite 3.6 result, a newly qualified production binding, or a physiology,
marker, capture, or cross-engine equivalence result. The artifacts contain
public synthetic command/state experiments; no private or raw mocap data was
used.

Ruff, focused MyPy, architecture function/parameter budget, and direct-file LoD
checks pass. The central repository pre-PR checks and normal hooks remain the
publication gates for this branch. The installed official MyoSuite SDK source
path/runtime used for the earlier elbow-task smoke was not available in this
worktree, so constructor compatibility and MyoSuite 3.0/MuJoCo 3.6 production
execution remain unverified.

## Next Proof Required

Before routing this kernel through F09 as an official MyoSuite provider, inspect
and bind the exact supported `MujocoEnv` constructor, its path/resource
resolution behavior, compiled actuator mapping, runtime and SDK source bytes.
Then run both required driver and iron variants through that provider with
their prepared full native state and frozen T01 command histories. Preserve
the source/runtime and physics evidence classes as separate identities; do
not reinterpret this MuJoCo 3.8 experiment as MyoSuite 3.6 qualification.
