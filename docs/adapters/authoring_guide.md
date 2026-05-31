# Adapter authoring guide

This is the repeatable path for adding a physics-engine adapter to Canonical
Core without inventing a parallel contract. A new adapter is mergeable only
when it exports a canonical model, remaps native state to `canonical-v2`,
declares capabilities, and passes the conformance gate.

Use this guide for engine adapters that touch state, dynamics, or engine I/O.
Pose-only work can start with the existing
`src/shared/python/pose_interchange/` APIs, but the same boundary rules apply.

## One-week outcome

By the end of the first week a contributor should have:

1. A small adapter module behind an existing public protocol.
2. A deterministic model export or loader path for the test fixture.
3. Explicit native-to-canonical remaps for position, velocity, acceleration,
   units, frames, quaternion order, and joint signs.
4. An `EngineCapabilities` report that advertises only verified support.
5. Unit tests for the adapter contract and a conformance run wired into the
   adapter PR.

Do not start by editing application code. First make the engine satisfy the
adapter boundary, then expose it through the registry, API, or UI.

## Existing contracts to reuse

| Concern                        | Contract                                                                                           | Use                                                                                                                            |
| ------------------------------ | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Dynamic state                  | [`docs/conventions/canonical-v2.md`](../conventions/canonical-v2.md)                               | Defines `(q, v, a, t)`, SI units, Z-up world frame, quaternion `wxyz`, body-local base angular velocity, and manifold updates. |
| Architecture decision          | [`docs/adr/0026-canonical-dynamic-state-v2.md`](../adr/0026-canonical-dynamic-state-v2.md)         | Explains why adapter remaps are required at the engine boundary and why CC-7 checks round-trip plus ID/FD consistency.         |
| Pose adapter API               | `src/shared/python/pose_interchange/protocol.py`                                                   | `PoseConventionAdapter`, `JointSlot`, `to_canonical`, `from_canonical`, and `joint_layout`.                                    |
| Pose adapter registry          | `src/shared/python/pose_interchange/adapters/__init__.py`                                          | `ADAPTER_REGISTRY` maps stable engine ids to adapter classes.                                                                  |
| Shared adapter helpers         | `src/shared/python/pose_interchange/adapters/_base.py`                                             | Quaternion order helpers, Euler conversion, default joint layouts, and joint encode/decode helpers.                            |
| Engine capability API (CC-5)   | `src/shared/python/engine_core/capabilities.py`                                                    | `CapabilityLevel` and immutable `EngineCapabilities`.                                                                          |
| Capability taxonomy            | [`docs/architecture/engine_capability_taxonomy.md`](../architecture/engine_capability_taxonomy.md) | Defines `FULL`, `PARTIAL`, and `NONE` promotion criteria for gradient, rollout, contact, and optimization surfaces.            |
| Backend trace API              | `src/shared/python/simulation_backends/protocol.py`                                                | `SimState`, `Trace`, `BatchTrace`, and optional `DynamicsProvider` / `BatchedBackend` protocols for rollout-style backends.    |
| Model generation/export (CC-3) | `src/shared/python/model_generation/`                                                              | URDF generation, URDF/MJCF conversion, validation, and library APIs used by model-export flows.                                |

The worked examples to read before authoring a new adapter are:

- `src/shared/python/pose_interchange/adapters/pinocchio.py`
- `src/shared/python/pose_interchange/adapters/mujoco.py`
- `src/shared/python/pose_interchange/adapters/opensim.py`
- `src/shared/python/pose_interchange/adapters/drake.py`
- `src/engines/physics_engines/jaxsim/jaxsim_backend.py`
- `src/engines/physics_engines/jaxsim/README.md`

The Jules adapter-scaffold work is tracked by
[#6780 / CC-8](https://github.com/D-sorganization/UpstreamDrift/issues/6780).
When that template lands, start from it and then fill in each section below.
Until then, copy the smallest existing adapter that matches your boundary:
`pose_interchange` for pose/state conversion, or an engine package backend for
dynamics/rollout.

## Authoring checklist

### 1. Define the adapter boundary

Pick the narrowest existing protocol that matches the engine surface:

- Pose or starting-state interchange: implement `PoseConventionAdapter`.
- Rollout backend: implement `SimulationBackend`; add `DynamicsProvider` only
  when mass matrix and bias forces are verified.
- Full engine integration: expose a `get_capabilities()` method returning
  `EngineCapabilities` and keep optional native imports lazy.

Write down the native engine layout before coding:

- native `q`, `v`, and `a` order;
- base pose order, especially quaternion component order;
- angular-velocity frame;
- joint units and signs;
- fixed, floating, or model-dependent joint order;
- unsupported coordinates that must be dropped rather than guessed.

### 2. Export or load the canonical model

Use the existing model-generation path where possible:

1. Generate or validate a URDF fixture with `model_generation` builders and
   validators.
2. Convert URDF/MJCF only through `model_generation.converters`, not a local XML
   string builder.
3. Keep native model files as derived artifacts unless the native format is the
   engine's source of truth.
4. Record which segment frames are native and which are canonical. `canonical-v2`
   expects SI units and Z-up world-frame semantics at the adapter boundary.

For a first adapter PR, prefer a tiny fixture that exercises floating base plus
two or three internal joints. Add the full biomechanical model after the
contract is green.

### 3. Implement native-to-canonical remaps

Every adapter needs an explicit remap table. Do not rely on adjacent array
indices being "obvious".

Minimum remap requirements:

- `q`: native configuration to `canonical-v2` order. Base orientation must be a
  unit quaternion in `(w, x, y, z)`.
- `v`: native generalized velocity to `canonical-v2` order. Base angular
  velocity must be body-local.
- `a`: native generalized acceleration to the same tangent-space layout as `v`.
- `t`: seconds.
- joint map: canonical joint name, native joint name, start index, length,
  units, sign, and limits. For pose adapters, this is `JointSlot`.
- provenance: stamp convention, frame, units, engine name, adapter version, and
  source model path or hash when materializing outputs.

For pose-only adapters, implement:

```python
from src.shared.python.pose_interchange.canonical import CanonicalPose
from src.shared.python.pose_interchange.protocol import JointSlot


class NewEngineAdapter:
    engine_name = "new_engine"

    def joint_layout(self, model=None) -> dict[str, JointSlot]:
        ...

    def from_canonical(self, pose: CanonicalPose, *, model=None):
        ...

    def to_canonical(self, engine_q, *, model=None) -> CanonicalPose:
        ...
```

For dynamics adapters, add equivalent `to_canonical_state` /
`from_canonical_state` helpers near the engine package until the canonical
state type is the public API. Keep them pure and unit-tested.

### 4. Declare capabilities conservatively

Use `CapabilityLevel.NONE` until the feature has a test that proves it through
the adapter API. Use `PARTIAL` when the native engine supports the feature only
for some model classes, behind side APIs, or without suite-normalized output.
Use `FULL` only when the suite adapter exposes it and conformance covers it.

Example:

```python
from src.shared.python.engine_core.capabilities import (
    CapabilityLevel,
    EngineCapabilities,
)


def get_capabilities(self) -> EngineCapabilities:
    return EngineCapabilities(
        engine_name="NewEngine",
        mass_matrix=CapabilityLevel.PARTIAL,
        jacobian=CapabilityLevel.FULL,
        contact_forces=CapabilityLevel.NONE,
        inverse_dynamics=CapabilityLevel.FULL,
        forward_sim=CapabilityLevel.FULL,
        contact_step=CapabilityLevel.PARTIAL,
    )
```

Update [`docs/architecture/engine_capability_taxonomy.md`](../architecture/engine_capability_taxonomy.md)
or [`docs/engines/engine_capabilities.md`](../engines/engine_capabilities.md)
when a new engine advertises non-`NONE` support. The docs must match
`EngineCapabilities.to_dict()` output.

### 5. Add contract tests before broad integration

The first tests should run without the native engine when possible. Use an
injectable fake model or mock native API and assert:

- unsupported convention tags fail clearly;
- arrays with wrong rank or length fail clearly;
- `from_canonical(to_canonical(native_state))` round-trips within documented
  tolerance;
- `to_canonical(from_canonical(canonical_state))` preserves canonical state;
- quaternion output is scalar-first and unit-norm;
- base angular velocity is body-local;
- joint signs and units are pinned with at least one non-zero asymmetric value;
- capabilities serialize with `to_dict()` and recover with `from_dict()`;
- optional native imports are lazy, so core imports work without the engine
  wheel installed.

Native-engine tests should be marked with the existing pytest markers such as
`requires_mujoco`, `requires_drake`, `requires_pinocchio`, `requires_jaxsim`,
`slow`, or `live_simulation`.

### 6. Run the conformance gate

CC-7 is the merge gate for adapter PRs that touch engine I/O. The conformance
suite should include:

- canonical round-trip checks for `q`, `v`, `a`, and `t`;
- inverse-dynamics versus forward-dynamics self-consistency;
- mass matrix and bias-force shape and symmetry checks when advertised;
- trace schema checks for rollout backends;
- tolerance-based comparisons only. Never require bitwise equality across
  engines.

Use the closest scoped command available in the repo. Typical commands are:

```bash
python3 -m pytest tests/unit/pose_interchange -q
python3 -m pytest tests/cross_engine -q
python3 -m pytest -m "not slow and not live_simulation" -q
python3 scripts/check_docs_governance.py
python3 scripts/check_doc_catalog.py
```

If the native engine is unavailable locally, the PR must still include the
mocked contract tests and clearly state which marked heavy tests need CI or the
self-hosted runner.

## PR readiness template

Paste this into the PR body and fill it out:

```markdown
Closes #<issue>

Adapter boundary:

- Engine:
- Protocol/API implemented:
- Native model source:
- Canonical model/export path:

State remap:

- q:
- v:
- a:
- quaternion order:
- angular velocity frame:
- units/frame:
- unsupported coordinates:

Capabilities:

- FULL:
- PARTIAL:
- NONE:

Conformance:

- Unit contract tests:
- Cross-engine/conformance tests:
- Heavy/native tests skipped locally:

Docs:

- Capability matrix updated:
- Adapter guide or engine guide updated:
```

## Common failure modes

- Hidden quaternion-order bugs. Pin the order with a non-identity rotation, not
  just `[1, 0, 0, 0]`.
- Velocity frame drift. Native spatial velocities often differ from the
  body-local angular velocity required by `canonical-v2`.
- Overstated capabilities. `FULL` means the suite adapter exposes and verifies
  the feature, not merely that upstream has a related API.
- Native imports at module import time. Optional engines must not break core
  imports on machines without the engine installed.
- Hand-built XML. Use model-generation builders/converters so model export,
  validation, and cross-engine tests share one path.
