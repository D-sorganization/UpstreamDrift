# F09c Native Replay Execution

Issue #11898 adds a strict consumer for frozen Tools T01 replay bundles. It
uses the authoritative F01 inventory and the native adapter API; it adds no
model catalog, input serializer, integrator, scoring rule, or physics gate.

`NativeAdapterBinding` explicitly connects one inventory row to the T01
adapter model/provider and loaded-model, full state-schema, channel-schema,
and ordered channel identities. F01 inventory provider identity is retained
separately in the receipt. The executor refuses unsupported or unavailable
inventory rows, mixed drive semantics, missing required capabilities, stale
bundle integrity, unexpected replay modes, state observation or feedback,
reset-enabled policy, non-relative time, and inputs outside the current direct
actuator-torque/ZOH adapters. The registered implementation currently invokes
the reviewed MuJoCo and Drake adapters only. Other engines and muscle
excitation remain represented and blocking.

Post-step validation checks the exact initial physical and numerical state,
full native time grid, applied torque samples, state/input/policy/model
identities, finite state and effort output, and complete horizon. Receipts
include content digests and separate `nq`/`nv` dimensions. They exclude local
model paths and remain unqualified. The receipt separately records that reset
is disallowed and leaves the reset count unknown because the adapters do not
instrument a counter; policy alone is not promoted to a measured zero. The
report retains all inventory rows and
the six required engines; it does not convert adapter execution into model,
observation, physiological, or cross-engine qualification.

## TDD and Validation

RED: `python -m pytest -q tests/unit/engines/test_feedback_native_execution.py`
failed during collection because the strict F09c module did not exist.

GREEN: the focused suite passes using independent synthetic bundle fixtures
and an independently generated one-hinge MJCF executed through the actual
MuJoCo native adapter. The fixture tests identity mismatch, tampered applied
input, private model-path omission, unqualified receipts, and missing rows.
The native smoke exercises adapter wiring only; it is not production-model or
physics qualification.

The design note is in
`manuals/upstreamdrift/chapters/13-feedback-comparison.qmd`; the API contract
is summarized in `SPEC.md`. Full repo gates and generated-manual governance
are recorded in the PR after the stacked F01c and native-adapter dependencies
are reconciled.

## Limitations and Next Handoff

T01 v1 currently has reviewed direct torque paths for MuJoCo and Drake. The
executor does not yet admit excitation, force, generalized-effort, externally
forced, or shared-rigid-body-emulation rows. It does not execute Pinocchio,
OpenSim, MyoSuite, or Simscape. These rows remain required and unqualified.
Actual native marker forward kinematics and anatomical attachment mapping
belong to a following F09 child. Do not interpret `qpos`, `qvel`, or native
integration-state arrays as marker positions.

Tests use no private capture, subject, or mocap data. No tolerances or
physiological claims are introduced, and the calculation registry stays
blocked.
