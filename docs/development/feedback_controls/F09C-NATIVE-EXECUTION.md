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
reset-enabled policy, non-relative time, and inputs outside the reviewed
adapters. The initial F09c adapters were MuJoCo and Drake; F09e #11914 adds
the published Pinocchio native torque adapter without changing the bundle or
inventory authority. The full-body MuJoCo child adds the exactly bound
`mujoco-full-body-generalized-rk4` provider for coordinate-ordered
`GENERALIZED_EFFORT` histories. It keeps complete q/v as its state instead of
inheriting the direct-torque adapter's MuJoCo integration-cache schema. Other
engines and muscle excitation remain represented and blocking.

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

Native output validation is split into bounded state-validation and receipt
construction helpers so the public validator stays within the repository
function budget. This structural refactor preserves the exact-state, input,
policy, and horizon checks. Regression coverage mutates the first native
numerical integration state and confirms that receipt construction is rejected.

## TDD and Validation

RED: `python -m pytest -q tests/unit/engines/test_feedback_native_execution.py`
failed during collection because the strict F09c module did not exist.

GREEN: the focused suite passes using independent synthetic bundle fixtures
and an independently generated one-hinge MJCF executed through the actual
MuJoCo native adapter. The fixture tests identity mismatch, tampered applied
input, tampered initial numerical state, private model-path omission,
unqualified receipts, and missing rows.
The native smoke exercises adapter wiring only; it is not production-model or
physics qualification.

The design note is in
`manuals/upstreamdrift/chapters/13-feedback-comparison.qmd`; the API contract
is summarized in `SPEC.md`. Full repo gates and generated-manual governance
are recorded in the PR after the stacked F01c and native-adapter dependencies
are reconciled.

## Limitations and Next Handoff

T01 v1 currently has reviewed direct torque paths for MuJoCo, Drake, and
Pinocchio plus the specifically bound full-body MuJoCo generalized-effort
consumer. It does not yet admit excitation, external-force, or
shared-rigid-body-emulation rows. Required model rows remain unqualified.
Actual native marker forward kinematics and anatomical attachment mapping
belong to a following F09 child. Do not interpret `qpos`, `qvel`, or native
integration-state arrays as marker positions.

## F09c Full-Body Generalized-Effort Consumer

`src/engines/physics_engines/mujoco/python/native_full_body_replay.py` builds
and replays a T01 bundle against the exact full-body spec. Input channels cover
the source coordinate order, with zero entries for all six root coordinates;
the consumer rejects nonzero root efforts before the shared integrator can
silently zero them. The existing `VectorPlant` evaluates its source-owned
contact and dual-grip weld dynamics at every derivative call in the shared
classical RK4 integrator. The provider uses eight fixed substeps per frozen
interval and disables closure projection. It does not call `mj_step`, assemble
or reset the seed, restore hidden state, or accept a feedback callback.

The output carries the q/v trajectory, applied generalized efforts, initial
pose/rate residual vectors, and the shared integrator's per-step maximum pose-
residual magnitudes. Those observations have no new acceptance threshold; an
off-manifold seed can be replayed diagnostically but is not
promoted as source-consistent or physically qualified. The q/v-only schema is
accepted only for this exact provider ID. Tests use mock-plant and synthetic
T01 contract fixtures; no private driver/iron capture or production native
full-body run was performed. This is an open-loop frozen-history consumer, not
a feedback producer or evidence that a feedback producer made the history.

The paired `record_native_full_body_feedback` producer records a Python
state-dependent controller once at each fixed interval start. It passes
detached copies of complete q/v, validates a finite full-coordinate effort and
exact zero on the six root entries before the shared integrator, and holds the
validated vector over eight RK4 substeps (32 acceleration-stage calls) before
the next controller call. The producer returns the ordinary T01 frozen-input
bundle and a separate typed provenance/trajectory record; fresh replay never
calls the controller. Controller source/configuration/reference digests are
caller-declared provenance. The recorder declares no generalized-effort
bounds, does not project/reset the initial state, and applies no closure
acceptance threshold. Manufactured-plant tests verify this software boundary
only; no native full-body execution or physical qualification follows.

Tests use no private capture, subject, or mocap data. No tolerances or
physiological claims are introduced, and the calculation registry stays
blocked.
