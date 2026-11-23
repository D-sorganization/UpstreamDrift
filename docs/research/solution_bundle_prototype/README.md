# Solution Bundle Research Prototype

This research-only schema is a partial experiment for #15493, not a Slice 2
delivery or an admitted solution format. The editable calculation-level
[research reference](solution_bundle_prototype.tex) is not the engineering
design manual. No capture or user data is included.

The prototype stores a JSON manifest and a content-addressed NPZ. It embeds
the model bytes, checks their SHA-256, compares named asset hashes and runtime
hash against caller-supplied expected values, and validates state, reference,
gain and spline arrays. The caller must keep the manifest SHA-256 independently
in a trusted location. NPZ bytes are hashed before parsing; object arrays and
duplicate JSON or NPZ members are rejected. A manifest cannot direct a loader
to an arbitrary NPZ path. Hashes detect differences against trusted expected
values; they do not authenticate the author, runtime or assets by themselves.

Saving creates and syncs a content-addressed NPZ, then publishes the complete
manifest using a same-directory hard link. This is atomic for process-level
visibility where hard links are supported; directory crash durability is not
established. An interrupted or competing write may leave an orphan NPZ or
temporary file; an orphan with the same content hash can prevent a repeated
save, and the prototype does not clean it up automatically. The output
directory must be trusted. The loader does not yet
enforce archive resource limits for hostile inputs.

`recorded_playback` copies stored values without running a plant. The separate
`run_mujoco_ideal` entry point uses the existing same-input RK4 integrator and
fixed-gain spline prototype. It accepts fixed equally spaced holds, rigid-shaft,
MuJoCo inputs with explicit total-command clipping, zero root input,
model-default environment and no external assets. State records are sampled
after each integration step; command and ideal input records carry their own
hold clock and are sampled once per hold. The number of holds must divide the
RK4 substep count; closure projection remains once per step. Unsupported policies or a
mismatched measured runtime fingerprint are rejected. The fingerprint covers
only a declared subset of source and binaries, not the full environment; it is
not an exact reproducibility guarantee. The recorded Git SHA is caller-asserted
and is not checked against a clean running checkout. The stored `ideal_input_tau` is the
software-clipped generalized input, not measured or post-actuator torque.
The separate `run_mujoco_ideal_with_trace` also records accepted RK4 stage
states and the ideal drive and external-force components observed in the
native KKT solve. Its trace must complete and is returned in memory; these
stage records are not yet persisted by the research bundle. The two forces
remain ideal generalized inputs and do not establish actuator torque.

`verify_forward_resimulation` compares outputs from a caller-supplied runner
using per-quantity tolerances. Even when passed `run_mujoco_ideal`, its result
cannot authenticate the runner or establish independent verification. The
metadata must label flight and feasibility unavailable. The v4 research schema
rejects older schemas unless explicitly migrated. `migrate_research_v2` converts
only already-authenticated, fixed-grid v2 one-hold records; callers must verify
their old manifest and payload before decoding them. Migration retains the old
runtime digest; a runner with changed code rejects that digest unless new
evidence is independently generated. `migrate_research_v3` accepts only
caller-authenticated no-impact research records. Full runtime locking, asset
embedding and root-detected impact replay, downstream applied-actuator evidence, physical gates, flight
metrics and independently replayed optimization results remain unimplemented.
The code must not be used to claim Slice 2 is complete or to qualify a swing.

The v4 prototype can replay a recorded, already-latched constant world force
on a specified model frame. Its start and end must lie inside the run without
ambiguous near-hold-edge timing. The actual pre-force q and qd are observed at
the stepper's split, including off-grid events, and compared with the recorded
event state. This is prescribed-force replay: it does not search for, verify or
certify on-face contact, a ball collision, or first contact. Metadata must set
`contact_verified` false. The force is external to the ideal generalized
controller input and is not recorded as actuator torque.
This research branch uses fixed held-step lattice boundaries on both sides of
the recorded force window, including an off-grid split. It does not search for
contact or bound integration error, and must not be used to certify or optimize
first-contact timing. Main's impact path has not been changed by this research.

Focused validation: `python -m pytest -q
tests/unit/motion_matching/same_input/test_solution_bundle_prototype.py`.
