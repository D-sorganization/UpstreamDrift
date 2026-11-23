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
fixed-gain spline prototype. It accepts only one-hold-per-step, rigid-shaft,
no-impact MuJoCo inputs with explicit total-command clipping, zero root input,
model-default environment and no external assets. Unsupported policies or a
mismatched measured runtime fingerprint are rejected. The fingerprint covers
only a declared subset of source and binaries, not the full environment; it is
not an exact reproducibility guarantee. The recorded Git SHA is caller-asserted
and is not checked against a clean running checkout. The stored `ideal_input_tau` is the
software-clipped generalized input, not measured or post-actuator torque.

`verify_forward_resimulation` compares outputs from a caller-supplied runner
using per-quantity tolerances. Even when passed `run_mujoco_ideal`, its result
cannot authenticate the runner or establish independent verification. The
metadata must label flight and feasibility unavailable. The v2 research schema
explicitly rejects v1 pending an actual migration; it does not silently
reinterpret old data. Full runtime locking, asset embedding, multi-hold and
impact replay, downstream applied-actuator evidence, physical gates, flight
metrics and independently replayed optimization results remain unimplemented.
The code must not be used to claim Slice 2 is complete or to qualify a swing.

Focused validation: `python -m pytest -q
tests/unit/motion_matching/same_input/test_solution_bundle_prototype.py`.
