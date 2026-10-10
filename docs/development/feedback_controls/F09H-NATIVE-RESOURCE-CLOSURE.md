# F09h Direct Native Model Replay and Resource Closure

Issue: [UpstreamDrift #11950](https://github.com/D-sorganization/UpstreamDrift/issues/11950)
References: [F09g preflight #11945](https://github.com/D-sorganization/UpstreamDrift/issues/11945), [F09 #11793](https://github.com/D-sorganization/UpstreamDrift/issues/11793)

## Purpose and Contract

F09h supplies a bounded direct MuJoCo replay provider for MyoSuite-sourced
muscle models. It consumes the existing Tools T01 `ACTUATOR_COMMAND` bundle;
it adds no transport schema, model inventory, qualification score, or second
integrator. The provider is explicitly an underlying MuJoCo route. It does
not claim to execute the official MyoSuite `MujocoEnv` wrapper or establish
official SDK parity.

The adapter restores the complete saved MuJoCo integration state and replays
the already frozen, ordered command samples with native `mj_step`. Its
registration binds the model source, compiled model, provider/runtime,
actuator channel order and compiled actuator laws, contact settings, solver,
integration method, timestep, state schema, and execution policy. It rejects
non-MuJoCo engine identities before the factory runs, global callbacks before
resource discovery or construction, wrappers/plugins, external-load payloads,
state-feedback/reset policies, command readback differences, warnings,
nonfinite output, off-clock steps, and non-unit free/ball quaternions.

## Bounded Source and Resource Admission

Before loading the provider model, admission resolves disk XML includes using
the verified MuJoCo 3.8.0 priority: the entry model's directory first, then
the including file's directory when the root candidate is absent. Hardened
XML inspection rejects absolute, URI, drive, UNC, symlink-escape and
unsupported file references before native parsing. Native `MjSpec` supplies
the reviewed mesh, texture, compiler-directory and cube-face metadata. The
current subset accepts XML includes, STL/MSH meshes, and PNG textures only.
Every discovered source and asset must remain under the declared resource
root and match its exact caller-supplied SHA-256. Superset inventories remain
valid. Resource hashes are rechecked after preflight compilation and factory
construction, and the preflight compiled MJB must match the registered loaded
model digest.

This is a version-bound disk-MJCF subset, not a general MJCF parser or
closed-world security claim. VFS/custom assets, URDF, `strippath`, hfields,
skins, attach/flexcomp, plugins/extensions, `content_type` overrides and
unreviewed file-bearing loaders are refused. Other MuJoCo versions or loader
surfaces fail closed pending separate inspection and tests.

## Verification and Limits

The focused native suite exercises synthetic include resolution, root-versus-
nested include ambiguity, missing and outside-root sources, URI and symlink
references, undeclared and altered mesh/texture/cube-face resources, compiler
directories, unsupported loaders/formats, callback ordering, and native
identity checks. Under the retained MuJoCo 3.8.0 environment, public driver
and iron replay matches all 31 frozen state samples for each model; the
discovered closure is 49 files per model and matches the supplied exact-hash
inventory. The pinned MyoSim arm fixture also replays from its full integration
state. These are underlying MuJoCo results and do not establish official
MyoSuite 3.6 execution, an admissible production initial condition, contact or
grip acceptance, physiology, capture validity, or six-engine equivalence.
Required model/engine rows and all unavailable or unqualified evidence remain
visible in the F09 denominator.

The source default of the public arm fixture includes adverse penetration and
large contact force. Its exploratory finite-difference analysis found the
critical derivative issue at a joint-limit boundary; it did not tune the
model or qualify a controller. Exact source and receipt are preserved in
[`F09G-MYOARM-DIAGNOSTIC.md`](F09G-MYOARM-DIAGNOSTIC.md) and
[`F09G-MYOARM-FD-AUDIT-MJ38.json`](F09G-MYOARM-FD-AUDIT-MJ38.json).
