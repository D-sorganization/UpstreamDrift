# F09d Native Marker Forward Kinematics

Issue #11907 adds an observation bridge from a validated F09c replay trajectory
to the existing positions-only output consumed by F09b. It does not create a
second replay authority, marker metric, scoring gate, or model inventory.

`execute_native_marker_replay` executes a frozen bundle through
`execute_native_replay_with_output`, validates an ordered `NativeMarkerMap`
against the exact T01/native adapter binding, and delegates FK to the same
native engine adapter that owns the model loader. Each attachment names one
native body/frame and one finite body-local point in metres. The map SHA binds
the ordered labels, frames, offsets, output world frame, timebase, source and
loaded-model digests, and native adapter provider identity.

The MuJoCo consumer restores each actual replay `qpos` into a fresh native data
object and uses `mj_forward` plus native body transforms. The Drake consumer
loads the exact bound plant, sets each actual replay configuration in a native
context, and queries native frame point positions in world. Both require a
strictly increasing complete simulation-relative output grid, exact loaded
model identity, known frame names, unique ordered labels, and finite arrays.
They never step a second simulation or inspect measured observations. F09c
remains the sole input/replay authority.

`NativeMarkerReplayEvidence` includes the native execution receipt and digest,
marker-map digest, native marker position digest and typed
`NativeMarkerPositionOutput`. Its qualification is always `unqualified`; a
caller may preserve `evidence.as_dict()` alongside an existing native
acceptance receipt for F09b's content-addressed record. It excludes raw model
paths, capture identifiers and subject data.

`build_native_marker_replay_report` retains every inventory row and all six
required engines. Missing maps, unsupported providers, unavailable runtimes,
and rejected identities remain visible. Its coverage property means only that
every required row has a generated FK output; it cannot promote F01
qualification or imply observational or physiological validity.

Tests include an independent MuJoCo floating-base and hinge transform fixture,
a Drake `nq != nv` floating-base fixture, a nonzero body-local offset, changed
mapping digest, stale model rejection, and unknown native frame rejection.
The tests establish only the FK software boundary. Current six-engine
availability remains unchanged: MuJoCo and Drake expose this torque-only
adapter path; OpenSim waits for its separate F07 geometry/full-state seam, and
Pinocchio, MyoSuite and Simscape remain required but unimplemented here. No
anatomical mapping, tolerance, physiology, contact, or cross-engine claim is
introduced.

The canonical design note is
`manuals/upstreamdrift/chapters/28-native-marker-forward-kinematics.qmd`.

## Updated F09c Parent Contract

The branch now includes the F09c parent update by a real merge. Native replay
output validation keeps bounded, separate state-validation and receipt
construction helpers, preserving the exact numerical-state, input, policy,
model, and horizon checks. The Tools mocap seam is extended only when the
bundle type is consumed, so importing observation qualification does not alter
the independent capture-tool availability probe. The combined focused
replay, observation, capture, and marker suite passes on this merged branch;
two Drake-specific cases skip in the local Windows environment because its
optional `pydrake` bindings are absent.
