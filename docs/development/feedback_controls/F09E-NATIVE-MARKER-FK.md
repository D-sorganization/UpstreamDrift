# F09e Pinocchio Native Marker Forward Kinematics

Issue: [UpstreamDrift #11914](https://github.com/D-sorganization/UpstreamDrift/issues/11914)
Parent: [F09 #11793](https://github.com/D-sorganization/UpstreamDrift/issues/11793)
Dependencies: [F09d #11907](https://github.com/D-sorganization/UpstreamDrift/issues/11907),
[Pinocchio native replay #11906](https://github.com/D-sorganization/UpstreamDrift/pull/11906)

## Implemented Boundary

F09e adds Pinocchio as an actual native consumer of the versioned F09 marker
output path. It invokes the existing `replay_native_pinocchio_torque_bundle`
once, then evaluates explicit native frame-local points at each returned
configuration by `PinocchioPhysicsEngine`. It reuses the provider implementation
from #11906 through merge ancestry; no adapter source snapshot is copied.

The F01 inventory row remains separate from T01 native bundle identity and the
F06 execution provider. The `NativeAdapterBinding` cross-checks inventory
package/variant/drive and source/model/provider hashes. The marker map adds
ordered labels, native frame names, finite local offsets, world output frame
and simulation-relative timebase. Receipts retain model/output/map identities
and are explicitly `unqualified`.

Pinocchio state is complete `qpos` and `qvel`, with configuration dimension
`nq` kept distinct from tangent dimension `nv`. Frame placements are derived
from the replayed q; they are not inserted as fictitious state. Fresh-engine
cache initialization is in the native replay policy. `state_reset_allowed` is
false while `state_reset_count` remains unknown (`None`).

The six required engines and every registry row remain visible. A generated
Pinocchio marker trajectory does not qualify its model, satisfy missing
engines, score observations, or establish anatomy, contact, muscle physiology
or cross-engine equivalence. Production Pinocchio model variants still need
their own reviewed inventory binding; the synthetic row is test-only.

## Validation

The actual Pinocchio 4.1 test builds a floating-base `nq=8`, `nv=7` URDF,
sets a nonzero base pose, hinge coordinate and frame-local offset, and compares
native marker output to independently composed root, joint-origin and hinge
transforms. It also verifies separate inventory/native IDs, unknown reset
count, all-row/all-engine retention, and unqualified status.

Run the actual native test with the owned WSL runtime:

```powershell
wsl.exe -d Ubuntu-24.04 -e bash -c 'cd /mnt/c/Users/diete/Repositories/Worktrees/UpstreamDrift-f09e-pinocchio-marker-11914 && /home/dieterolson/.venvs/codex-feedback-pinocchio-11900/bin/python -m pytest -q -c /dev/null --confcutdir=tests/unit/engines tests/unit/engines/test_feedback_native_markers.py -k pinocchio_markers'
```

`-c /dev/null` avoids a repository pytest setting requiring the optional
pytest-asyncio plugin, absent from the owned provider environment. The isolated
run warns about the repo's `unit` mark because repository marker configuration
is not loaded; this is not a provider or numerical warning.

The native Pinocchio test caught an existing `PinocchioPhysicsEngine` issue:
Pinocchio 4.1 frame objects do not expose `.id`. The existing public
`get_link_transforms()` implementation now indexes `data.oMf` with the frame's
enumeration index, matching the native frame container. The F09e native test
exercises that corrected path.

TDD record: the first actual-provider run failed because F09 execution had no
Pinocchio adapter registration. After adding the native output path, it exposed
the `.id` incompatibility; the test passed after the public frame-index fix.
Later regressions also cover a mismatched native model ID and unknown frame.

The canonical design-manual reference is
[`manuals/upstreamdrift/chapters/30-pinocchio-native-marker-forward-kinematics.qmd`](../../../manuals/upstreamdrift/chapters/30-pinocchio-native-marker-forward-kinematics.qmd).
