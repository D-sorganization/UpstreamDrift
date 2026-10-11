# F07 Visual Resource Closure Turnover

## Contract

Issue #12231 adds a visual-only resource closure to the native OpenSim muscle
bundle adapter. Resource-free models continue to return the unchanged Tools
T01 `ExperimentReplayBundle`. A source containing body-owned `attached_geometry`
Mesh elements uses the explicit
`opensim-native-visual-resource-closure/1` envelope, which nests the unchanged
T01 bundle and binds the source SHA-256 plus each normalized relative `.vtp`
path, native component path, and asset SHA-256. The closure digest covers the
schema version, source digest, and ordered resource descriptors.

The admission boundary parses XML before native loading. It accepts only
`Mesh.mesh_file` references inside a `Body`'s `attached_geometry`, rejects all
other file references, and checks exact resource closure, path containment,
symlinks, counts, and byte bounds. Native `Mesh` components are allowed only
when their native component paths match this source-owned manifest. Replay
copies source and resources into an owned temporary snapshot, rebuilds the
bundle identity before integration, and rechecks the executed snapshot.

This is an asset identity and visual-component integration contract. It does
not authorize wraps, contact, forces, chart dynamics, mechanical inputs, or
full-body OpenSim qualification. No Tools T01 fields or defaults change.

## Verification

Use the dedicated OpenSim 4.6 Python environment and set `CASADIPATH` to its
OpenSim package directory:

```powershell
python -m pytest --noconftest -o addopts= -q tests/opensim/test_native_muscle_bundle.py
```

The native Mesh regression compares states and muscle forces with a matched
resource-free model, exercises bundle serialization and owned replay, and
rejects changed asset bytes. Optional-provider skips outside this environment
are not native acceptance evidence.

## Current Status

Implementation is in review on `fix/12231-source-bound-visual-assets`. The
source and fixture tests are being validated against OpenSim 4.6; repository
hooks and CI remain outstanding. Issue #12231 remains open until merge.
