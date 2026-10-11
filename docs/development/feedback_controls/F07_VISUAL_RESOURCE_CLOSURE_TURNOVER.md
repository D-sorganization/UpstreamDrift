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

PR #12232 remains in review on `fix/12231-source-bound-visual-assets`.
The current integration includes a normal merge of main and regenerated
agent-context views. OpenSim 4.6 regression validation passed 36 tests with
zero skips across `test_native_muscle_bundle.py` and
`tests/unit/engines/test_native_replay_contracts.py`; the ledger unit suite
passed 11 tests. All five central pre-PR gates passed before the final receipt
formatter exclusion.

A normal pre-push hook exposed a formatter rewrite of the existing native
`fd_neck_closed_loop.json` receipt. JSON semantic equality was verified and
its exact committed bytes restored. The explicit Prettier exclusion preserves
its byte-bound provenance and ledger freshness. Normal commit/push hooks and
remote CI must pass on the final head before merge. Issue #12231 remains open
until merge; this contract supplies no full-capture or physical qualification.

## Current Main Reconciliation

Normal merge of main `9e5f87c4dd` preserves the current club-face calibration and fixture bytes. The concurrent visual-branch formatting-only commit was compared as parsed JSON before resolving its three fixture conflicts to current main. Explicit formatter exclusions retain those canonical bytes. Agent-context views were regenerated and checked; all boundary reviews match their source inputs. The pinned OpenSim 4.6 regression again passed 36 tests with zero skips. Chapter 49 is included with its unresolved inventory blocker; approved calculations remain zero and manual release stays blocked. Full-capture muscle matching, independent replay, physical support/grip and all 17 production parity rows remain open.
