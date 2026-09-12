# Native Drake Matching Handoff

## Status and Ownership

Issue #10022, under epic #9921; branch `feat/10022-native-drake-equivalence`, based on `842756bd3`. Worktree: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-drake-native-10022`. All qualification jobs are terminal. No native matching fit or MATLAB job was started by this lane. MATLAB R2025b is the required release.

The native adapter is implemented and qualified through the 0.8-second baseline. This is **custom rigid constrained Drake execution**: actual Drake URDF tree mass, gravity, bias, spatial Jacobian and bias acceleration enter an exact six-dimensional KKT solve. Shared native replay supplies continuous DOP853 integration and the original world-force to primitive effort mapping. It is not stock Drake discrete SAP simulation. The existing generic golfer simulator remains a separate model and must not inherit these qualification claims.

The full 1.8138888889-second capture is not matched. Pinocchio remains the faster optimizer in these measurements: about 3.23 seconds versus Drake 11.08 seconds per baseline replay. Use Drake as an independent candidate verifier; do not start another expensive optimizer merely because this engine is now available.

## Qualification Evidence

- `evidence/inventory.json`: 27 scalar coordinates; all 16 frame identities; all 31 physical solids. Total mass 77.60581783574679 kg; individual masses and local COM match exactly, COM inertia error at most 2.78e-17. Whole-model world COM differs from fresh Pinocchio by at most 1.11e-15 m at six states. All damping zero; restored position and velocity limits unbounded; gravity [0, 0, -9.80665].
- `evidence/pinocchio-parity.json`: zero plus 27 unit primitive efforts at six recorded moving baseline states. Maximum acceleration difference 5.07e-10, scaled difference 1.46e-11; frame transform difference 9.35e-16. Continuous 0.8-second replay gives q difference 4.55e-8, rate difference 9.52e-7, marker-coordinate difference 7.70e-9 m; closure pose/rate 4.61e-12 / 7.37e-11.
- `evidence/r2025b-parity.json`: direct recorded R2025b states/primitive efforts give acceleration difference 2.90e-9, scaled 3.08e-10; all frame transforms differ at most 8.99e-15. Same-input continuous replay gives q difference 4.10e-6, rate difference 1.29e-4, frame-origin position difference 1.00e-6 m; closure pose/rate 3.12e-12 / 6.03e-11. This direct continuous comparison samples six actual recorded times; it is not an all-internal-sample certificate.
- `evidence/raw-evidence.zip`: exact executed runtime sources, frozen environment, original runners, inputs, reference-specific candidate, raw JSON and NPZ trajectories, and per-file SHA256 inventory. Adjacent JSON may be formatted by hooks; use raw ZIP identities for exact reproduction.

The original candidate `2afaf8b21a05a44b071e7328e2d624bba5f6a999aa85920b53b61b43961e6673` is used for fresh Pinocchio parity. R2025b reference coefficients differ. Its own candidate `39e77b31af411463018ee9ac3cb9cb0d6d29984355b95413f63245d3575659ec` is preserved in `evidence/r2025b-candidate.json` and the ZIP; never compare these two trajectories as identical input.

Native model raw SHA256: `b817fea76407f71b29e42aeb188890df53d2d86874c7e5875d14d353f4a1a248`. R2025b source raw SHA256: `f69eb1a6ad54a72666a01fdf84a21009bb56eb74510fcbdce39b4dbf1c69fad5`. The large R2025b reference remains in the parent raw checkpoint archive and ControlTower at `C:/Users/diete/native-tight-reference-9967-02.json`; it is not duplicated in this ZIP.

## Runtime and Exact Commands

ControlTower SSH alias `controltower`; WSL distribution `ControlTower-Runner`. Isolated Python environment `/home/dieterolson/drake-native-10022`, Drake 1.57.0, Python 3.12, NumPy 2.5.3, SciPy 1.18.1. Runtime `/home/dieterolson/drake-native-runtime-10022-02` is immutable after qualification. It uses namespace packages and does not establish full desktop application import/deployment qualification. Existing Pinocchio runtime14 and environment were not modified.

Prefix commands with `ssh -o BatchMode=yes controltower wsl -d ControlTower-Runner --` from Windows. In WSL:

```bash
export PYTHONPATH=/home/dieterolson/drake-native-runtime-10022-02
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/home/dieterolson/drake-native-10022/mplcache
/home/dieterolson/drake-native-10022/bin/python /mnt/c/Users/diete/qualify_drake_native_10022_04.py --model /mnt/c/Users/diete/native_geometry_spec_9967.json --candidate /mnt/c/Users/diete/native-root-force-9967-02/returned-candidate.json --urdf /mnt/c/Users/diete/native-golf-9967-01.urdf --sidecar /mnt/c/Users/diete/native-golf-9967-01.sidecar.json --reference /mnt/c/Users/diete/drake-native-reference-10022-01 --output /mnt/c/Users/diete/NEW-UNUSED-OUTPUT
/home/dieterolson/drake-native-10022/bin/python /mnt/c/Users/diete/qualify_drake_r2025b_10022.py --assets /mnt/c/Users/diete --reference /mnt/c/Users/diete/native-tight-reference-9967-02.json --output /mnt/c/Users/diete/ANOTHER-UNUSED-OUTPUT
DRAKE_NATIVE_ASSETS=/mnt/c/Users/diete /home/dieterolson/drake-native-10022/bin/python -m pytest /mnt/c/Users/diete/test_drake_native_live_10022.py -q
```

The checked-in `reproduction/` sources preserve the final runners. `stage_runtime_02.py` and `archive_evidence.py` describe the original fixed experiment paths and intentionally refuse overwriting; adapt output names when preparing a new immutable experiment. Prefer extracting the complete archived runtime to a new directory over copying a changed live checkout. Verify every SHA256 against the manifest before reusing claims.

## Implementation and Tests

`src/engines/physics_engines/drake/python/native_model.py` implements `NativeDrakeModel`, satisfying the existing `NativeReplayEngine` protocol through `accelerations`, `frame_poses` and `closure_errors`. Instantiate with exact URDF, sidecar and native model bytes; supply a factory to the existing `replay_candidate` or `replay_window`. No shared optimizer fork is needed. No analytic Drake trajectory sensitivity API has been added or qualified.

The source validates bundle identity before loading Drake, checks scalar topology and exact state/effort inventories, rejects nonfinite data, retains native coordinate ordering through explicit Drake indices, and returns detached closure diagnostics. The KKT solver rejects singular constraint systems instead of regularizing or silently dropping constraints. Helpers share the engine-independent validator. Its old Pinocchio import remains an identical re-export; extraction commit `3a839341c` is already integrated on root as `d234d6133`.

TDD: five analytic KKT tests first failed on the absent module, then passed. Nine sidecar binding tests include a failing-before-implementation shared-contract identity test. Three actual Drake runtime contract tests pass for mass/inventory/unbounded limits, invalid state maps/nonfinite values, and detached closure diagnostics. Native pulses and replay then independently qualify numerical behavior. Local Ruff and mypy pass. Design-manual inventory check passes while broader manual release remains `blocked-inventory-required`; this is not model scientific approval.

```powershell
python3 -m pytest tests/unit/motion_matching/test_drake_native_model.py tests/unit/motion_matching/test_native_urdf_binding.py -q --no-cov
python3 -m ruff check src/engines/physics_engines/drake/python/native_model.py tests/unit/motion_matching/test_drake_native_model.py tests/unit/motion_matching/test_drake_native_live.py
python3 -m mypy --follow-imports=silent src/engines/physics_engines/drake/python/native_model.py
```

## Sequential Next-Agent Instructions

1. Read this document and the top of the parent Pinocchio checkpoint. Check issue #10022 lease before editing; do not modify another running engine's runtime or environment. Root owns shared solver changes.
2. Verify archived hashes and rerun the three live contracts. Reproduce the two candidate-specific parity checks above before claiming a working installation on another machine.
3. Integrate the adapter commit into the main matching branch. The validator extraction is already there; do not apply it twice. Keep generic golfer and custom native rigid execution clearly distinguishable in any registration or user-facing engine choice.
4. Add a tested engine selector/factory to the shared run configuration only with root coordination. Reuse candidate, effort profile, marker projection and continuous integration; no copied fitter. Record `execution_mode=drake-tree-custom-rigid-kkt` in every result. Demonstrate full installed-package import before adding desktop deployment claims.
5. Replay the next **accepted** Pinocchio fit using identical polynomial coefficients, state, geometry, marker masks and clock. Compare pointwise predictions, not subtraction of C3D RMS scores. If discrepancies arise, isolate static frames/inertia, state-specific accelerations, then integration tolerances in that order.
6. Extend verification to the full capture only after a meaningful candidate exists; keep the last two frames' missing club markers missing. For alternate geometry, degree, start or C3D, issue a new identity and repeat gates. Baseline qualification does not transfer automatically to new model physics.
7. If native sensitivities or a stock Drake simulator are desired, treat them as separate tested capabilities with explicit closure semantics. Do not replace exact rigid constraints by compliant or discrete constraints and preserve the old equivalence label.
8. Save each experiment to an exclusive directory, record runtime/source/environment hashes, archive raw receipts, update this handoff, run focused checks, and commit incrementally. Never promote a full-swing match from this baseline evidence.

## API References

[Drake MultibodyPlant Documentation](https://drake.mit.edu/pydrake/pydrake.multibody.plant.html) defines the mass, gravity, spatial Jacobian and bias acceleration operations used by the adapter. The qualification reports, rather than availability of these APIs, establish the numerical scope claimed here.

The later explicit inertia/COM qualification is preserved separately in evidence/raw-inventory.zip, including both raw reports and its runner.
