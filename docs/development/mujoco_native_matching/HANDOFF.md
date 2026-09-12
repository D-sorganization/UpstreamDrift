# Native MuJoCo Matching Handoff

## Run18 Effort Audit Addendum

[Run18 Effort Correction Audit](RUN18_EFFORT_AUDIT.md) and
`evidence/run18-effort-report.json` independently verify 65 saturated correction
controls across 23 channels. All active entries are ±2, not the eight widened
±10 entries. The report distinguishes world forces, root-base forces and joint
generalized torques, and records actual polynomial extrema and time samples.
Eleven otherwise ±2 torque boxes allow corrections greater than2 after the
0.8-second Bernstein basis interval because run18 covers0.85 seconds. These
are numerical correction boxes, not total physical limits or feasibility proof.
The audit made no runtime or physics changes; the parent owns live run19 and
any subsequent optimization decision. Exact raw inputs/script/report and three
passing extrema tests are retained in `evidence/raw-run18-effort-audit.zip`.

## Native Import and Export Sequence Fixed

The root MuJoCo package now exposes `Engine` lazily. Importing the native
adapter no longer loads the generic humanoid/GUI/model stack as a side effect.
The exact combined pytest collection that previously selected the older vendor
writer now resolves the local writer and exports successfully. Existing
`from ...mujoco import Engine` consumers retain the same cached class export.
No writer, precision setting, vendor tree or shared alias implementation changed.

Two fresh-process regressions exercise the configured pytest import paths,
native-adapter-then-export sequence, local writer identity and a COM value
preserved within 1e-17, plus lazy `Engine` resolution/caching and unknown attribute
failure. Both fail with the previous eager package initializer (including the
original `numeric_precision` TypeError) and pass with the lazy facade. The
combined import, bundle, native live, URDF export and identity suite passes
27 tests. The validated bundle factory and completed qualification outputs are
unchanged.

Scope: this fixes the native import pathway. The Tools alias finder can still
include downstream-owned `src` spellings in its alias list when unrelated
canonical or top-level packages load. A fleet-wide alias ownership correction,
if pursued, belongs in the Tools source and normal dependency-pin workflow;
the change here does not claim to repair every arbitrary generic import order.

## Validated Portable Bundle Factory Qualified

`NativeMujocoModel.from_native_bundle(urdf_bytes, sidecar_bytes, model_bytes)`
now accepts the same three artifacts as the Pinocchio and Drake lanes. The
shared `native_urdf_contract.validate_native_urdf_bundle` rejects mismatched
hashes and closure/frame/coordinate/gravity semantics before engine construction.
The existing byte-based constructor is unchanged. This is **validated canonical
geometry conversion to MJCF, not direct MuJoCo URDF parsing**. The exact supplied
URDF and sidecar hashes are retained in engine metadata.

New immutable run `native-mujoco-bundle-10021-01` passes the same 168 moving pulse
cases and 0.8 s baseline, with unchanged results: scaled acceleration error
3.85410e-10 and continuous marker error 2.84716e-9 m. Receipt and exact output,
source, URDF and sidecar bytes are saved under `evidence/bundle-report.json`
and `evidence/raw-bundle-qualified-run.zip`. Previous qualification artifacts
are untouched. Add `--urdf <raw-root>/native-golf-9967-01.urdf --sidecar
<raw-root>/native-golf-9967-01.sidecar.json` to the Pinocchio-reference command
below to reproduce the factory qualification in a new output directory.

Six new tests first failed on the absent factory; all 12 live tests pass after
implementation. They include valid conversion mass/frame/acceleration identity
and five malformed bundle variants rejected before runtime import. The small
synthetic tests exercise identity/semantic contracts, not URDF parsing; the
new real native run consumes the actual qualified exported URDF and sidecar.

A separate existing package integration problem was encountered while preparing
tests: `export_native_urdf` resolves an older Tools-aliased `URDFWriter` that lacks
`numeric_precision`. No exporter or Tools dependency was changed in this lane.
Consuming the existing qualified bundle works; regenerating that bundle through
the affected full-package exporter needs a separate shared-boundary fix.
This is import-order dependent: a standalone export and isolated pytest probe
pass with the local writer, whereas collecting `test_native_mujoco.py` with that
probe resolves the writer from
`vendor/ud-tools/src/shared/python/model_generation/builders/urdf_writer.py`.
Its module name remains `src.shared.python.model_generation.builders.urdf_writer`
but its signature lacks `numeric_precision`. Exact reproduction from this worktree:

```powershell
python3 -m pytest tests/unit/motion_matching/test_native_mujoco.py `
  C:/Users/diete/Repositories/simscape-tour-checkpoints/test_native_urdf_import_probe_10021.py `
  --noconftest -q -s --tb=short
```

The preserved probe imports `native_urdf`, reports its `URDFWriter` module,
source file and signature using `inspect`, then exports the original native
geometry bytes. Before the lazy-facade fix above, the failure occurred even when
live simulation tests were deselected; collecting the engine test was sufficient.
Preserve this historical reproduction as the regression trigger, not a current
native blocker.

## Scope and Ownership

Issue [#10021](https://github.com/D-sorganization/UpstreamDrift/issues/10021), branch
`feat/10021-native-mujoco`, based on Pinocchio commit `842756bd3`.
Worktree: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-mujoco-native`.
The parent owns shared fitting, sensitivity and replay code. This lane adds only
MuJoCo-specific source, qualification runners, tests and evidence. MATLAB R2025b
is the required release. No existing MATLAB, ControlTower or Pinocchio runtime
was modified. The new runtime executes locally with MuJoCo 3.3.4 and Python 3.13.

## Execution Semantics

`native_mjcf.export_native_mjcf(model_bytes)` creates the native tree from the
same source geometry as the verified Pinocchio model. It preserves 27 scalar
primitives in their declared order, all 31 source solids (zero-mass references
remain zero), 16 marker frames, transforms, gravity and the grip closure.
Physical solids in a rigid body are aggregated using rotated center-of-mass
inertias and the parallel-axis theorem. There are no synthetic helper masses,
joint limits, damping, friction, armature, contacts or tracking feedback.

**The qualified execution mode is `NativeMujocoModel.accelerations`, not stock
`mj_step`.** It obtains M, bias, site Jacobians and their time derivatives from
MuJoCo and solves the rigid acceleration constraint explicitly. The MJCF weld
is useful as a model declaration; stock MuJoCo equality constraints are compliant
and have not been qualified as equivalent. The adapter uses the existing
`native_replay.replay_candidate` model-factory protocol and shared DOP853
integrator. It does not implement a second fitter.

This converter consumes the canonical native geometry artifact. It does not
claim a direct MuJoCo URDF-parser roundtrip; native multi-axis joints are composed
on bodies to avoid assigning nonphysical masses to intermediate URDF links.
URDF plus sidecar remains the fleet interchange source; the validated factory
described above preserves the shared identity contract and distinguishes
conversion from direct parsing.

## Qualification Results

Direct R2025b six moving states through 0.8 s reproduce all 16 frame transforms
within 8.66e-15 and accelerations within 3.51e-9. Same-input continuous replay
gives maximum position-coordinate error 1.33432e-5, rate error 0.01210,
marker component error 1.26345e-6 m, and marker component RMS 5.93755e-8 m.
The marker trajectory reference uses MuJoCo FK at recorded R2025b states;
independent native FK is checked separately at every declared frame at six states.

Independent Pinocchio baseline replay, candidate canonical SHA
`2afaf8b21a05a44b071e7328e2d624bba5f6a999aa85920b53b61b43961e6673`,
passes 168 cases: zero plus 27 unit primitive efforts at six moving states.
Maximum scaled acceleration error is 3.85410e-10, frame error 1.33e-15,
continuous marker error 2.84716e-9 m, q error 1.71e-8 and rate error 7.52e-6.
This qualifies a baseline through 0.8 s. It is not a full-swing fit, proof for
arbitrary geometry, derivative qualification, or stock MuJoCo integration proof.

**Provenance correction:** the tight R2025b reference uses different polynomial
coefficients from the later `native-root-force-9967-02` candidate. The initial
run01 cross-input comparison failed and must never be cited as parity evidence.
Run02 and subsequent runs use an explicitly constructed reference candidate,
with the exact reference coefficients/q0/qd0 and reference source hash. The
runner now rejects mismatched inputs and initial states before continuous
comparison. No target fitting or reference-state injection is used.

## Files and Reproduction

Authoritative raw root: `C:/Users/diete/Repositories/simscape-tour-checkpoints`.
The following inputs already exist there; preserve their exact bytes:

- `gemini-ms-audit-20260912/native_geometry_spec_9967.json`, SHA
  `b817fea76407f71b29e42aeb188890df53d2d86874c7e5875d14d353f4a1a248`.
- `native-input-pulses-9967-01/case-00.json` through `case-27.json`.
- `native-tight-reference-9967-02.json`: R2025b state, effort, frame and dense
  trajectory reference. Also available on ControlTower; its hash is in reports.
- `native-mujoco-r2025b-reference-candidate-10021.json`: reference-specific
  polynomial candidate; its marker attachments come from the baseline candidate.
- `native-root-force-9967-02/returned-candidate.json`: original Pinocchio baseline.
- `mujoco-pin-reference-10021`: independent Pinocchio reference NPZ, receipt and
  original producer runner copied from ControlTower's
  `C:/Users/diete/drake-native-reference-10022-01` and
  `C:/Users/diete/qualify_drake_native_10022.py`. Coordinate and source identities
  are checked by the consumer, which hashes the actual reference bytes.

Run from this worktree with `PYTHONPATH` set to the worktree and
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`:

```powershell
python3 docs/development/mujoco_native_matching/qualify_native_mujoco.py `
  --model <raw-root>/gemini-ms-audit-20260912/native_geometry_spec_9967.json `
  --candidate <raw-root>/native-mujoco-r2025b-reference-candidate-10021.json `
  --pulses <raw-root>/native-input-pulses-9967-01 `
  --reference <raw-root>/native-tight-reference-9967-02.json --output <NEW-directory>
python3 docs/development/mujoco_native_matching/qualify_pinocchio_reference.py `
  --model <raw-root>/gemini-ms-audit-20260912/native_geometry_spec_9967.json `
  --candidate <raw-root>/native-root-force-9967-02/returned-candidate.json `
  --reference <raw-root>/mujoco-pin-reference-10021 --output <NEW-directory>
```

Outputs are exclusive; never overwrite an existing run. Reports bind source,
model, candidate and reference hashes. Read `passed` and individual metrics;
process exit zero alone does not mean parity passed. Adjacent repository JSON
may be formatted by hooks; raw ZIPs preserve original bytes for hash checks.

## Engineering Validation and Next Assignments

The first feature test collection was blocked by an uninitialized pinned Tools
submodule, then initialized using `git submodule update --init vendor/ud-tools`.
Isolated and full-package tests subsequently passed. Zero-mass reference support
was added after its regression test failed with the original strict positive
mass check. Six live tests cover force sharing, composed rotation order,
source identity, geometry/state contracts, zero-mass references, and centripetal
acceleration of a welded rotating pair (exercising Jdot). They are marked
`live_simulation` and `requires_mujoco`; opt in explicitly. No derivative fitter
is implemented or implicitly claimed by these tests.

1. Review source and retained qualification receipts; re-run from a clean runtime
   if importing/cherry-picking this lane. Direct mypy and Ruff must pass.
2. Integrate the qualified adapter through the shared model-factory protocol;
   preserve the candidate's source/model/capture hashes and world-force mapping.
   Do not route matching through the older generic MuJoCo golfer model.
3. Use the qualified bundle factory for portable model ingestion. Do not call
   conversion of native geometry a direct URDF-parser roundtrip. Keep malformed
   identity and semantic rejection tests green when extending the shared contract.
4. Qualify acceleration sensitivities before any analytic-gradient fitting. Keep
   the parent-owned solver and six-degree polynomial layout. Evaluate current
   fitted candidates independently and retain missing-marker masks.
5. Extend parity beyond 0.8 s only with matching-input independent reference
   coverage. Full C3D coverage remains 1.8138888889 s; no full swing is matched.
6. Update this handoff and issue after each controlled experiment, record live
   process handles and immutable output paths, and commit incrementally.
