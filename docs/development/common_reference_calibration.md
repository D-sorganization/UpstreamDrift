# Common-Reference Calibration Implementation

## Scope and Delivery State

Epic #9897; children #9898 and #9900. Branch
`feat/9898-common-reference-sessions` in the isolated
`UpstreamDrift-common-calibration` worktree, base3482a5a39. This branch is a reviewed implementation checkpoint with no PR yet. It is reachable through Calibration in this worktree,
but the running test application still uses the earlier merged wizard source.

Tools #5140 merged as0a561daff18302ee143b214fe6a6138455d8c542. Its complete tree
is identical to qualified branchc98402cb1 and includes numerical #5136, which was
closed as redundant. The local vendor checkout uses0a561daff for development.
Do not commit this pin alone: the impact owner verified it would regress the
function-generator launcher. Main's interim pin remains4dabe900c. The context
owner owns the final combined gitlink/Cargo/pip/catalog alignment after Tools
#5144, then the impact owner qualifies the exact installed consumer. Historical
private-checkout404 and rate-shard timeouts remain failures, not passing evidence.

## Implemented Workflow

- Capture-owned immutable reference sessions reuse canonical Tools US Letter,
  A4, yardstick and metre-stick geometry. Measured dimensions create distinct
  targets. Camera selections reuse existing lens profiles, declared optical
  settings and physical identity. Moving a camera or changing declared optics
  disables further marking/solving until a new session is created.
- Actual original video frames are decoded outside the UI process and retained
  as lossless PNGs with bounded metadata and SHA-256. Existing trim/crop selections
  never change calibration coordinates. Repeated observations retain physical
  point IDs, placement names, notes, inclusion and held-out status.
- Standard Qt controls, ImageCanvas, QUndoStack and RigProcessRunner provide
  point editing, keyboard entry, undo/redo, worker progress, cancellation and
  actionable errors. A separate loader verifies/decompresses archived frames
  without blocking Qt. Saved capture profiles are restored without retaining
  an unrelated previous combo-box selection.
- Canonical Tools fixed-intrinsics estimation consumes reviewed lens settings,
  explicit world-anchor translation/orientation and identified observations.
  The adapter converts camera-from-world into existing world-from-camera records
  once, preserving all lens coefficients. Camera/placement residuals and Tools
  limitations are shown before explicit operator acceptance.
- Saved estimates can be reopened against their matching observation revision.
  Acceptance rechecks frames, profiles, anchor, revision hash and the exact result
  digest shown to the player. Reviewed copies do not overwrite numerical results.
  Source evidence hashes provider/adapter files and records numerical runtime
  versions. It is source provenance, not a full SBOM or physical approval.
- Reconstruction corrects original distorted detections into ideal pixels before
  layout averaging and fitting, with a calibration signature preventing repeated
  correction. Camera records retain distortion for downstream video overlays.
  OpenCV supplies the numerical operation; no new lens or camera solver is copied.
- Native Help and the generated operator guide share `guidance.py`. Regenerate
  with `python3 scripts/generate_reference_calibration_guide.py`; `--check` detects
  drift. The calibration inventory remains explicitly blocked under #9900 in the
  engineering manual registry.

## Provider Boundary and Responsiveness

UpstreamDrift owns its local Sidekick cluster. The isolated worker launches with
`python -I`, resolves the selected Tools family first and leaves the parent's
namespace unchanged. Do not remove the fallback guards or add global aliases.
This checkout-based worker still requires exact packaged-runtime qualification.

Import tracing found calibration bookkeeping loading reconstruction, pose and
physics modules. IntrinsicsRecord and swing-export rendering imports are now
lazy. Separate validation samples changed from about2.85s to1.26s; catalog samples
varied0.39–0.97s under shared host load. These are diagnostic observations, not
controlled speedup benchmarks. Native OpenCV/SciPy already run the numerical work;
there is no evidence here supporting a Rust rewrite as the next improvement.

## Validation Checkpoint

The broad Capture Rig and reconstruction run completed successfully:594 tests
on Python3.13, including export, wizard, body-model and reference cases. Log:
`TEMP/capture-reference-integrated-regression.log`. Later history/Qt rerun7 passed,
and projection/Qt refinement25 passed. The four new distortion overlay cases
first failed against ideal projection and then passed after the shared adapter
fix. Actual OpenCV5.0.0 passed20 lens/reference-projection tests, including
5/8/12/14-term models. Atlas/parity49 passed. These are synthetic/software results.

Full Ruff and format pass (6941 files), mypy28 source modules plus20 projection
modules pass, and LoD reports no growth across3050 source files. The actual
working-tree architecture check covered41 changed Python paths and passed.
Normal pre-commit hooks passed before the most recent projection/performance
refinements; repeat on the final staged set. Generated help/atlas freshness and
manual governance pass; the calculation inventory remains release-blocked.

Native Python3.12 setup, placement, solve and Help pages were inspected at850×720
and640×560. Qt scroll areas and a two-column footer keep actions reachable;
Calibrate Again and Add Another Placement are explicit. Latest native screenshots:
`TEMP/capture-reference-dialog-visual-hn7cyubr`. Point editor was previously
reviewed at640×580. These are synthetic UI fixtures, not physical camera evidence.

The first broader Python3.12 environment lacked declared pydantic-settings and
pytest-qt dependencies; installed them in an isolated temporary environment.
That run ended581 passed,2 failed,12 errors, including a native Help crash and
a180s existing Simscape real-log calculation timeout. The numeric timeout was
reproduced with a stack in existing model/simscape.py and SciPy Rotation; no
calibration code was involved. Do not relabel this broad run as passing.

Subsequent scoped checks identified an excessively wide footer, now corrected,
and repeated PNG decompression during evidence checks. Saved-evidence validation
now rechecks capture identity, dimensions and original byte hashes without
re-decoding an already-archived frame; actual point inspection still decodes it.
Measured-target operations also defer player/media imports until actual frame
extraction. This removes work rather than increasing test timeouts. Production
Python3.12 scoped rerun passes51 tests in37.11s, including real video, revision
history, the isolated numerical worker, standard Help navigation and distortion
projection. Log: `TEMP/capture-reference-py312-lazy.log`. The fixture timeout
remains20s; no test limit was raised to obtain this pass.

## Limits and Remaining Acceptance

- Profiles and settings are operator declarations. Unreported camera/ring movement
  cannot be detected. Paper does not replace lens calibration. Ruler endpoints
  alone cannot establish camera poses. A wholly held-out placement cannot seed
  its own transform. Pixel residuals do not certify metric physical accuracy.
- Sessions/results remain capture-owned. Polished guided transfer of reviewed
  camera geometry to a new swing capture is still required for repeat practice.
  Calibration and swing recording orchestration needs review before epic closure.
- Original-frame selection currently uses a numbered field and point editor;
  a standard scrubber/thumbnail selection flow remains a UX follow-up in #9897.
- Saved revisions assume a single dialog writer. Cancelling marking may retain
  unused archived frames. Do not delete evidence automatically. Referenced lens
  files must stay available; portable profile packaging remains to be qualified.
- #9900 manual calculation inventory, physical multi-camera/zoom evidence and
  packaged-runtime qualification are not approved by synthetic tests.
- Final provider pin, current-main integration, accurate parity/capability metadata,
  complete operator screenshots, commit/PR/CI/merge are still pending.

## Concurrent Work and Live Applications

Session `capture-product-01a08427-common-calibration` owns the reference package,
calibration actions/profiles/evidence, scoped lazy swing-export imports and
reconstruction lens boundary. Presence and #9898 lease renewed through05:51UTC
on2026-09-10. #9900 lease through05:25UTC. Impact sequencing message35dd44d5 was
acknowledged; there are no active scope conflicts in the last complete inbox.

Preserve live app PID54812 (`Capture Rig — main 56552f245`), older app61500 and
atlas2963. They serve frozen earlier checkouts, not this branch. The analysis
owner controls comparison/coaching/model work (#9932/#9942); do not edit it.
The context owner controls #9915/catalog/provider alignment. Continue only in this
isolated tree and merge normally after compatible provider qualification.
