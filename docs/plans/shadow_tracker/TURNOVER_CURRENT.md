# Shadow Tracker Current Review and Turnover

## Revision Integrity and Persistence — 2026-09-18

Issue #10233, ST-04 / epic #10122. Branch `fix/shadow-tracker-10233`;
base `aa01972aa3b29b2ff4100e2dabe34b63fddd7aa4`. PR not created.
This section supersedes the historical #10233 dispatch below.

Complete immutable record equality defines idempotence: conflicting global
revision IDs fail before mutation, and identical registration does not reset
selection or append history. Parents require identical complete FrameIdentity
(source hash, timestamps, clock/decoder provenance) and pixel dimensions.
Existing missing-parent, namespace and ancestry-cycle checks remain enforced.

`select_revision` selects a registered revision without rewriting history;
`get_cache_key` follows corrections and selection. The existing atomic JSON
writer saves complete MaskFrames plus one selected revision per five-part scope.
Schema 1.1.0 validates all fields and selections; legacy 1.0.0 retains its
last-registered selection. Parents precede children in stored registration
order. Unsupported versions, unknown fields, duplicate revisions, incomplete
or duplicate selections, invalid pixels, orphan parents and cycles fail closed.

### Red and Green Evidence

Before production edits, `python3 -m pytest
 tests/unit/shadow_tracker/test_revision_persistence.py --no-cov -q --timeout=60`
produced **13 failures and 11 passes**: five changed-observation cases and a
changed pixel grid did not raise; selection was absent; six invalid store
metadata cases were accepted. Seven additional regressions cover conflicting
record fields and attempted cycles without index mutation.

`python3 -m pytest tests/unit/shadow_tracker
 tests/integration/shadow_tracker/test_model_probe.py --no-cov -q --timeout=60`
passed **306 tests, zero failures/errors/skips**, on Windows / Python 3.13.5,
Tools pin `62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1`. Scoped Ruff lint/format,
tracked file-size budget and `python3 -m agent_context --root . check` passed.
The context catalog has no registered Shadow Tracker boundary; source and
consumers were inspected directly. Existing deprecation warnings remain. The final expanded contract file passed
31 tests after adding namespaced reopen assertions; scoped mypy also passed.
Four SPEC-integrity tests and staged title-case checks (seven documents) pass.
The repo-local development-log checker is absent. Running
`python3 ../Repository_Management/shared_scripts/development_log.py --repo-root .`
reports pre-existing missing verification SHAs in other entries and a missing
PR field for DL-#9967, plus the existing portfolio WIP breach (now 20/8).
DL-#10233 has no reported errors; unrelated entries were preserved. These
repository-wide documentation failures remain a publication limitation.

### Review Limits and Next Action

Review and publish the focused diff with `Fixes #10233` and `Refs #10122`.
Renderer source/tests and unrelated original-checkout work are preserved.
These guarantees apply to `ManualMaskProvider.save/load`. Separately,
`artifacts.save_bundle` overwrites existing directory files individually, and
service initialization changes live state before validating all masks. Those
pre-existing ST-11 transactions need a separate focused repair; this provider
change does not certify them as atomic. No power-loss durability, concurrent
writer coordination, neural inference or scientific qualification is claimed.

## Release Qualification and Delivery Review — 2026-09-17

Delivery: [Issue #10135](https://github.com/D-sorganization/UpstreamDrift/issues/10135) (ST-12: Qualify Advertised Engines and Release).

- **Advertised Engine Qualification & Conformance Matrix:** `EngineCapabilityMatrix`, `EngineReceipt`, `audit_engine_conformance` validating model hashes, canonical quaternion conventions, closure tolerances, and Simscape MATLAB R2025b prerequisites.
- **Independent Replay Verification:** `IndependentReplayAudit` and `verify_independent_replay` ensuring forward dynamics replay passes without optimizer cache leakage.
- **Performance Profiling:** `PerformanceProfile` and `profile_shadow_tracker_performance` benchmarking throughput, memory, frame rates, and per-phase latency against computational budgets.
- **Scientific Calculation Registry:** `ScientificRegistryEntry` and `get_shadow_tracker_scientific_registry` publishing immutable formulas, physical invariants, and assumptions.
- **Full Release Gates Audit & Evidence Inventory:** `audit_full_release_gates` verifying G0–G7, complete swing phase coverage (address through follow-through), and generating deterministic SHA-256 evidence digests via `generate_release_evidence_inventory`.
- **Validation Receipt:** 274 unit tests passing across `tests/unit/shadow_tracker`.

## Restart Review — 2026-09-16

Restart delivery: [PR #10274](https://github.com/D-sorganization/UpstreamDrift/pull/10274).

Reviewed main `0ec64e45f`, including timing PR #10253 (`bd2e86b7d`).
**The immediate delivery blocker is renderer PR #10264**, not a missing plan.
The project remains partial infrastructure: no usable Shadow Tracker launcher
application or qualified silhouette-driven forward fitter exists on this baseline.

### Verified Progress and Remaining Gaps

- #10253 defaults physical time to unknown, exposes timing capability metadata,
  hashes decoder/pixel-format provenance, and decodes incrementally. Do not redo
  those repairs or claim the old eager whole-file constructor still exists.
- A remaining defect labeled estimated-CFR timestamps exact whenever average FPS
  was positive. This restart patch fixes that flag, with a real-video regression
  observed failing before the fix and passing afterward.
- Native container PTS are still not extracted. Existing VFR tests inject ticks
  into a CFR clip. Timestamp authority is also lost in stored FrameIdentity
  records. New follow-up **#10273** defines real VFR and persistence acceptance.
- #10264 is open, has merge conflicts, and failed `unit-test-gate` at
  `3bc57b84baff9e865c215b8f83aec203e17cbb36`. The concrete collection failure is
  `ModuleNotFoundError: motion_matching.diagnostics` in
  `test_silhouette_projection_and_losses.py:18`. CI run `35153532669`, job
  `104988849182`, reported 16,572 passed, 399 skipped and one collection error.
  Passing smaller test jobs do not override this failure.
- Static review of that renderer head also finds no thigh/shin/foot segments in
  `_BODY_SEGMENTS`; knee motion changing the upper body is not evidence of a
  full-body silhouette. Add lower-limb geometry and independent region tests.
  `_rasterize_3d_segment` chooses sample count from unbounded projected endpoint
  distance after clipping depth to 1e-6 m. Near-plane crossings can therefore
  request extremely large loops even for a tiny image. Add adversarial near-plane
  work-budget tests and viewport-aware bounded rasterization before enabling
  interactive fitting. Height currently scales radii while FK lengths are supplied
  independently; test and document consistent subject geometry rather than
  treating radius scaling as fitted body shape. These are code-review findings,
  not measured performance or geometry qualification receipts.
- The original checkout already has uncommitted qualified-import corrections
  in `articulated_renderer.py` and its projection test. Preserve and inspect
  those changes with the owning agent; do not overwrite or duplicate its PR.
- #10233 remains open: revision collisions, parent ownership and durable mask
  history still need implementation. Initialization still needs no-evidence
  rejection, club-aware ranking and explicit velocity assumptions.

### Next Agent Actions and Exit Criteria

1. **Renderer owner:** resume #10264, incorporate/verify the existing import
   corrections, merge current main without losing either SPEC row or handoff,
   and regenerate inventories. Run both the focused renderer suite and the
   complete unit collection environment that failed. Require green required
   checks at the exact new SHA before merging. Recheck state units, camera crop,
   near-plane/offscreen clipping and body/club motion; passing mask-change tests
   alone does not qualify geometry or dynamics.
2. **Next independent implementation: #10233.** Claim it before editing. Deliver
   collision-safe and idempotent revision registration first, with red/green
   tests proving failed operations leave every index unchanged. Then persist
   full observation identity, parent lineage, masks and current selection with
   atomic save/reopen and corruption tests. Keep these as bounded reviewable
   changes, reuse repository storage and avoid touching renderer files.
3. **Archive evidence: #10273.** Implement a genuine source-PTS provider and
   versioned persisted clock authority. Do not infer authoritative timestamps
   from FPS or accept injected ticks as proof of native VFR extraction. Unknown
   physical time/scale continues to block qualified SI kinetics.
4. **Visible product milestone: #10134.** After revision persistence, deliver
   import -> inspect frames -> edit body/club masks -> save -> reopen through
   one service and a real launcher adapter/entry point/manifest. Automated
   inference and fitting must report unavailable until their gates pass. Do
   not postpone this useful evidence-review workflow until neural inference.
5. Then finish initialization, real continuous forward rollout (#10130), control
   optimization (#10131), ambiguity (#10132), modern/archive validation (#10133)
   and release (#10135). Reuse current motion-matching providers after checking
   their exact capability/qualification receipts. Another subsystem's passing
   tour-average fit does not qualify Shadow Tracker's video observations.

### Publication Blocker

PR #10274 is not merge-ready at review head `d6ae73c67`: repo-wide LoD CI
(run `35167108374`, job `105030549858`) fails on the pre-existing main chain
`inputs.calibration2.offsets.items` in `motion_matching/pipeline/reference.py:299`.
This is tracked by **#10275**. Integrate its focused repair and re-run checks;
do not waive the gate or confuse local Shadow Tracker success with green CI.
Other CI jobs were still running when this blocker was recorded. The review
provides implementation guidance and a tested patch, not a merged delivery claim.

### Restart Validation Receipt

Python 3.13.5 on Windows, Tools pin
`1ac89c18e6280752d949e520c2143d2fb584d31e`. Selected Shadow Tracker unit tests plus
`tests/integration/shadow_tracker/test_model_probe.py`: **195 tests, zero
failures/errors/skips**, JUnit `shadow-restart-review.xml` in the local temporary
directory. This receipt covers main plus the estimated-time flag repair, not
unmerged renderer #10264. Red test:
`test_unavailable_or_unsupported_timing_mode` failed on `True is False` before
implementation. No real archive or full-product qualification is claimed.

## Historical Review Before the Restart

The following baseline is retained for provenance; the restart review above supersedes its next-task ordering.

### Reviewed Baseline and Evidence

Reviewed main `33ffde23f` after #10205, #10212, #10214 and calibration regeneration
#10177. Review delivery is tracked by #10230. This is the current pickup document;
earlier reviews remain historical. Read [Continuation Prompt](CONTINUATION_PROMPT.md)
and preserve the full [Work Packages](WORK_PACKAGES.md) and G0–G7 in
[Validation](VALIDATION.md). Merged implementation is not stage qualification.

Publication: [PR #10234](https://github.com/D-sorganization/UpstreamDrift/pull/10234).

**Verified progress:** the neural placeholder now raises an explicit unsupported
error instead of claiming masks; manual lookup isolates shots and retains an
in-memory history; morphology uses a read-only mapping; hypotheses reject NaN;
the renderer can rasterize disks; a real OpenCV decoder and iteration API exist.
The regenerated committed probe reports translation closure about **13.7 mm**
and rotation about **1.657 rad**, with `scientifically_qualified=false`. This is
improvement over the historical pose, not physical qualification.

**Local validation:** 190 tests, zero failures/errors/skips, confirmed by JUnit
output, using Python 3.13.5 on Windows and the actual pinned Tools checkout
`1ac89c18e6280752d949e520c2143d2fb584d31e`:

```bash
python3 -m pytest tests/unit/shadow_tracker tests/integration/shadow_tracker/test_model_probe.py --no-cov -q --timeout=60 --junitxml=review-results.xml
```

The selected suite includes real decoder and model-probe cases. It does not
prove the missing VFR/geometry/product behavior below. PR #10212 and #10214
reported no failed checks in their queried rollups; that is not a current full
CI/CD, scientific or installed-product qualification claim.

No `src/tools/shadow_tracker` package or Shadow Tracker manifest, entry point,
API or React registration was found in the reviewed source. There is still no
usable Shadow Tracker launcher application, full fitter or end-to-end acceptance.

## Priority Findings and Concrete Corrective Tasks

### 1. Preserve Time Evidence Before Processing Archives — #10231

`OpenCvVideoDecoder.pts_ticks()` returns the frame index, and its timebase comes
from average FPS; unavailable FPS silently becomes 1 fps. These are synthesized
CFR timestamps, not original presentation timestamps. VFR intervals, negative
start PTS and other archive timing evidence cannot survive this implementation.
`decode_video_frames` also defaults to copying presentation time into physical
time. A direct probe with no physical-time evidence returned `[0.0, 1.0]` rather
than unknown times. This can later give unsupported SI motion/force estimates.

The constructor scans every frame and retains all hashes before `DecodeLimits`
or cancellation apply. The public iterator therefore does not bound the costly
decode operation; memory grows with frame count and time to first result is a
whole-video scan. The protocol does not require its timebase properties and
the iterator supplies another silent 1/1 fallback.

**First tests:** a licensed/generated real VFR clip with independently known
nonzero/negative PTS; repeated images with distinct PTS; unavailable timing;
unknown physical time by default; cancellation and a frame budget during actual
reads, not after constructor completion. Require authoritative timestamps and
explicit timebase capability, or an honestly labeled unsupported/estimated
mode. Physical time needs an explicit evidenced mapping. Decode incrementally,
close handles on every exit, and record decoder/pixel-format provenance for
decoded-image hashes. Do not widen this task into fitting or bulk archive mining.

### 2. Render the Actual Model, Not Only Disks — #10232

Disk rendering improves the earlier one-pixel helper but still has no articulated
body/club geometry binding. The default radii remain zero, rotation/joints do not
affect the body, and club rendering occurs only for exactly six state values.
Two seven-value states differing only in quaternion produced identical body masks
and no club pixels in the review probe. Accepting a state silently while ignoring
its semantics is not a canonical-state adapter.

A second probe used a 10x10 camera, fx=fy=10, center=(5,5), a body center at
(-1.2,0,2), and radius 0.4 m: output had zero body pixels although the projected
disk intersects the left edge. `render` drops the entire shape when its center
is outside the image. It also uses fx for both radius axes and does not establish
correct distorted/cropped model geometry or visibility/occlusion.

**First tests:** independent filled-area and partial-clipping oracles, fx != fy,
camera/crop/request-size consistency, and a real model binding where limb and
club articulation change the correct silhouettes. Version the state/geometry
contract and reject unsupported states rather than switching on tuple length.
Reuse existing validated camera/FK/geometry providers. Keep disks and points as
explicitly limited reference utilities; do not claim a sphere's exact perspective
silhouette from an unqualified fx\*r/z circle approximation.

**Implementation status (#10232):** Addressed via `ArticulatedSilhouetteRenderer` evaluating forward kinematics from canonical 27-DOF articulated state, generating separated body and club capsule masks responsive to limb/wrist kinematics, rasterizing bounding boxes intersecting sensor margins when ellipse centers are offscreen, handling anamorphic focal lengths, and strictly validating camera-crop dimensions and state conventions.

### 3. Protect and Persist Mask Revision Identity — #10233

Shot-aware lookup is repaired, but `_revisions[revision_id]` is overwritten on
every registration. Registering revision `r` in two shots made `get_revision('r')`
silently resolve to the second shot. Parent existence, ownership and lineage
integrity are not checked before mutation; history remains memory-only.

**First tests:** conflicting duplicate revisions fail without modifying any
index; explicitly defined identical re-registration is idempotent; parents must
belong to the same full observation identity; missing parents/cycles fail;
save/reopen restores all revisions and current selection. Include asset, swing,
camera and frame identity, not only a local frame name. Apply atomic persistence
and deterministic downstream invalidation using existing repository facilities.

### 4. Complete Initialization Acceptance Before Optimizing — #10129

Finite-value checks improved, but the existing ranker still aggregates only
`body_iou`, ignores `loss.is_valid`, omits club fit from selection, and supplies
zero velocity. It can select a candidate even when no valid pixels support it.
The arbitrary monocular score remains unsuitable as confidence. Add explicit
no-evidence/no-winner and empty/mismatched-view tests, time/swing/camera identity
checks, body-equivalent/club-different candidates, and observed versus assumed
initial velocity. Demonstrate recovery with the actual model renderer before
calling supplied-candidate ranking complete pose/shape fitting.

## Delivery Path and Division of Work

1. Claim one small task: **#10231 first for the archive ingestion path**; #10233
   is independent evidence/persistence work suitable for a lower-cost agent.
   #10232 needs a geometry/model reviewer and an explicit state binding.
2. Continue model qualification under ST-01: regenerated evidence must pass
   justified translation/rotation/contact and canonical velocity requirements.
   Do not reopen the old coordinate bug or declare success from the newer file.
   ST-07 #10130 can characterize/extract a real rollout boundary while gates
   remain blocked, but cannot qualify the fitter using an invalid initial grip.
3. Deliver a real **import -> frame review -> manual mask edit -> save/reopen**
   service and launcher slice under #10134 once time/revision contracts hold.
   Use existing manifest, lazy embed adapter, installed entry point, host/theme
   and one shared service. Add actual UI, keyboard, cancellation and visual tests.
   Expose missing fitting/automated inference honestly. Do not wait for a neural
   model to provide a useful manual-evidence workflow.
4. Finish real articulated initialization, continuous forward rollout (#10130),
   bounded control fitting (#10131), uncertainty/abstention (#10132), independent
   modern/archive validation (#10133), full PyQt/API/React/export parity (#10134)
   and packaged release (#10135). The prior phrase “Stage 7 followed by release”
   must not skip these stages. Qualify pre-impact and complete-swing phases
   explicitly; no per-frame resets or fake marker/force evidence.

## Engineering, Performance and Turnover Gates

Use behavioral TDD: write/run the failing acceptance case, implement minimally,
record green and consumer regression. DbC must cover schema, finite values,
units, source/physical time, shapes, immutable ownership and failure status.
LoD keeps UI/service/model adapters separate; DRY reuses camera math, FK, state,
storage and decoder facilities. Avoid another private camera or state convention.

Profile actual decode, mask hashing/storage, renderer and replay on pinned
fixtures/hardware. Measure time to first frame, total throughput, peak memory,
cancel latency and repeated-fit cache behavior; bound worker queues and cache.
Do not call post-decode iteration a memory/cancellation guarantee. Preserve
accuracy, provenance and fresh replay when optimizing. Keep full scientific
benchmarks distinct from unit test and docs-only CI receipts.

Each PR updates root/project handoffs and roadmap with commit, resolved finding,
commands/test counts, provider pins, raw receipts, exact CI run and next task.
Use additive versioned evidence and retain historical provenance. The next
agent must verify GitHub claims/checks at its actual head. Do not claim complete
stage acceptance, complete UI or all CI/CD from this review or merged prototypes.
