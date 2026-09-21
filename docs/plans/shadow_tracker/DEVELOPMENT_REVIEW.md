# Shadow Tracker Implementation Review and Delivery Handoff

For newer implementation findings and pickup instructions, read
[Current Turnover](TURNOVER_CURRENT.md). This document
preserves the earlier review and full delivery requirements.

## Review Baseline and Verdict

Reviewed commit `c3395229839cfb66ded994bbef0a39fcb985e1e6` on 2026-09-14.
Merged implementations: source/frame records #10145, camera bridge #10147,
and masks #10148; #10149 updates the dispatch table. Review tracking: #10150.
Published in [PR #10152](https://github.com/D-sorganization/UpstreamDrift/pull/10152).
This document reviews those changes and defines the remaining delivery work.
It does not certify an implemented fitter, a working launcher feature, or a
scientifically qualified model. The full implementation epic is still #10122.

The foundation is useful: immutable records, explicit unknown physical time,
mask provenance, lazy public exports, and reuse of existing camera types and
SE(3) inversion. However, passing packet tests does not establish complete DbC,
integration, performance, or scientific validity. Fix the concrete findings
below before widening the input boundary. Use [Continuation Prompt](CONTINUATION_PROMPT.md)
for the next agent; the old A/B/C implementation prompt is historical.

## Verified Evidence

- Re-ran 111 tests: 108 unit cases plus three model-probe integration cases,
  including real MuJoCo replay and CLI persistence. All passed on Python 3.13.5,
  Windows 11, using the exact pinned sibling Tools provider.
- Scoped Ruff check and mypy passed for the current package; these do not prove
  TDD history, branch coverage, or correctness outside tested cases.
- Inspected current source, provider contracts, packet tests, merged PR bodies,
  and launcher registration surfaces. Search found no Shadow Tracker entry in
  `src/config/models.yaml`, `pyproject.toml`, `src/launchers`, `src/tools`,
  `src/api`, or `ui/src`. There is no implemented end-to-end user workflow.
- PR #10148 reports an initial missing-module failure, then 20 passing mask
  tests. That is evidence of its reported development sequence, not proof that
  every behavior was independently developed test-first. Future PRs must retain
  runnable behavioral red/green evidence, not only collection failures.
- Existing physics evidence remains negative: see
  [Qualification Findings](QUALIFICATION_FINDINGS.md). Re-running its diagnostic
  successfully is not model qualification.

Commands actually run from the reviewed checkout:

```bash
python3 -m pytest tests/unit/shadow_tracker tests/integration/shadow_tracker/test_model_probe.py -m "unit or live_simulation" --no-cov -q --timeout=60
python3 -m ruff check src/shared/python/shadow_tracker tests/unit/shadow_tracker
python3 -m mypy src/shared/python/shadow_tracker --follow-imports=silent --ignore-missing-imports
```

For this review, `TOOLS_REPO_PATH` pointed at the initialized provider in the
sibling checkout. Use a properly initialized pinned provider in a fresh worker
checkout. Python 3.11/3.12 compatibility must still be proven by current CI.

## Findings and First Corrective Packet

Track fixes in #10151 under ST-02 (#10125). Preserve the public schemas unless
an explicit reviewed compatibility decision changes them. All examples below
were reproduced using the existing test factories and public functions.

| Priority | Finding and Evidence                                                                                                                                                                                                                                                       | Required Regression and Repair                                                                                                                                                                                                                                                             |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| P2       | `_validation.check_payload_keys` calls `.keys()` without a type guard. `SourceAsset.from_dict([])` raises `AttributeError`; `MaskFrame.from_dict({})` raises `KeyError('frame')`. `validate_frame_sequence([None])` also leaks `AttributeError`.                           | Reject wrong container/element types with field-specific `TypeError`; reject missing required fields with an explicit documented `ValueError`. Test every record and mixed-type unknown keys. Centralize key validation and check sequence elements before accessing their attributes.     |
| P2       | Schema values of integer type and integer `rights_status` produce `ValueError`, contrary to the frozen wrong-type policy. `FrameIdentity(physical_time_s=1, ...)` silently becomes `1.0` although the frozen field is `float` and the pickup checklist prohibits coercion. | Add type/value matrices. Apply the common `TypeError`/`ValueError` policy consistently. Resolve physical-time integer acceptance explicitly against the frozen contract; do not let an implementation silently redefine it. Preserve unknown time and rational PTS.                        |
| P2       | `MaskFrame.from_dict` builds all three byte arrays before constructor dimension/length checks. An oversized `Sequence` whose `__getitem__` raises is read before its known-invalid length is rejected. This violates the explicit pre-conversion size requirement.         | Validate positive dimensions and all three payload lengths before reading/converting pixels. Use a sentinel sequence to prove that invalid length or dimensions cause zero pixel reads; preserve 0/1, bool, immutable ownership and foreground-validity checks.                            |
| P2       | Hashing a 1920x1080 mask materializes three Python lists and a JSON string on every property access. A single instrumented call peaked at 71.19 MiB of traced allocation and took 7.887 s.                                                                                 | Establish an uninstrumented repeated baseline and a separate memory benchmark before changing code. Keep JSON/hash compatibility; cache or stream only with byte-for-byte parity tests. Never invoke full-mask serialization/hashing in every objective evaluation or UI repaint.          |
| P3       | Reflection/NaN camera tests reject input in the existing observation constructor before calling the bridge. The projection test uses an optical-axis point with no distortion.                                                                                             | Retain provider tests but label their scope accurately. Add off-axis distorted point parity through actual consumers, tolerance-boundary cases, input/output ownership tests and both conversion directions. Do not claim a discovered camera arithmetic failure from these coverage gaps. |
| P3       | Root/project handoffs still describe A/C as ready and B as waiting, although all three are merged. Roadmap packets have no completion status. PR #10148 body contains literal escaped newline text.                                                                        | Replace stale pickup pointers with this review and the next issue. Publish readable PR bodies via `--body-file`; attach exact head/check evidence. Keep historical receipts intact.                                                                                                        |

The instrumented hash measurement used Python `tracemalloc` around one
`observation_hash` access on a preconstructed all-background, all-valid frame.
Tracing overhead is substantial: 7.887 s is **not** a production latency
estimate or a release threshold. It exposes allocation behavior requiring a
benchmark. Constructor validation and source decoding were outside the timer.

### Reproducing the Early-Length Finding

Turn this into a failing unit test before fixing the implementation. Here
`payload` is `a_valid_mask.to_dict()` for a 2x2 image:

```python
from collections.abc import Sequence

class Oversize(Sequence):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        raise RuntimeError("pixels consumed before length rejection")

payload["body"] = Oversize()
MaskFrame.from_dict(payload)
```

Current result: `RuntimeError`. Required result: a `ValueError` naming body
length, before `__getitem__` is invoked. Add annotations in committed tests.

## Full Delivery Sequence

This table refines, rather than replaces, [Work Packages](WORK_PACKAGES.md),
[Contracts](DATA_CONTRACTS.md), and [Scientific Gates](VALIDATION.md). The
original G0–G7 requirements remain in force. Work in focused PRs; completing a
row or a small slice never closes the whole epic.

| Stage and Tracking                             | Deliverable and Boundary                                                                                                                                          | Evidence Required Before Advancing                                                                                                                                                                 |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0: Contract Repair, #10151                     | Harden existing records without schema drift; fresh public-import and provider compatibility checks.                                                              | Behavioral red/green tests for findings; all packet tests; unchanged canonical hash fixtures; malformed-input matrix.                                                                              |
| 1: Model Qualification, #10124, #10140, #10141 | Correct named/native IK mapping, regenerate affected evidence, separate translational/rotational grip units, qualify canonical state/velocity and model geometry. | Independent FK parity, real engine replay, physical profile with justified limits, full-body/head/club/contact fidelity. No optimization on invalid evidence. Specialist review required.          |
| 2: Full Contracts, #10125                      | Freeze provider, camera-hypothesis, model, job, candidate and result contracts around existing camera/canonical types.                                            | Units/frame/time/ownership round trips; unsupported capabilities and missing evidence remain explicit; no marker RMSE substituted for pixels.                                                      |
| 3: Ingestion and Masks, #10126–#10127          | Local decode adapter, exact source/processed lineage, shots and synchronization; manual corrections followed by optional pinned segmentation.                     | Tiny licensed real videos: VFR, negative PTS, duplicates, interlace, cuts, mirror/crop and slow motion; no different-swing multiview fusion; manual revisions persist and invalidate descendants.  |
| 4: Projection, #10128                          | Filled body/club silhouette renderer using the same qualified model binding, valid-pixel-aware residuals and visibility.                                          | Independent analytic raster/projection oracle, offscreen/occlusion/thin-club cases, distortion/crop parity, G1; no false zero loss for missing pixels.                                             |
| 5: Initialization, #10129                      | Subject appearance and inertial assumptions separated; fixed morphology; camera/scale/handedness alternatives and initial velocity.                               | Known synthetic recovery and deliberately ambiguous single-view cases; no forced zero velocity or claimed unique depth.                                                                            |
| 6: Continuous Rollout, #10130                  | Engine-local adapter decoupled from marker scoring, narrow shared protocol, declared controls and contact model.                                                  | Regression of original consumers; one initial state, no per-frame resets or undeclared root forces; fresh replay and refinement, complete phase coverage.                                          |
| 7: Fitting and Evidence, #10131–#10132         | Budgeted control fitting, checkpoints and cancellation, multi-hypothesis results, per-quantity evidence gates.                                                    | Known-control recovery then mismatched-model holdouts; failed physics overrides lower image loss; unknown time/scale blocks SI kinetics; calibrated uncertainty or explicitly labeled sensitivity. |
| 8: Real/Archive Validation, #10133             | Locked modern reference cohort, degradation suite, reviewed historical pilot with rights and identity provenance.                                                 | Subject/session/source split isolation, held-out cameras, failure/yield reports, equal-budget baselines; archive workflow success is not ground-truth 3D validation.                               |
| 9: Product Integration, #10134                 | Shared service and persistent jobs, launcher workspace, reviewed result/export UI and React capability parity.                                                    | Real import-to-replay-to-export journey; cancellation/restart/error/empty/unsupported states; executable launcher and cross-model consumer tests.                                                  |
| 10: Release, #10135                            | Qualified advertised engines, installer/optional dependencies, versioned bundles, user/help documentation and measured performance profile.                       | Full applicable CI/CD at release SHA, install/smoke and real-data/scientific gates, artifact round-trip, no silent missing-engine skips.                                                           |

Image-only ingestion sub-slices and an evidence-review UI may proceed before
physics qualification only after a bounded child issue and explicit dependency
decision are recorded, as ST-D7 did for A/B/C. They must label fitting unavailable
and must not waive the full parent dependencies. Do not dispatch an entire
research stage to a cheaper agent with instructions to choose missing physics.

## Launcher and User Experience Acceptance

Use the existing launcher and tool-host architecture. Re-read the current
providers before coding; the inspected registration surfaces are:

- `src/config/models.yaml`: one discoverable Shadow Tracker tool tile.
- `src/tools/shadow_tracker/`: proposed thin GUI, `__main__.py` and lazy embed
  adapter; the package does not exist at the reviewed baseline.
- `pyproject.toml`: matching `upstream_drift.embeddable_tools` entry point and
  actual optional dependencies. Follow `embedded_tool_bootstrap.py` source
  fallback policy; test installed entry-point discovery as well as source runs.
- Existing `launcher_embed` and `src/launchers/embedded_host.py`: tab/dock
  lifecycle, workspace persistence, theme, help and clean shutdown.
- `src/api/routes` / `services` and `ui/src/App.tsx`: reuse job/service and
  manifest route patterns for React; neither UI owns solver or camera logic.

The intended workspace has an import/source panel, synchronized video/mask/
render views and timeline, plus a results/job panel. Acceptance is behavioral:

1. Import a local historical clip or a calibrated capture set. Show source,
   timing, camera and rights assumptions in plain language. Missing timing is
   visible and never silently replaced by nominal FPS.
2. Select the golfer and swing, review body/club masks, correct an outline,
   undo/redo and save. Show occluded/unknown regions distinctly from background.
3. Review initial pose and alternative hypotheses. Use an overlay opacity
   control and frame stepping to compare alignment; show assumptions affecting
   scale/depth and which quantities cannot be inferred.
4. Start a bounded fit, see stage/progress/elapsed/budget, cancel responsively,
   and resume only from a compatible persisted checkpoint. Never block the GUI
   event loop with decoding, hashing, inference, rendering or dynamics.
5. Inspect synchronized observed masks and fresh replay, per-phase residuals,
   contact/grip failures and uncertainty. Failed results remain inspectable
   but cannot be exported with a validated status.
6. Export a versioned provenance bundle and compatible model motion; open it
   through the intended pose/replay consumer and verify coordinates, timestamps
   and independent replay. Save/reopen restores the same evidence and revisions.
7. Verify keyboard-only use, focus order, labels/tooltips, non-color-only status,
   readable light/dark themes, scaling, narrow windows and useful errors. Add
   automated UI tests and rendered screenshot review with a human-readable
   visual receipt; screenshots alone do not prove workflow behavior.

Do not advertise a ready tile for a shell or a diagnostic-only probe. An earlier
beta evidence viewer must disclose unavailable fitting. The final acceptance
requires the complete journey with a real engine, not a mocked service response.

## Engineering and Performance Gates for Every Slice

**TDD:** Record the requirement, a behavioral failing command and failure,
minimal implementation, green result and affected-consumer regression. Add
property/metamorphic tests for invariant spaces and independent numeric oracles
for geometry/dynamics. Use fakes for orchestration failure paths, never to prove
engine accuracy or scientific acceptance. No blanket xfail, widened tolerance,
test deletion or reduced collection to obtain green CI.

**DbC:** Test preconditions, postconditions and persistent invariants at each
public boundary: types, schema, finite values, dimensions, time bases, units,
frames, ownership, budgets and terminal states. Model illegal transitions,
unknown/unsupported input and partial output explicitly. Validated-result
construction must require independent replay and passing scientific gates.

**LoD:** UI depends on the application service; the service depends on narrow
providers; engine-local adapters own native details. Do not inspect private
engine arrays through object chains in the UI/shared solver. Existing immutable
camera DTO fields may be read inside the dedicated conversion boundary; avoid
cosmetic getters that merely hide coupling. Add import-boundary tests.

**DRY:** Search shared providers and read their public facades first. Reuse
camera transforms, canonical state, body bindings, job lifecycle and theme.
Centralize actual repeated validation, but avoid a second generic framework.
Keep one service path for PyQt, API, React and CLI. Record reuse decisions in PRs.

**Performance:** For each stage, benchmark cold/warm wall time, p50/p95 latency,
throughput, peak RSS and VRAM where applicable, copy/allocation volume and cache
hits. Use pinned hardware, dependency/model hashes, fixed 480p/1080p/4K fixtures,
single/multiview and short/long clips. Separate instrumentation from timing.
Decode in bounded chunks; bound queues/cache and worker count. Cache identity
must include source, masks, cameras, model, parameters and provider versions.
Verify edits invalidate only affected descendants and resume rejects mismatches.

Freeze actual numeric budgets with evidence before release; no performance gate
is currently qualified. Proposed interaction targets to validate are p95 under
100 ms for local controls and cancellation acknowledgement under 1 s, while
worker termination has a separately bounded timeout. Fit-runtime limits depend
on measured model/hardware profiles. CI should test deterministic work/allocation
bounds; stable dedicated runners should enforce timing regression budgets.
Optimize hotspots only after profiling and re-run accuracy/replay parity after
vectorization, caching, concurrency or precision changes.

## CI/CD and Completion Audit

Every PR runs scoped unit/type/lint/format tests, affected providers and consumers,
inventory regeneration when source files change, and current repository gates.
Add real-media, renderer, real-engine, API and GUI jobs as those paths land;
declare optional capabilities and keep skips visible. Required release jobs
must fail when their qualified engine or assets are missing. Use R2025b for
Simscape evidence. Do not change unrelated CI or add baselines to hide failures.

For launcher work, extend the actual manifest/embedded-host tests under
`tests/config/launcher_manifest` and `tests/launchers`, then exercise the installed
application. For React, use repository package scripts and generated API types.
For release, build/install clean artifacts with and without optional extras,
run the bundled reproducible example, and test saved-project migration/replay.
Inspect required GitHub checks at the exact commit and actual CD artifact smoke
results. A docs-only green run cannot certify code or an installer.

Maintain a gate ledger with requirement ID, implementation SHA, test command,
fixture/provider hashes, CI run, raw result, reviewer and unresolved limitation.
The full epic closes only when every stage, G0–G7, UI journey, advertised engine,
performance profile, install/export consumer and documentation requirement has
direct passing evidence. Missing evidence means incomplete. Historical footage
may legitimately yield an abstention; success includes explaining that result,
not inventing an accurate motion for an unidentifiable clip.

## Required Turnover per PR

Record merged/head SHA, issue state, allowed scope, exact red/green commands,
test totals/skips and environments, actual CI/CD URLs, changed public contracts,
reproduction fixture hashes, screenshots/performance/scientific receipts where
applicable, known limitations, and the next unblocked issue with its first test.
Update root/project handoffs, roadmap and the one PR-keyed SPEC row. Preserve
historical receipts. Do not leave stale prompts asking the next worker to redo
completed A/B/C tasks, and do not mark full ST-02 or the fitter complete.
