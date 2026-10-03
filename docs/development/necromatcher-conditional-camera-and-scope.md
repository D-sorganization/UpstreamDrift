# Conditional Camera and Two-Hand Scope

## Completed Comparison

The read-only native camera study completed at producer
`55d17cfedf2e72e1a3c5a99c0f24ec1fa6f993cf`. Tiger retains the reviewed
half-open interval `[0,191)`; Hogan retains `[0,750)`. Two camera states per
player give 1,882 dense pose assessments. Original captures, models and saved
motion are unchanged. No new fit or camera seed was registered.

Tiger's cutoff conservatively excludes the uncertain right-hand-release
transition and subsequent follow-through. Twenty selected original images
were reviewed; this does not certify contact in every retained frame or measure
a physical release instant. Historical whole-capture fits and arm-morphology
measurements remain immutable. Their 210-frame Tiger statistics are not matched
comparators for this 191-frame study.

| Player            | Saved Camera Body RMS (px) | Conditional Camera Body RMS (px) | Decision            |
| ----------------- | -------------------------: | -------------------------------: | ------------------- |
| Tiger, 191 frames |         24.082673308003265 |               24.780344535782294 | Retain saved camera |
| Hogan, 750 frames |         10.136350597234246 |               10.090998303144502 | Retain saved camera |

Tiger worsens in all three predefined temporal thirds. Hogan's approximately
0.045 px body gain accompanies a worse seen shaft comparison. Neither candidate
is adopted or claimed as historical camera calibration.

## Public Method and Separate Metrics

The existing public `initialize_camera_hypothesis` performs SQPNP and RefineLM.
It receives native marker world points at saved poses, original positive-confidence
training observations and fixed assumed intrinsics. Tiger supplies 260 pairs
from 20 frames; Hogan supplies 286 pairs from 23 frames. Eligibility uses positive
confidence; the initializer itself is unweighted. Fresh body evaluation uses
the canonical confidence-weighted point RMS. These objectives differ, so a camera
initialization need not improve the reported weighted body metric.

One static camera is initialized per player. Temporal thirds are diagnostics,
not independent cameras or a search over preferred phases. Body remaining-frame
scores and the three shaft buckets below were already viewed and are not blinded
or untouched validation. Authored confidence and localization uncertainty remain
uncalibrated.

| Player | Shaft Bucket        | Observed Segments | Saved Raw RMS (px) | Conditional Raw RMS (px) |
| ------ | ------------------- | ----------------: | -----------------: | -----------------------: |
| Tiger  | V1 training         |                 1 |  66.69082241158017 |        66.12204236997074 |
| Tiger  | V2 seen evaluation  |                 1 | 145.90363070979896 |       131.22537829337338 |
| Tiger  | New diagnostic seen |                 2 |  131.7246097523064 |        142.2361142852277 |
| Hogan  | V1 training         |                 1 |  16.99773620516702 |       16.415242382053716 |
| Hogan  | V2 seen evaluation  |                 2 |  23.03782475542354 |         23.6458170359115 |
| Hogan  | New diagnostic seen |                11 |  36.89037372166752 |        36.38423228039291 |

Tiger V1 training frame 209 is excluded before term construction. V2 frame 60
remains an abstention and frame 150 remains observed. Abstentions have null
diagnostics and do not enter observed RMS denominators. The new diagnostic
cohort remains unadopted. Hogan frame 375 overlaps authored buckets; the buckets
are kept separate. Interior image fragments do not establish physical shaft
endpoints or shaft length.

## Closed Execution and Independent Review

Root observed producer session **74740** close with exit 0. Its finally receipt
reports operation completion and preservation. After 17 root-rerun SDK-free
tests, independent checker session **92074** closed with exit 0. Root accepted
the exact serialized camera/evidence/metric identities and complete source,
library, project, runtime and authority byte preservation. The 14,805 canonical
tracked files and 391 library files are unchanged; four Gitlink directory
entries are not file bytes. Counts do not replace full-map verification.

The independent check does not recompute native projections or reconstruct the
capture clock. Native canonical binding and immutable original archive/frame
bytes remain authoritative for those claims. No physical, anatomical, coupled
clinical ROM, continuous feasibility, rolling-shutter causation or dynamics
qualification follows from preservation acceptance.

| Artifact                       | SHA256                                                             |
| ------------------------------ | ------------------------------------------------------------------ |
| Released native config         | `ad1c1486880c426377c0cc203d27503fdc8f9359f4ee57fd6a2d4c8bc4827ff2` |
| Released native freeze         | `26becba9a03534d443bc194c22289e73d4e8bb720b3feb23117e5e5bf6b68bb8` |
| Native finally receipt         | `98d6cba317c227573d0439e40d995986680ab579a0ffa049e354ae19750f02ec` |
| Independent root acceptance    | `852d579c442bda76793ad959beda6e172e49df6b254e46f8e986b1acc8890d5f` |
| Current six-page report source | `1f02d20877d16672f2172749856ec8710c018701914624de73791cfeaf267cde` |
| Current report PDF             | `fd7f379e735f2f54f38ecada7b520d0849a3fb5b712f3b9aa18c3a4b1e9988df` |
| Current report root review     | `d1c2b94975d8b7dab7d2f60ad877cbdaecbbcbe5c3bb5f74d1c9b00d97ace8e4` |

## Artifacts and Repeatability

The original controlled preparation and releases remain under
`Repositories/Temp/conditional-camera-initialization-preparation-v2` and
`conditional-camera-independent-close-v1`. Completed outputs are on the local
desktop under `Necromatcher Review 2026-10-01/Conditional Camera Initialization Diagnostic V1`.
The editable report and PDF are in `Conditional Camera and Two-Hand Methods V1`;
the current edition is Methods V2. Its prior pending-review edition remains
under `accepted-pending-review-v1`. All six current pages were visually reviewed,
with zero overfull or underfull boxes. Existing MiKTeX compiled two passes with
installation disabled after the built-in compiler's platform-directory failure.

The source is also tracked as
[LaTeX Supplement](necromatcher-conditional-camera-and-scope-report.tex).
Machine outputs and large media stay outside Git. No new ControlTower transfer
occurred. The report's prepared manifest is historical preparation authority;
the current root review identifies the final source/PDF and accepted checker.

For a new controlled reproduction, use the public owners and frozen procedure
as a method reference. Capture fresh full baselines at the actual new producer,
review inputs and scope, keep gates disabled until reviewed, and create new
exclusive outputs. Use SDK-first native execution with one-thread BLAS and UTF-8;
preserve failures and observe process closure before independent checking.
Archived authorization does not admit another invocation, and the completed
exclusive output directory must not be reused. Forty-two preparation tests cover
selection, scope and failure boundaries; the independent check has 17 tests.

## Durable Fitting Scope — Issue #11414

The camera diagnostic already honors the two-hand window. Durable queue,
worker, persistence, descendant and native/web consumer integration is implemented
under [#11414](https://github.com/D-sorganization/UpstreamDrift/issues/11414).
The approved review window and actual fitted first/last source times are distinct.
Excluded body or shaft rows must reject before work; descendants inherit or narrow
the scope and cannot erase or widen it.

### Review Registration and Reuse

The raw `necromatcher/source-fit-scope-review/1` receipt declares the capture
identity, capture hash, complete source-clock hash, half-open frame bounds,
canonical first/excluded frame identities, review reason and uncertainty policy.
It must explicitly remain authored and uncalibrated. Exact container PTS does
not establish physical time or hand contact.

Use **Import Reviewed Window** in the native or web refit controls. Raw receipt
imports are bounded to 1 MiB and share `import_fit_source_scope_review`; the
existing immutable library stores the exact bytes. HTTP registration runs in
the shared Starlette threadpool; native registration runs in its background
worker, keeping frame authentication off the interface/event-loop thread. The receipt identity is
derived from its SHA256. An identical existing receipt is freshly checked and
reused; another asset cannot be overwritten. The returned scope uses an
`assets/...json` reference relative to the library, so removing the external
import file does not break recall or swing-package export.

HTTP consumers send a multipart file to
`POST /api/v1/necromatcher/fits/{fit_id}/source-scope-reviews` and use the returned
`necromatcher/source-fit-scope/1` declaration in `source_scope` when submitting a
refit. `GET /api/v1/necromatcher/source-scope-reviews/{review_id}` performs fresh
receipt/capture validation. Removing an imported declaration retains inherited
scope. A declaration alone cannot replace registered evidence.

Queue admission checks body, shaft and contact rows before scheduling. Workers
rebind before native compilation. Publication checks the exact requested
training selection and selected-domain binding, rather than accepting any
internally consistent subset of the reviewed window. Complete contact/config
snapshots and immutable parent lineage are checked again during recall; cyclic
lineage fails with a bounded error. Scoped overlay requests and manifests retain
the same reviewed window and selected-domain provenance. Historical unscoped
records keep their original producer identity and request shapes.

### Implementation Validation

The root combined lane passed 294 Python cases, including a real original
capture/native-bound parent through raw receipt registration and queue admission.
The original parent bytes remain unchanged and no candidate fit is published by
that scheduling test. The root interface lane passed 39 tests; the consumer
owner's broader lane passed 45. Ruff, format checks, pinned mypy, application
TypeScript and scoped ESLint passed. Seven document-title audits passed;
design-manual governance still reports `blocked-inventory-required`.

Preserved failed runs include six missing-library-method cases, receipt handoff
schema rejection, two cyclic-recall cases, publication selection/configuration
tampering, import and overlay admission failures. The first root combined run
passed 290 and failed four schedule-worker fixture cases because their fake
library lacked `load_fit`; that fixture now exposes its existing parent record.
Production validation was retained and the complete 294-case rerun passed.
These are implementation checks, not new historical fit results or remote CI
acceptance. Historical library assets remain unchanged. A focused upload
responsiveness follow-up passed 13 root cases with the asyncio plugin explicitly
loaded, preserving canonical error mapping and real native-bound admission.
The first follow-up runner lacked that plugin; its two runner failures are
retained separately from the implementation RED case.

A future scoped Tiger fit uses original training frames `0,10,...,190`, creates a
new immutable version and records `sampled_parent` initialization honestly.
The raw receipt preparation is
`Repositories/Temp/tiger-two-hand-receipt-preparation-v1/tiger-both-hands-review.json`
(2,636 bytes; SHA256
`a71221938c31e57719440566d5e91a9871fc428cc285c63ed08326bec95f94dc`).
It adapts the exact root human review and passes typed byte/schema validation;
it is not registered in the historical library. Fresh canonical capture binding
and controlled execution remain required. The preparation gate is disabled.
The whole-capture preserved spline cannot silently be claimed as an exact
restricted restart. The existing authored contact schedule ends at frame 209;
it needs an explicit source-clock restriction with new in-domain review anchors.
Do not silently drop contact records or zero post-release grip losses and report
them as passed constraints. Scope integration and a new actual motion fit remain
separate acceptance steps. The full Necromatcher goal remains active.
