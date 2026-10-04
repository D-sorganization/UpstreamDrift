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

The restricted Tiger trial uses original training frames `0,10,...,190` and
creates immutable versions with explicit initialization provenance.
The raw receipt preparation is
`Repositories/Temp/tiger-two-hand-receipt-preparation-v1/tiger-both-hands-review.json`
(2,636 bytes; SHA256
`a71221938c31e57719440566d5e91a9871fc428cc285c63ed08326bec95f94dc`).
It adapts the exact root human review and passes typed byte/schema validation;
the controlled author stage registered these exact bytes in the historical
library. Fresh canonical capture binding remains required on reuse.
The whole-capture preserved spline cannot silently be claimed as an exact
restricted restart. The existing authored contact schedule ends at frame 209;
its explicit source-clock restriction ends at frame 190, retaining the existing
in-domain review anchors at frames 105 and 120 for the final unpinned phase.
Do not silently drop contact records or zero post-release grip losses and report
them as passed constraints. Scope integration and an optimized scoped motion result remain
separate acceptance steps. The full Necromatcher goal remains active.

### Restricted Trial and Partial Publication

Producer `6695085027bdf4f354d867136930931591002d80` first tested the shorter
sampled-parent initializer without optimizing or writing the library. The public
Hermite domain rejected it because physical velocity violated Bernstein control
bounds. This failure motivated a separate authored initialization operation;
it did not justify silently projecting a strict restart.

The native preflight preserved all 14,825 tracked source files and 391 library
files. The authored initialization changed 198 knot velocities to zero and
changed no knot positions. Its initial training body RMS was 31.8610 px; its
maximum scaled geometric residual was 656.398. These describe the seed and do
not establish complete constraint feasibility or scientific qualification.

The two-stage execution saved `tiger-both-hands-authored-seed-v21`, then started
a strict preserved-spline fit from that exact new seed. The authored worker
reported 59.4807 seconds and no optimizer invocation. The strict worker exceeded
the unchanged 300-second wall budget; root observed session 32543 close with
exit 1. No final `tiger-both-hands-scoped-fit-v21`, final candidate or completion
receipt was published. Terminal evaluation counts and solver elapsed time are
unavailable; the failure cannot distinguish setup costs from solver costs.

The failure receipt verifies original non-index asset bytes and original index
records were preserved. Ten allowlisted files were added: the registered scope
review, authored seed, successful author-run records and failed strict-run
records. No active run remains. The partial closure proof is in
`Repositories/Temp/tiger-both-hands-fit-preparation-v2/root-execution-release-v1`;
the durable journal and preservation receipt are in
`Desktop/Necromatcher Review 2026-10-01/Tiger Both-Hands Scoped Fit V1`.

Do not apply a completed-final-fit assessment or final-fit rendering proposal to
this partial result. The V4 partial checker under
`Repositories/Temp/tiger-two-hand-partial-close-root-release-v2` independently
accepted the seed and failed-stage serialized records; its result SHA256 is
`f33565880fd09a2a03287b125fe5f638e827c9b419eddf84611bbda8959bbe80`.
No strict final start was independently computed.

Setup profile V1 failed on an ambiguous ndarray adapter/receipt JSON; the failed
artifacts and preservation proof remain intact. V2 completed five public stages:
parent binding 0.964918 s, seed binding and shaft admission 19.146040 s, strict
seed public initializer 0.571843 s, parent body/shaft metrics 2.148624 s and seed
body/shaft metrics 2.134822 s. The independent serialized assessment SHA256 is
`104a935990ec4a25a005153f0d1524037a08b0d638e0ac43e9855bb7a0aa0bdd`;
root acceptance is under
`Repositories/Temp/tiger-both-hands-profile-independent-v1/root-release-v1`.
This read-only setup profile establishes no timeout cause or solver speedup.

On identical 191-frame dense and 20-frame training domains, confidence-weighted
body RMS is respectively 35.102046/31.861015 px for the seed and
24.082673/21.776978 px for the parent. The seed is worse; retain the parent.
Do not increase the budget, infer missing solver telemetry or retry automatically.

Canonical captions now derive authored-seed status from the authenticated fit's
operation, request policy and strict false optimizer/convergence fields. They
display **UNOPTIMIZED AUTHORED RESEARCH SEED** above the existing research and
clock text. Publication/download reconstructs this label from fresh canonical
fit metadata and rejects a forged or omitted seed declaration. Legacy captions
retain their existing layout. Root passed 58 caption/video/job regression cases;
Ruff and pinned mypy passed. Earlier 29 synthetic profile/partial-check cases
remain historical. Current root checks passed 33 partial V4, 14 profile V2,
24 independent-profile and nine still-checker cases.

### Accepted Seed Stills and Preservation

Published producer `0143b39f571cbe67c57091fb2ae19e7d7eadae3b` rendered only
seed frames 0, 150 and 190 in
`Desktop/Tiger Both-Hands Authored Seed Shape and Skeleton V1/selected-stills`.
Each retains opacity 0.35 model-proxy shapes, skeleton and the conspicuous
unoptimized authored research seed caption. Root viewed all three; the middle
frame has substantial wrist/arm/neck mismatch. Unweighted display-marker RMS
is approximately 6.45/44.14/18.27 px, distinct from weighted body metrics above.

The independent serialized source/image result SHA256 is
`dd779807794509a8d2c10912e4cc1d3b74a611015b0708efae51f50920d98217`;
root acceptance is under
`Repositories/Temp/tiger-authored-seed-stills-independent-v2/root-release-v1`.
It verifies source/frame/hash/clock, geometry provenance, scope and caption
bindings, not independently recomputed native projection or complete overlay-union
pixel parity. Both independent profile and still checks preserve the complete
14,826-source-file/401-library-file maps. No parent replacement, optimized result,
new full video, ControlTower delivery or scientific qualification is claimed.
Tiger remains restricted to [0,191); exact physical release is unmeasured and
Hogan is unchanged.

Performance child #11445 now implements public authenticated_read: one canonical full PNG/BGR/clock validation per capture per operation, fresh complete file hashes and metadata checks on reuse/close, cross-library/thread/task and nested-context rejection, immutable DTOs and failure reset. Final root validation passed 93 cases across two lanes (86 plus seven public-owner/fingerprint cases; one Windows symlink-privilege skip and inherited import warnings). Canonical public SHA256 validation is reused, and execution fingerprints cover shadow_tracker; this API checkpoint is separate from the accepted artifacts at 0143. This integration checkpoint implements short authentication boundaries for queue admission, worker setup and fresh output checks, candidate/delayed publication and storage. Contexts close before scheduling, computation, delivery or persistence; delayed publication closes before add_fit, which uses its own fresh validation context. Root passed 188 distinct cases (185 SDK-free plus three native temporary-fixture cases; one Windows symlink-privilege skip, three native cases deselected in the SDK-free lane and nine inherited warnings). Historical performance remains unmeasured and profiling pending; no strict 300-second retry, speedup or new ControlTower delivery is authorized. Accepted artifacts at 0143 are unchanged.

The root integration acceptance receipt is
`Repositories/Temp/authenticated-refit-integration-root-review-v2/root-acceptance.json`,
SHA256 `13710191499ca3c558f5a554fba92e5f69aa908a8216312d711d9dc1f1b55e1d`.
Native regressions used temporary fixtures; no historical Library operation ran.

Methods V4 is the latest compiled supplement in the existing Desktop report
folder. Root reviewed all eleven pages, including three selected seed figures.
The accepted TeX SHA256 is
`2f59fa11f48cc31e3c2cde3c6ba59ec59989aa074b54281704910e02c2f8527f`
(27,501 bytes); PDF SHA256 is
`92719ded542ca88009678e64d5e80d84b1edbc66f9b723005640b376185bbd91`
(3,245,413 bytes). External `root-report-review-v4.json` SHA256 is
`f0dc26f2c3cad7abe530230f0b6bbc0da09163acb118482dfda92d1b282b57a4`
(5,488 bytes). The canonical review record may be formatter-normalized while
retaining the exact external review semantics.

The source uses three pinned external Desktop PNG paths from the accepted seed
pilot. These preserved images are required local repeatability dependencies;
the TeX is not a portable standalone bundle. Compilation used existing MiKTeX
with installer disabled after the built-in compiler could not locate platform
directories. Earlier Methods V2/V3 source/PDF/reviews remain preserved; their
historical page reviews do not replace the eleven-page V4 review.
