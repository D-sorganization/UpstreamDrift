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

Performance child #11445 now implements public authenticated_read: one canonical full PNG/BGR/clock validation per capture per operation, fresh complete file hashes and metadata checks on reuse/close, cross-library/thread/task and nested-context rejection, immutable DTOs and failure reset. Final root validation passed 93 cases across two lanes (86 plus seven public-owner/fingerprint cases; one Windows symlink-privilege skip and inherited import warnings). Canonical public SHA256 validation is reused, and execution fingerprints cover shadow_tracker; this API checkpoint is separate from the accepted artifacts at 0143. This integration checkpoint implements short authentication boundaries for queue admission, worker setup and fresh output checks, candidate/delayed publication and storage. Contexts close before scheduling, computation, delivery or persistence; delayed publication closes before add_fit, which uses its own fresh validation context. Root passed 188 distinct cases (185 SDK-free plus three native temporary-fixture cases; one Windows symlink-privilege skip, three native cases deselected in the SDK-free lane and nine inherited warnings). Published integration f2dfe completed read-only Profile V3: five stages took 0.672506/1.464802/9.053819/1.511551/0.460402 s; only seed binding decoded 210 PNGs, once in one canonical full authentication. Source 14,835/Library 401, project/runtime/config were preserved; assessment is byte-identical to V2. Root acceptance SHA256 b8469dc86d8aa72e1be2499786a107c847dd2e11d32f20567aa1effb1947a8e4 records this invocation; differing instrumentation/context/hardware prevents an isolated speedup or timeout-cause claim. No strict retry or new ControlTower delivery is authorized; accepted 0143 artifacts are unchanged. Pure restriction #11450 has a frozen V2 core (SHA256 b3504e6c64c7f97116a108b97a06d6ecba8732f3d38dd004751789e0bc93ce58) with 23 plus 53 agent cases. Root accepted 91 distinct cases (23 core plus 68 disjoint compatibility/bounds/initialization cases; synthetic MuJoCo-first fixtures only, nine inherited warnings and 38 unmarked cases reported only). Root acceptance SHA256 c0c2c44509650d37dc9516aa4185a54b828a49767bcca4ab8d8b2af64786b2ff records Ruff/format, pinned mypy and governance checks; manual release remains inventory-blocked. Pure q/v/a roundoff and Bernstein encoding evidence 17ddcd74c27276a376174c202500baa10e389d09b97ca27c3c42e9d1e4b778b5 is not native feasibility, pixel agreement, seed adoption or queue integration. Subsequently, native session 50689 closed exit0 at published 81aa448794a0ce763cdc2dd6e503d3f9b42200e9; root serialized closure accepted ce833a420c5e208b82bc430fe4be4c4c16830aee54d9c8c357bdb1af9e775e9c. Full source 14,837/Library 401/project/runtime/config/helpers matched the fresh baseline. Restricted-parent body RMS exactly equals saved parent (dense 24.082673308003265, remaining 24.337344114034266, training 21.776977995696026 px); 381-point q/v/a roundoff and typed receipt equality hold. The full 176 x 12 dimensionless scaled constraint matrix has maximum 17.023155533359613 at ground:heel_l, source 18.501816666666667; continuous contact is uncertified and nodes differ from V21, so no V21 656 comparison applies. Retained frame 0 raw shaft RMS 66.69082241158017 px/angle 28.301292322830996deg remains mismatched. No optimizer/convergence/physical qualification, telemetry null; no adopted seed, new fit, overlay rendering/pixel-parity evaluation or ControlTower delivery. Subsequent backend software checkpoint: root accepted paired restrict_initialization/restricted_spline operations with canonical SDK-free restriction receipts, preserved prior/bounds, parent agreement within 1e-12, scope/hash admission and worker/storage/recall/delayed-publication checks. Legacy semantics and public counts (eight option fields, seven start_native_refit arguments, eight builder arguments) are unchanged. Root passed 87 distinct SDK-free cases with nine inherited warnings; pinned mypy on all five owners and Ruff/format passed. Root acceptance SHA256 dc43749d0e69c6c43051712c5d37b819fbbe950edd3cf5dfb12f426a61b80bed authenticates the software freezes. Backend software was published at 5197f8639818321b7cf8f86d4748a25ce79aa278 with all normal hooks passing. Synthetic native end-to-end session 4534 then closed exit 0: the public queue, clean worker, storage and recall preserved the nonuniform restricted curve on [1,6), with four training/five dense frames, 43 locked coordinates and five rejected queue/storage counterexamples. Root closure SHA256 a26073a9c7bef6c8d5f2dd12b249c8e5f848be49c9fb0ba63d0a5c98f011d9bc authenticates the independently closed SDK-free audit (session 97412): fresh 14,844-source-file/401-historical-file/runtime maps are unchanged. Independent pure q/v/a errors are 3.47e-18/2.78e-17/1.78e-15; saved native pixel metrics agree with the actual run, but were not independently reprojected by the serialized audit. Optimizer counts remain null; no optimization, historical seed registration, new overlay or ControlTower delivery occurred. This synthetic software acceptance does not alter the earlier Methods V6 historical native-evaluation checkpoint or establish physical qualification. Subsequent caption software adds UNOPTIMIZED RESTRICTED RESEARCH SEED through the shared still/video composer and authenticated manifest, with eight strict DTO fields and mutually exclusive authored/restricted flags. Root passed 80 distinct cases in MuJoCo-first temporary fixtures: canonical exports and tampered lineage, manifest claims and exact legacy/authored pixels; configured pinned mypy, Ruff and format pass. Caption root acceptance SHA256 12a5e9e99d0b5d3827fcddc229324bf9e5525e448ae430acb4a3e0881a1f73d0 records this later software checkpoint, separate from the Methods V7 appendix at 5197. No new historical seed or overlay is claimed. Next: publish through normal hooks, then separately reviewed historical admission and the unchanged strict restart. The full goal remains active.

The root integration acceptance receipt is
`Repositories/Temp/authenticated-refit-integration-root-review-v2/root-acceptance.json`,
SHA256 `13710191499ca3c558f5a554fba92e5f69aa908a8216312d711d9dc1f1b55e1d`.
Native regressions used temporary fixtures; no historical Library operation ran.

Methods V7 is the latest compiled supplement in the existing Desktop report
folder. Root reviewed all eighteen pages through contact sheets and new pages
17 and 18 full-size; this does not claim every page was separately viewed
full-size. Accepted TeX SHA256 is
`c0b0f66c30adf4f11057b36cdb8cfef9afb6ec59627389e5f638e0f395fed060`
(47,804 bytes); PDF SHA256 is
`ab89dc0eda39384bea0451526dcf679efda84efbf02b231a6d4f083b39485afd`
(3,333,748 bytes). Candidate review SHA256 is
`09ea96dce4dfe40e58f096134610214b817a94468aa5f33d4808255407ba013f`.
Root acceptance is
`Repositories/Temp/necromatcher-methods-v7-candidate-v1/root-acceptance.json`,
SHA256 `51a8da1aaa9b590100354d8db1956431450b955d87ba57894b97da0df673c810`
(1,971 bytes). The canonical review embeds this checkpoint and preceding V6
reviews. The earlier body is unchanged except edition/cover labels; the new
appendix describes reusable restriction and actual synthetic native validation.

The same three pinned external Desktop PNGs remain earlier V21 authored-seed
figures and local repeatability dependencies, not a portable standalone bundle.
Existing MiKTeX compiled three installer-disabled passes; the final two have no
overflow or unresolved warnings. The built-in platform-directory failure is
preserved. Exact V6 source/PDF, sixteen rendered pages, reviews and diagnostics
were archived before publication; V2/V3/V4/V5 also remain intact. No new
historical overlay, restricted seed registration or ControlTower transfer is
claimed by this report checkpoint.
