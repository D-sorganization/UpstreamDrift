# Native Camera and Model Hypothesis Admission

Issue #11376 under #11357, #11232, #11226 and #11229 owns this versioned boundary.
It creates an immutable, unoptimized research seed from an authenticated saved
final spline and an explicitly authored candidate model/camera. It does not
calibrate a historical player, infer physical playback time, or optimize motion.

## Public Contract

`workspace.NativeHypothesisRequest.from_record` validates the exact
`necromatcher/native-hypothesis-request/1` schema. Its five nested inputs are:

- `parents`: saved fit/capture IDs and exact fit, capture, source-byte and complete
  source-clock hashes;
- `model`: an already registered candidate model ID/XML hash, exact native
  definition and named body-local marker offsets;
- `camera`: the canonical finite pinhole `CameraProjection`;
- `mapping`: explicit native coordinate order, metres/radians, free names and
  the saved final first pose as reference;
- `gauge`: positive authored stature and explicit world origin/orientation/source
  statements. These are conditional hypotheses, not subject measurements.

The gauge is explicit authored metadata. It does not itself enforce a model
dimension, world-origin constraint or calibrated camera/subject scale.

Nested records are immutable. Definition bytes preserve the native exporter's
JSON serialization convention; receipt dictionaries are detached copies.
Direct model construction requires the same exact UTF-8 byte serialization as
`json.dumps(definition, allow_nan=False)`. Other whitespace/formatting fails
early; `HypothesisModel.from_record` supplies the supported byte convention.
Unknown keys, qualification promotions, inferred units, nonfinite values and
ambiguous strict-restart declarations reject rather than receive defaults.

`bind_native_hypothesis(library, source_fit_id, request)` reopens canonical assets
and authenticates same-swing ownership, complete source frames, exact container
PTS, original encoded PNG hashes and decoded BGR/decoder-domain frame identities.
The PNG and decoded-frame hashes are separate domains. The complete clock digest
uses the same canonical byte convention as source-bound shaft admission.

`load_native_model_binding(library, model_id, definition_bytes, coordinate_units)`
reproduces the registered XML hash through the public native exporter and checks
registered/compiled coordinate order and compiled scalar units. It needs no
stored candidate fit. The legacy fit loader delegates to this resource seam.

## Explicit Rebound Start

The candidate must preserve the parent's named coordinate/free/locked mapping,
all dense poses, knot clock and coefficients. Canonical Hermite evaluation checks
the stored parent samples. The explicit reference must equal the saved first
pose exactly. Recomputed candidate poses may differ from saved samples only by
finite floating-point roundoff, with zero relative tolerance and an absolute
tolerance of `1e-12` in each coordinate's native units. Admission and publication
both enforce this tighter hypothesis-specific limit; the existing general
preserved-spline validator retains its original tolerance. Saved parent poses
are never substituted for freshly evaluated candidate output. Only the native
definition identity is rebound to the
new candidate; this is distinct from a same-model strict restart. Marker labels
and their order remain unchanged, and native body existence is checked by the
compiled public marker provider. Camera and offsets may change only explicitly.

`author_native_hypothesis(library, source_fit_id, new_fit_id, request)` uses the
fixed clean-worker transport and canonical initializer/result builder. The SDK
loads first in the child. The existing refit operation keeps its original
three-argument transport boundary, cancellation, timeout and cleanup behavior.
No user-selected Python module/callback or second job scheduler is introduced.

The seed retains the complete saved fitting configuration and explicit native
scales. Strict initialization evaluates the exact rebound coefficients; it
performs neither optimization nor implicit bound projection. Missing recorded
scales fail. Contact hypotheses are freshly source-bound against the candidate;
they are not measured contacts. Prior shaft results remain parent provenance,
not newly admitted candidate shaft measurements. Any later shaft continuation
must use its existing fresh canonical evidence binding.

## Persistence and Recall

The result records the authored request, complete capture identities, original
and rebound start snapshots, current source/runtime stamps and false camera,
anatomy, physical-time and continuous-certification flags. Optimizer/convergence
remain false. Body pixel diagnostics describe candidate projection only.
Recall reauthenticates parents and checks lineage, exact dense poses, camera,
markers and both start snapshots. The full inherited recipe is checked at
publication and recall, including training frame identities, coordinate scales,
confidence weighting and fitting configuration. Missing lineage, changed worker
stamps or recipe tampering cannot publish a seed. Existing assets are never
overwritten.

## Reproduction and Turnover

Use the normal repository Python path and pinned Tools gitlink. Run the focused
`test_necromatcher_hypothesis_*`, model-binding and native-worker-transport tests,
then existing native/refit/shaft compatibility tests, scoped Ruff and the pinned
pre-push mypy hook. New worker, helper and facade bytes are included in the one
canonical execution fingerprint, with mutation regressions.

Validation uses temporary fixture libraries imported through the real capture
boundary and a real compiled native model. No canonical Tiger/Hogan hypotheses,
new player fits, optimizer trials or public release are claimed. The engineering
manual inventory remains blocked pending independent inventory/release evidence.
Root coordinates commits and any subsequent real-player authorization.

At the isolated implementation checkpoint, 160 focused/compatibility cases passed;
after final formatting, 19 native-worker, SDK-free import and source-fingerprint
cases passed on the exact source bytes. The native fixture checks exercise a
changed camera, one compiled candidate, a clean child initializer and rejection
of an unknown native body. Eight worker-response tamper cases reject publication;
configuration, scales and training-frame tampering were reproduced red before
the shared parent-recipe guard. Pinned mypy, scoped Ruff/format, title case and
design-manual governance passed. These are implementation checks, not player
calibration or release evidence.

The independent correction checkpoint adds nextafter and near-limit roundoff
fixtures, an early noncanonical-definition-byte rejection, camera-only admission,
an actual 25 mm authored torso-transform/FK change and fresh canonical shaft
binding/projection of a newly admitted seed. The final combined hypothesis,
native, refit, shaft and persistence lane passed 144 cases. This fresh shaft
fixture is a separate source-authenticated assessment, not inherited parent
metrics or a historical-player trial. Issue #11376 acceptance still requires
reviewed real-candidate projection/provenance evidence before closure.

## Published Forearm Review Context

The tracked standalone source is
[Forearm Matching Results Supplement](necromatcher-forearm-results-supplement.tex),
with its [Final Review](necromatcher-forearm-results-supplement-review.json).
The 12-page PDF was compiled and every page visually reviewed. Its required
relative graphics and repeatability assets remain in the verified Desktop
package; copying the TeX alone does not provide those assets. This supplement
does not replace the authoritative QMD engineering design manual.

The original numerical producer is
`00b7d9846bd2fb2572b7c84f03cfc6c9699323cb`. Tiger V17 and Hogan V14 matched
31/33-coordinate arms remain nonconverged, rejected research results despite
passing the five finite research targets. The separately delivered ControlTower
packages contain 129 paired-shape/repeatability files and 27 report/repeatability
files, with remote byte-hash verification. Hogan's full shaded-video heuristic
failed; no full shaded-video qualification is implied. The new row-time software
path has 53 passing focused checks, but no historical row-time calibration or
new solver-budget result is claimed. Design-manual release remains blocked.
