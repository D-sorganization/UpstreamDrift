# Historical-Player Research Package Procedure

UpstreamDrift #11317 supplies the sanitized provider boundary consumed by
AffineDrift #4828. This is a local research handoff, not publication, registry
attestation, original-media redistribution, or scientific acceptance.

## Public Interface and Authority

The SDK-free workspace facade exposes `ResearchAuditPin`,
`build_historical_research_package`, `export_historical_research_package`, and
`historical_research_schema_bytes`. Existing public `NecromatcherLibrary`
recall verifies immutable fit/model/capture bytes and exact source frame
identities. Schema validation is mandatory when export is invoked. The validator is loaded
lazily so the normal workspace facade remains importable when the optional
`jsonschema` dependency is absent. An explicit export then raises a descriptive
availability error; there is no ad hoc validator or unvalidated fallback. This
wave uses the existing host environment, which already provides the declared
schema-validation dependency; it installs/upgrades nothing and adds no dependency
declarations. Both dependency-present validation and dependency-absent facade
import/error behavior are tested.
The exporter does not compile a model, run an estimator, fit a
trajectory, render video, or modify the source library.

The provider contract is
`docs/api/contracts/historical-player-research-v1.schema.json`, with identity
`urn:upstreamdrift:historical-player-research:v1`. Its exact bytes were copied
from the reviewed Affine consumer prototype (SHA-256
`50da3be15d7efb87bd798d017d622d1ccc256ef5b7dcc42f7b42055067bb97fb`).
Repository Prettier normalization subsequently changed the byte hash to
`a194bf0c8145e417262f460c9bc4ea817b73d491b8468be7dfcd17a88334cb30`
without changing schema semantics; both provider and consumer use these exact
normalized contract bytes. Acquisition must pin the normalized file hash and
byte size, while the earlier hash remains prototype provenance only.
The schema is versioned independently of qualified mocap publication v1.
Only rejected, nonconverged, monocular research records are admitted in this
version. Finite target passes do not promote qualification or rights.

## Explicit Evidence Pins

Construct one immutable `ResearchAuditPin` per selected fit with:

1. Stable saved fit ID.
2. Independent assessment summary path and plain lowercase SHA-256.
3. Fresh full independent audit path and its separate SHA-256.
4. Read-only original downloaded source-video path.
5. Rights status: `unresolved`, `restricted`, or `documented_local_use`.
   Distribution remains `not_authorized` in all three cases.

The full-audit path must equal the summary's selected fresh-audit reference;
its explicit SHA must match both actual bytes and that reference. The selected
run must be unique, use a supported verified assessment protocol, and agree
with full-audit fit/model/capture identities and metric values. The canonical
saved numerical execution stamp must match the full audit and summary producer.
Unknown protocols fail closed rather than guessing their verification meaning.
V12 and V14 have explicit adapters because their existing summary field names
differ; neither adapter contains player-specific logic.

Saved training/held-out/dense image RMS must agree with fresh evidence within
1e-8 pixels absolute tolerance. Source/observation/training counts and unknown
visibility weight remain explicit. Original source bytes are independently
rehashed at export and compared with the canonical capture metadata and full
independent audit source identity. That media is never copied into the package.

## Hashes and Exact Source Clock

Each exported record contains plain SHA-256 values for source-video bytes,
capture archive bytes, native model XML bytes, saved fit JSON bytes, and a
research input binding. The input binding is SHA-256 of UTF-8 canonical JSON
with sorted keys, two-space indentation, no nonfinite values, and one final
newline. Its fields are:

```json
{
  "schema": "historical-research/input-binding/1",
  "fit_sha256": "<plain saved fit SHA-256>",
  "request_options": "<complete saved request-options object>",
  "assessment_sha256": "<plain reviewed summary SHA-256>",
  "full_audit_sha256": "<plain reviewed full-audit SHA-256>"
}
```

The placeholder for request options denotes the actual JSON object, not a
string in the serialized binding. This commits to full saved options and both
assessment artifacts without leaking their machine paths or large arrays.
Source-clock start/end use canonical `FrameIdentity.presentation_time`
rationals, serialized as numerator/denominator seconds. Original container PTS
must be exact and strictly increasing; physical time remains unknown. No
physical FPS, force, torque, anatomical reconstruction, skin mesh, or visible
club shaft is inferred.

Metric RMS is confidence-weighted Euclidean image error, retaining all positive
image observations and unknown visibility weight 0.5. Sampled native gap,
rotation, and penetration are converted only from meters to millimeters and
radians to degrees. They remain model-conditioned finite assessment quantities,
not physical measurements or continuous nonlinear certificates.

## Export and Consumer Review

Use `build_historical_research_package(library, pins, source_commit)` for
read-only candidate bytes. The export implementation's exact 40-hex
`source_commit` is distinct from each record's historical numerical producer.
Its repository tree URL is a code reference, not a claimed public package URL.
Record preparation brackets all canonical and external inputs twice; changed
bindings refuse export. Records and targets are sorted for deterministic bytes.

`export_historical_research_package(library, pins, source_commit, destination)`
requires a new destination, creates it exclusively, and returns SHA-256, size,
contract identity, source revision, and local/rejected scope. No existing file
is overwritten. Save the exact schema bytes alongside the manifest for the
Affine consumer's independent hash/size pins. Do not export before the reviewed
provider source is committed and frozen by the owning workflow.

The Affine consumer then performs its separate strict admission, atomic local
draft installation/recall, and QMD preview. Upstream exporting success is not
consumer integration proof. Preserve both providers' provenance and rights
warnings in the local preview; do not add a public route or authoritative
release-registry entry from this research artifact.

## Turnover and Verification

Focused tests exercise generic third-player output, deterministic bytes,
status separation, changed external/source/model/capture/fit inputs, rehashed
but contradictory metrics, source-clock promotion, metadata safety, exact
schema parity, SDK-blocked cold import, and refusal to overwrite. Production
functions remain bounded and typed. Root owns Git/lease coordination and the
same-wave development log, SPEC, actual export receipt, and later local
consumer preview. Synthetic tests are implementation evidence; actual
Tiger/Hogan exports must be separately recorded after source freeze.

## Club-Evidence Follow-Up

Issue #11318 tracks source-club evidence and overlay validity after actual
attachment-seed/visible-shaft mismatch. This export preserves that limitation;
it does not turn generic seed geometry into a measured shaft, public player
model, or accepted historical reconstruction. Additional club evidence requires
its own source-bound contract and validation before it can change qualification.
