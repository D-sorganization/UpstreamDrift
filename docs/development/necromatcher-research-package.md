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

## Actual Frozen Provider Export

The published export implementation was
`6654881b0788c77620a0afb8a9a6763f3ee1ccb9`. The actual local receipt
`historical-research-export-v1-receipt.json` is hash-verified as
`5d024fa8afe4c9e88639f20de46b4f7098b44e8a70b121ec1e96dc2c70e9f7ed`.
The exclusive Desktop `Affine Local Research V1` bundle contains:

| Artifact                                    | Bytes | Plain SHA-256                                                      |
| ------------------------------------------- | ----: | ------------------------------------------------------------------ |
| `historical-player-research-v1.json`        | 6,849 | `757a644c5408e4f5fc54904bd2918290f2c3af7d77d9a3b4045420334f1bc93d` |
| `historical-player-research-v1.schema.json` | 8,745 | `a194bf0c8145e417262f460c9bc4ea817b73d491b8468be7dfcd17a88334cb30` |

Public build and export bytes were equal. Source fingerprint
`0e36ff102e21393582257f6804265163613ec3a7aee797db2e5aa0a9ec2e221a`,
runtime fingerprint
`e927059cdd87573bb5c3bfd78c509429ef251caa5a1d3c6fd67cb8e3e64f3ba5`,
library assets and external evidence were unchanged across the export. The
before/after stamp observation timestamps differ as expected; complete stamp
objects are not claimed byte-identical. The operation ran no optimizer and
copied no original media.

The selected Tiger record is `tiger-geometry-weight-variant-fit-v14`, produced
numerically by `dafab40c107dbcd22d1c3a91bfcc79b102076f65`; the Hogan record is
`hogan-probe-density-variant-fit-v12`, produced by
`b32878a828d3b96bfc796bee368ef60cb692d117`. Tiger retains 210 source frames,
22 training frames, 2,730 dense observations and 286 training observations;
Hogan retains 750 source frames, 23 training frames, 9,607 dense observations
and 286 training observations. Exact presentation-time intervals are
`[3003/200, 659659/30000]` and `[110, 4049/30]` seconds respectively.

Both records remain rejected, nonconverged monocular research with unresolved
rights, unknown physical timing and distribution not authorized. Normal
commit/push hooks passed, including pytest, and the subsequent 476-case
Upstream integration selection passed. These implementation checks do not
promote either record. Actual Affine admission, atomic install/recall and local QMD preview were
subsequently recorded separately below. The provider export alone does not
prove that consumer stage or authorize publication.

## Actual Local Affine Consumer Evidence

The separate actual receipt `local-admission-preview-receipt.json` is 21,943
bytes, SHA-256 `ccf4615f4eb3d51a51b6b3a1af98b9bfd92abda4683be20b26744ff6c67136c2`.
Public inspection wrote no files. Exclusive draft installation, idempotent
installation, exact-byte recall and deterministic recalled QMD succeeded;
input, source, driver and authoritative registry brackets remained unchanged.
The consumer checkout base was `29472661de971899bfb7cdc4a3f92b3e9eee19aa`, with
tested working-tree source fingerprint
`4c712713d0aa4dde9afc34876a3b73a0074bbe04515b23215cca9f410cc6af09`.
That base does not identify a commit containing the consumer changes.

Quarto 1.8.26 rendered the standalone local HTML with `--no-execute` and exit
code zero. QMD identity: 4,850 bytes,
`811e52a380d275d119525bc9f7b013711c8cf8f2f33fe585ec60e268c48ace19`;
HTML identity: 26,172 bytes,
`079655676dbc588731a9c82c195e0e24db922eec11b4e5f35c7124e39340abe8`.
A separate actual browser accessibility-tree/three-screenshot review recorded
`browser-review-receipt.json`, SHA-256
`f39ffed5532bbc8700b32e50053abf4e0dafd335ad838c6b255130d3c6edadef`.
It inspected title, Hogan/Tiger rejected statuses, metrics, finite Tiger ground
failure and limitations, with no clipping/overlap in the reviewed default
1234-by-712 viewport. Manual user acceptance, public deployment and general
accessibility qualification were not established.

The package still contains Hogan V12 and Tiger V14. No new optimization,
original-media import, scientific promotion or authoritative registry mutation
occurred. Post-budget-refactor focused consumer validation passed 42 cases with
95.35% module coverage and strict mypy/Black/Ruff. The final unchanged SDK-first
offline run passed 6,685 tests, with 26 skips, 187 existing-marker deselections
and 61 warnings in 600.28 seconds. The earlier 6,681-pass run is historical
before the budget refactor. Host Python 3.13.5 remains distinct from the
repository target 3.12. Keep the exact local receipts and preview separate from
future consumer commits and publication review.
