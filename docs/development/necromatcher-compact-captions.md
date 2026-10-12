# Compact Research Caption Procedure

## Scope and Qualification

Issue #11356 adds opt-in `compact_research_v1` captions through the existing
source-overlay exporter and owned video jobs. This is a display change, not a
fit, anatomical calibration, contact measurement or physical reconstruction.
Historical exports remain unchanged. No native or web checkbox is introduced
in this implementation checkpoint.

## Public Recipe

Import `CaptionOverlayOptions` and `export_fit_video` from the public workspace
facade. Supply `caption_overlay=CaptionOverlayOptions()` to an exclusive output
destination, or pass the same typed keyword to `NativeVideoSession.submit`.
The HTTP video-export request accepts
`"caption_overlay": {"style": "compact_research_v1"}`. Unknown styles and extra
keys reject. Omitted or null options preserve the legacy request hash, worker
keyword call, caption pixels and manifest absence. The worker still imports its
SDK first; requesting captions does not create another scheduler or renderer.

## Layout and Provenance

The renderer measures complete text with OpenCV before drawing. It retains
source dimensions and uses a bottom strip of at most 20 percent of image height
(at most 48 pixels for 320 by 240), with nonoverlapping measured glyph/stroke
bounds and no font smaller than the existing low-resolution minimum. If all
required text cannot fit, export fails instead of truncating qualification or
source identity. The strip occludes source pixels within its declared rectangle;
it does not promise visibility of objects under that rectangle.

Visible text retains research status, unqualified camera/anatomy, unknown
physical time, source-frame index and exact rational source presentation seconds.
The compact matched-marker RMS is display diagnostics, not the fitting objective.
Green denotes observed landmarks, blue the native rigid rig and attachment seeds,
yellow residuals, magenta observed interior fragments and cyan the projected
infinite authored axis. Display-colored model surfaces receive a separate proxy
label with opacity; blue is never relabelled as the surface mesh.

Enabled manifests retain full legend/qualification/metric semantics and exact
per-frame layout records alongside the existing decoded frame identity, encoded
PNG hash, matched count/RMS, source cadence, model/fit/capture hashes and shaft or
surface provenance. Publication/download authenticate the recipe, bound source
identities/dimensions and regenerated complete caption records. Recalled jobs
expose the stored caption recipe; export success does not promote scientific
acceptance. No original source image is overwritten.

## Verification and Future Actual Review

Focused tests exercise measured bounds, mandatory text, rational clock identity,
unsupported dimensions, pixel preservation outside the caption strip, strict
API/session admission, captured recipe hashes, recalled ownership, manifest
tampering, exact disabled keyword/pixel parity and a real two-frame synthetic
codec fixture. These are implementation tests, not historical player validation.

A later actual display review requires a reviewed clean published producer and
separate execution authorization. Use new exclusive Hogan/Tiger destinations,
the same saved fit/model/camera/clock and unchanged shape/shaft options. Verify
all decoded dimensions/counts/cadence and source/library before/finally hashes.
Independently inspect matching original, legacy and compact stills at the same
source PTS, particularly Hogan 200/375/550 at 320 by 240. Record remaining text
occlusion and readability without claiming improved motion quality, historical
calibration or physical-time qualification. No actual historical export has been
performed for this caption checkpoint.

## Restricted Seed Provenance

The lossless restriction operation in #11450 uses the same compact caption
composer for stills and video. A freshly Library-authenticated fit with
`restrict_initialization` and `restricted_spline` provenance receives the leading
line `UNOPTIMIZED RESTRICTED RESEARCH SEED`. Its enabled manifest declares
`restricted_initialization_seed: true`; publication reconstructs that status
from the canonical fit and rejects missing, forged or conflicting declarations.
The text classifier checks mode and unoptimized status. The Library restriction
owner remains responsible for parent hashes, receipt derivation, selected scope,
source clock, bounds and curve identity; a mapping alone is not authentication.

`CaptionFrame` appends the optional strict-boolean `restricted_seed` field after
the existing seven fields. Authored and restricted flags are mutually exclusive.
No user option claims either status. Ordinary and authored captions retain their
existing text and layout; omitted caption options retain legacy output. All
source dimensions, shape opacity, skeleton layers and measured strip limits
continue to apply. The label does not imply improved fitting, optimized motion,
calibrated contact, individual anatomy or physical-time qualification.

Before a historical export, recall and authenticate the admitted restricted
seed, record its parent and canonical receipt hashes, use the approved source
interval and export to a new destination. Preserve source/Library/runtime
before-and-finally evidence. For Tiger, reviewed support is frames 0 through 190
inclusive; transition and released-hand frames remain outside the fit. Inspect
original and overlay at identical rational PTS and retain the unoptimized label
until a separately qualified result supports a different status.
