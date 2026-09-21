# Image-Evidence Contract Freeze

## Decision ST-D7

Freeze the following **image-only** records and boundary tests for implementation.
These decisions do not depend on a usable full-body model. The first plan placed
all of ST-02 behind ST-01; the measured model defects show why that dependency
should apply to dynamics/provider/result contracts, not source timestamps or
mask identity. ST-01 remains open. No physics gate is removed or relaxed.

Only the fields below are frozen. Solver request/result, full-body canonical
state mapping, physical-time estimation, benchmark release thresholds and model
qualification remain specialist work. Do not implement those under these packets.

## Common Rules

Use Python 3.11-compatible frozen, slotted dataclasses with keyword-only fields.
Reject booleans where integers are required; reject implicit numeric strings.
Invalid type raises `TypeError`; valid type with invalid value raises `ValueError`.
Messages name the field. All IDs are nonempty, trimmed strings; do not silently
trim or repair. Hashes are exactly 64 lowercase hex characters. Schema tags must
match exactly. Records own immutable values; modifying constructor inputs must
not alter a record. Do not duplicate camera or geometry algebra.

Serialization is explicit `to_dict()` and `from_dict(payload)` per record.
Unknown fields are rejected in v1. Never store an absolute local file path in
portable records. No IO or model import occurs during record construction.

## Packet A: Source and Frame Identity

Implement in `src/shared/python/shadow_tracker/source_records.py`.
One private `_validation.py` may hold genuinely shared validation. It must not
grow into a generic second contracts framework; reuse the repository's
preconditions where semantics match. Keep façade exports lazy.

### SourceAsset

| Field                   | Type                                            | Rule                                                          |
| ----------------------- | ----------------------------------------------- | ------------------------------------------------------------- |
| `schema_version`        | `str`                                           | Exactly `shadow-tracker/source/1.0.0`                         |
| `asset_id`              | `str`                                           | ID                                                            |
| `source_uri`            | `str`                                           | Nonempty trimmed provenance locator; parsing never fetches it |
| `content_sha256`        | `str`                                           | Hash                                                          |
| `width_px`, `height_px` | `int`                                           | Positive                                                      |
| `rights_status`         | `Literal["unknown", "restricted", "permitted"]` | Exact enum; does not confer authorization                     |
| `rights_note`           | `str`                                           | Nonempty when status is not unknown; may be empty for unknown |

Use standard URI parsing solely to reject local filesystem paths and empty
schemes. Permit `https`, `http`, and `urn`; this record is a locator, not a network
access policy. User files can use a `urn:sha256:<hash>` locator.

### FrameIdentity

| Field                                                      | Type            | Rule                                                          |
| ---------------------------------------------------------- | --------------- | ------------------------------------------------------------- |
| `schema_version`                                           | `str`           | Exactly `shadow-tracker/frame/1.0.0`                          |
| `asset_id`, `shot_id`, `swing_id`, `camera_id`, `frame_id` | `str`           | IDs                                                           |
| `pts_ticks`                                                | `int`           | Signed presentation timestamp; negative start PTS is legal    |
| `timebase_numerator`, `timebase_denominator`               | `int`           | Positive; fraction must be reduced                            |
| `physical_time_s`                                          | `float \| None` | Finite if known; bool/string rejected                         |
| `physical_time_reason`                                     | `str`           | Nonempty when time is unknown; otherwise evidence description |
| `frame_sha256`                                             | `str`           | Hash of the decoded observation authority                     |

`presentation_time` is a property returning `fractions.Fraction` exactly.
It never supplies or estimates `physical_time_s`.

`validate_frame_sequence(frames: Sequence[FrameIdentity]) -> None` requires a
nonempty sequence of the same asset/shot/swing/camera, unique frame IDs and
strictly increasing presentation time. Negative starting PTS is valid. Known
physical times must strictly increase across the known subset; unknown gaps
remain unknown. Different cameras are separate sequences. A repeated source
image at a later PTS is legal and keeps its own ID; repeated PTS is rejected.

## Packet B: Immutable Binary Mask Observations

Implement in `src/shared/python/shadow_tracker/mask_records.py` after Packet A.
This slice handles a deterministic binary-mask baseline; soft probabilities,
segmentation models and camera warps are separate future slices.

### MaskFrame

| Field                   | Type            | Rule                                                        |
| ----------------------- | --------------- | ----------------------------------------------------------- |
| `schema_version`        | `str`           | Exactly `shadow-tracker/mask/1.0.0`                         |
| `frame`                 | `FrameIdentity` | Reuse Packet A                                              |
| `width_px`, `height_px` | `int`           | Positive                                                    |
| `body`, `club`, `valid` | `bytes`         | Exactly width \* height bytes, each value 0 or 1, row-major |
| `revision_id`           | `str`           | ID                                                          |
| `parent_revision_id`    | `str \| None`   | ID if supplied; cannot equal revision_id                    |
| `producer_id`           | `str`           | Manual tool or provider/checkpoint locator                  |
| `correction_note`       | `str`           | Nonempty for a revised mask                                 |

Require body and club to be zero wherever valid is zero. Body/club may overlap
because class ownership at boundaries can be ambiguous; neither is silently
subtracted. All-zero valid means missing evidence, not a good empty silhouette.
Add `has_valid_pixels` and `has_observed_foreground` boolean properties. An
all-background valid frame is allowed and remains distinct from missing data.

Serialize bytes as JSON integer lists in this tiny baseline. Reject values other
than the integers 0 and 1 (including booleans), with exact size checks before
conversion. Bytearray inputs must raise TypeError, not leak mutable ownership.
Do not add a hard-coded production video-size limit here; resource policies
belong at ingestion. Add a canonical SHA-256 `observation_hash` property over
UTF-8 JSON using sorted keys, compact separators and `allow_nan=False`.
The hash includes frame provenance and revision metadata as well as mask values.
Cache invalidation follows this identity; never overwrite an earlier revision.

## Packet C: Camera Direction Conversion

Implement in `src/shared/python/shadow_tracker/camera_bridge.py`; independent of
Packets A/B. Read both existing camera contracts first:

- `pose_estimation.observations.CameraCalibration` uses **camera-to-world**:
  `X_world = R_wc X_camera + t_wc`.
- `motion_pipeline.contracts.CameraExtrinsics` uses **world-to-camera**:
  `X_camera = R_cw X_world + t_cw`.

Implement `to_pipeline_camera(camera: CameraCalibration)` returning a tuple of
the existing pipeline `CameraIntrinsics` and `CameraExtrinsics`, and
`from_pipeline_camera(camera_id, image_size_px, intrinsics, extrinsics)` returning
the existing observations `CameraCalibration`. Do not introduce new camera types.
Use the existing SE(3) inversion helper where its convention is applicable;
verify `R_cw = R_wc.T`, `t_cw = -R_wc.T @ t_wc` in independent tests.

Supported v1 calibration is a pinhole K matrix with zero skew, bottom row
`[0,0,1]`, positive focal lengths, proper finite rotation and finite translation.
Reject unsupported K instead of silently dropping skew. Support no distortion,
four Brown-Conrady coefficients `(k1,k2,p1,p2)` and five `(k1,k2,p1,p2,k3)`;
other lengths fail. No distortion canonicalizes to five zeros on round trip;
four coefficients canonicalize to five with k3=0. Frame directions, projections
and values are preserved; original representation length is not preserved.

Images are not resized/undistorted by this bridge. A crop or mirror must already
be represented in the calibration for the selected image plane or handled by a
future explicit transform layer. Unknown camera hypotheses do not call this
bridge; never default them to identity.

## Required Independent Acceptance Cases

| Packet | Case                                                           | Expected Result                                                |
| ------ | -------------------------------------------------------------- | -------------------------------------------------------------- |
| A      | PTS -1, timebase 1/24, physical time null                      | Fraction(-1,24); physical time remains null                    |
| A      | Timebase 2/48                                                  | ValueError; unreduced input is not silently repaired           |
| A      | Same image hash, IDs f1/f2, PTS 0/1                            | Valid sequence; retained repeated image                        |
| A      | Same frame ID twice or PTS 1 then 0                            | ValueError naming field/order                                  |
| A      | Different swing IDs in one sequence                            | ValueError; no cross-swing fusion                              |
| A      | rights unknown with empty note                                 | Preserved unknown; not permission to fetch                     |
| B      | 2x2: body [1,0,0,0], club [0,1,0,0], valid [1,1,0,0]           | Valid, foreground present                                      |
| B      | Change a byte or parent revision                               | Different observation hash                                     |
| B      | valid all zero, body/club zero                                 | Valid record; both evidence properties false                   |
| B      | valid all one, body/club zero                                  | Valid pixels true, foreground false                            |
| B      | Foreground in invalid pixel; pixel 255; wrong payload length   | ValueError                                                     |
| C      | R_wc=I, t_wc=(1,2,3)                                           | R_cw=I, t_cw=(-1,-2,-3)                                        |
| C      | 90-degree Z rotation with nonzero translation                  | Round-trip proper rotation and correct transformed translation |
| C      | Nonzero skew, reflection, NaN, unsupported distortion          | Explicit failure; no lossy conversion                          |
| C      | World point obtained from X_camera=(0,0,2), K center=(320,240) | Both projections give (320,240)                                |

Each packet must also test schema/unknown fields, exact type validation,
serialization round trip and ownership. No packet may claim readiness of the
full ST-02 contract set or the Shadow Tracker fitter.
