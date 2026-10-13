# Club Marker and Calibration Acquisition Requirements

Issue #10382 (MS-108). This page states what a capture must provide so that
each golf swing output is observable. It is kept in sync with the code by
`tests/architecture/test_club_marker_acquisition_doc.py`: every `REASON_*`
constant and every named limit below is checked against the source.

The limits are not defined here. They live in
`src/shared/python/motion_matching/club_face_target.py` and are reused, without
any tolerance of their own, by
`src/shared/python/motion_matching/club_observability.py`. If a value matters
to you, read the named constant.

## Output Policy in One Paragraph

`clubhead_position`, `clubhead_speed` and `sequencing` are advertised with an
uncertainty. `face_angle`, `joint_loads`, `impact` and `ball_flight` are
withheld as "not validated" unless the receipt supplies both a `calibration`
and an `evidence` entry (`output_policy.py`, `ADVERTISED_OUTPUTS` and
`WITHHELD_OUTPUTS`). Face angle has a further gate: when the observability
audit finds the head markers unable to fix the club roll, `face_angle` is
`unavailable` and cannot be validated at all.

## Minimum Head-Marker Triad

Face angle depends on the club's rotation about the shaft, which only a rigid,
non-collinear set of head markers can fix. The capture must therefore provide:

- **Three head markers** attached to the `Clubhead` frame, named by
  `HEAD_TRIAD_LABELS` (the first three `club` labels of the tour capture
  contract): `Marker_2:2:1`, `Marker_2:2:2` and `Marker_2:2:3`. The remaining
  `club` labels are not part of the triad.
- **Non-collinear placement.** The three offsets, expressed in the club marker
  frame, must not lie on a line. The spread of the offsets must reach
  `MIN_TRIAD_SPREAD_M`; below it the triad is degenerate and rotation about the
  common line is unobservable.
- **Rigid shape.** The attached offsets must describe the triad that was
  actually captured. The median rigid-fit residual between them must stay at or
  below `MAX_TRIAD_SHAPE_RESIDUAL_M`. Offsets copied from another club or head
  fail this check, so measure the offsets for the club used in the session.
- **Capture columns.** Every attached head marker must have a column in the
  capture (a label in the label list). A marker that was attached in the model
  but never exported is treated as missing.

## Required Calibration Artefacts

A withheld output reaches `validated` only with both artefacts, recorded in the
receipt `outputs` entry for that output:

- `calibration`: the calibration used for the session, for example the static
  address calibration (the tour capture contract can require one through
  `static_calibration_required`) and the camera or marker-system calibration
  that produced the metre-scale points.
- `evidence`: the validation evidence that the output matches a reference.

Supplying `calibration` and `evidence` does not override an unobservable
triad: `face_angle` stays `unavailable` while any `REASON_*` below applies.

## Frame Coverage

At least one frame must have all three head markers valid and finite. Frames
that lack the full triad are reported as gaps (half-open frame runs) and as a
coverage fraction (`observed_frames` over `total_frames`). Keep gaps short and
away from the impact interval; a capture with no such frame is unobservable.
The tour capture contract additionally rejects markers whose missing-frame
fraction exceeds `max_gap_fraction`.

## What Makes an Output Unavailable

`audit_face_observability` returns one reason; any value other than
`REASON_OBSERVABLE` makes `face_angle` unavailable.

| Constant                      | Text                                              | Fix at acquisition                                                                |
| ----------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------- |
| `REASON_OBSERVABLE`           | observable                                        | Nothing; all requirements met.                                                    |
| `REASON_TOO_FEW_MARKERS`      | fewer than three club head markers                | Attach and export all three head markers.                                         |
| `REASON_DEGENERATE_TRIAD`     | club head markers are collinear or coincident     | Spread the markers so they are non-collinear by at least `MIN_TRIAD_SPREAD_M`.    |
| `REASON_MARKERS_NOT_CAPTURED` | club head markers missing from the capture        | Export every head marker as a capture column.                                     |
| `REASON_NO_OBSERVED_FRAME`    | no frame observes the full club head triad        | Keep all three markers visible in at least one frame.                             |
| `REASON_SHAPE_MISMATCH`       | club head offsets do not match the captured triad | Re-measure the offsets for the club actually used (`MAX_TRIAD_SHAPE_RESIDUAL_M`). |

The exact strings are those of `club_observability.py`; the table is
checked against them by the contract test.

## Pre-Session Checklist

1. Three non-collinear head markers fixed to the head, offsets measured in the
   club marker frame.
2. Marker labels match `HEAD_TRIAD_LABELS` and are exported in the capture.
3. Static address calibration recorded when the capture contract requires it.
4. The full triad stays visible through the swing, including impact.
5. Calibration and evidence recorded in the receipt for every withheld output
   you intend to claim.
