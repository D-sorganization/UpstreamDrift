# Necromatcher Fixed Row-Time Shaft Diagnostics

## Purpose and Qualification

The shared `historical_fit` facade exposes `ShaftRowTiming`,
`RowTimedShaftAssessment` and `assess_row_timed_shaft`. This is a read-only
fixed-assumption diagnostic. It runs no optimizer, estimates no sensor readout,
changes no footage and does not qualify the camera, anatomy, physical time,
shaft flexure or human range of motion. Issue #11357 remains open.

This document is a research procedure. The engineering design manual remains
`manuals/upstreamdrift`; its independent release gate remains blocked.

## Image Formation Assumption

For each observed fragment sample `(x, y)`, evaluate the saved motion at

`t(y) = t_PTS + scan_direction * readout_source_seconds *
(y / (image_height - 1) - reference_row_fraction)`.

The mapping is explicitly `authored_full_encoded_raster`, with a default
midpoint reference fraction of 0.5. The caller specifies either scan direction
and a finite nonnegative fixed readout in encoded-clock seconds. The physical
playback scale, original sensor rows, crop, fields, stabilization and exposure
reference are unknown. Other row mappings reject; these values cannot be
presented as measured sensor timing.

At each sample time, the diagnostic reuses `ImageFitResult.evaluate_source_times`,
canonical native marker FK, `CameraProjection`, and the same projected authored
infinite shaft-line geometry as the existing shaft residual. The camera remains
fixed at its saved pose. The two visible fragment samples are image observations,
not physical club endpoints. With nonzero readout they constrain two separate
instantaneous projected lines, so no common straight-line angular diagnostic is
reported.

## Identity and Support Admission

The immutable timing declaration binds the exact reviewed evidence digest and
source-clock digest. The diagnostic checks those identities, native model and
coordinate order. These checks validate declarations; they do not authenticate
caller-created records. Before use with stored research data, the canonical
workspace boundary must rebind original capture/source hashes, complete PTS
clock, decoded-pixel and PNG-byte identities, saved native definition and spline.
Do not use this mathematical helper as a publication or worker capability.

All observed endpoint times must lie inside the saved spline's closed support.
If any endpoint is unsupported, the complete evidence role/configuration rejects
before evaluating motion or FK. Never extrapolate, clamp, discard an endpoint,
adjust the readout after seeing the error, or change the RMS denominator.
Abstentions retain absent measurements and absent endpoint times.

Zero readout delegates to the legacy shaft assessment and preserves its exact
raw distances and RMS. Nonzero readout reports unweighted signed perpendicular
endpoint pixel errors and their RMS over all observed endpoints. Confidence and
authored uncertainty do not transform this diagnostic into a calibrated error.

## Controlled Comparison Procedure

1. Rebind canonical evidence and motion; pin producer, runtime, model, camera,
   source, full clock, decoded frames, PNG bytes, spline and review roles.
2. Freeze a finite readout sensitivity grid and both scan directions. Keep the
   saved motion, camera, morphology and evidence fixed.
3. Check complete-role support before native evaluation. Preserve rejected
   cases, with their reason, beside supported results.
4. Compare the exact zero-readout baseline with supported cases. Keep source
   images and all old observations unchanged; preserve source/library before
   and finally identities around actual execution.
5. Review residuals and images together. A lower diagnostic error under an
   authored timing assumption is sensitivity evidence, not historical readout
   identification or model acceptance.

The Tiger V17/Hogan V14 encoded-clock support screen tests readout fractions
0, 0.25, 0.5 and 1 of the encoded frame interval under the midpoint full-raster
assumption. Tiger V1 training has six unsupported cases, at boundary frames 0
and 209; the complete affected role/configuration must reject. Its V2 seen
evaluation has eight supported cases. Hogan V1 has eight supported cases and
V2 has sixteen. These are support counts, not fitted shutter results. V2 was
already inspected and is not untouched validation.

## Exposure, Flexure and Convergence

Exposure integration, shaft deformation, moving-camera pose and physical
playback time require separate hypotheses and evidence. Do not use an optimized
readout to absorb incorrect body dimensions, rig geometry or unconverged motion.

All four current forearm fits exceeded their evaluation budgets. A separate
budget study must reuse each original arm's exact start and first-pose prior:
the locked 31-coordinate Tiger V16/Hogan V13 parent final motions, and the
expanded 33-coordinate unoptimized Tiger V17/Hogan V14 seeds. Keep the objective
and quality limits unchanged. A new restart from an optimized final motion
changes the reference prior and is not a pure budget comparison.

## Software Validation

Tests cover exact zero-readout legacy parity, static-motion parity, an analytic
rotating-axis counterexample for both scan directions, complete-role rejection
before motion evaluation, foreign clock/evidence/model identity, invalid numeric
and mapping declarations, degenerate projection and explicit abstention.
The existing shaft residual tests exercise the shared geometry extraction.
Synthetic fixtures validate software only; no actual historical row-time
residual run or shutter calibration is claimed by these tests.

## References

[Oth et al., Rolling Shutter Camera Calibration, CVPR 2013](https://www.cv-foundation.org/openaccess/content_cvpr_2013/html/Oth_Rolling_Shutter_Camera_2013_CVPR_paper.html)
models camera timing using a known calibration pattern. That setting differs
from the present uncalibrated historical moving-shaft footage.

The canonical research review is
[Necromatcher Camera, Morphology and Shutter Review](necromatcher-camera-morphology-shutter-review.md).
