# Camera, Morphology, Shutter and Human Motion Qualification

Tracked work: #11357, with physical club acceptance under #11318 and adjustable
model-surface overlays under #11356. The Necromatcher goal remains active. This
procedure separates recorded evidence, authored assumptions and tested hypotheses;
it does not qualify the current four Tiger V16/Hogan V13 fits, which remain
nonconverged and rejected.

## Recorded Source Evidence

The accepted Tiger container window is 15–22 s: 210 frames at 1280 × 720 with
encoded rate 30000/1001. Hogan is 110–135 s: 750 frames at 320 × 240 and 30/1.
The source-only V2 audit independently recomputed all 960 presentation timestamps
and provenance-prefixed PyAV BGR pixel identities against preserved captures.
Original video hashes stayed unchanged.

FFmpeg 7.1 `idet` classified all Tiger frames as progressive in both its single-
and multi-frame results. Hogan single-frame results were 20 top-field-first,
14 bottom-field-first, 604 progressive and 112 undetermined; multi-frame results
classified all 750 as progressive. Neither accepted window has an exact adjacent
decoded duplicate. Hogan has one near pair at indices 166–167, with RGB mean
absolute difference about 0.3196 on an 8-bit scale. Selected stills show smear,
raster artifacts and thin-line localization uncertainty.

These classifications describe decoded image patterns, not original camera
hardware or exposure chronology. Progressive flags and lack of exact duplicates
do not identify original readout, recording rate, prior field processing or
playback speed. The original exposure duration and physical time mapping remain
unknown. See the [FFmpeg 7.1 implementation](https://ffmpeg.org/doxygen/7.1/vf__idet_8c_source.html)
and [shutter timing definitions](https://docs.baslerweb.com/electronic-shutter-types).

The earlier V1 Hogan screen used the opening montage, outside the accepted
capture. It remains preserved but is superseded; its 31 duplicate pairs are not
findings for the accepted Hogan window.

## Immutable Checkpoint Artifacts

The local review root is `Desktop/Necromatcher Review 2026-10-01`. New folders
are separate from the unchanged 1,504-artifact publication index.

| Artifact                                             | SHA256                                                             |
| ---------------------------------------------------- | ------------------------------------------------------------------ |
| Source Shutter Diagnostics V2 / diagnostic receipt   | `3c40c234611b50962a6e8cc706dc68a6752b51943ea0292c403dd9d1663d1652` |
| Source Shutter Diagnostics V2 / scientific review    | `033b56a0cc9bc6d73e0f453cb3220cb83ab8fd5f121a896d05f29aed18b47bb6` |
| Source Shutter Diagnostics V2 / preservation receipt | `e1cca0b10b136681e3d3acc7e56eb8f201fde350c7d99def6c47f5e3ce38b52d` |
| Source Shutter Supplement V2 / LaTeX source          | `181e01789cb6407c1cd65544dd7f4445e87be6f819240e4e13e263f2edb09a84` |
| Source Shutter Supplement V2 / nine-page PDF         | `179c35ee7538ac238093b50543783dd7cb59463bb1321683ec0311d8388e4c68` |
| Source Shutter Supplement V2 / verification receipt  | `61a5b01f0943857d84f9ccf4e5138a95f9661dbd7c41f7f7f06eef2c8c286d9d` |
| Saved Camera Description V1 / bracketed description  | `85460ba4e30339c7cae4312b9468573a794aa95a68bc4f1bfdb3bd6dfd438f20` |

The supplement compiled with the installed MiKTeX compiler, installer disabled;
all nine final pages were visually reviewed. It accompanies the preserved
74-page report. It does not replace the calculation design manual or scientific
acceptance inventory.

## Camera and Player Baselines

The saved camera is a fixed pinhole hypothesis. Focal entries equal image width,
and the principal point is the raster center. The SDK-free camera description
checks four raw fit hashes and its method/config bytes before and after; matched
control/variant arms have identical cameras. No native model or optimizer ran.

| Saved Hypothesis | Model-World Camera Center (m) | Optical Axis Azimuth / Elevation (deg) | Horizontal / Vertical FOV (deg) |
| ---------------- | ----------------------------- | -------------------------------------- | ------------------------------- |
| Tiger V16        | (4.5171, -1.1150, 3.2243)     | 157.1797 / -31.5440                    | 53.1301 / 31.4173               |
| Hogan V13        | (4.0103, -0.8367, 1.7238)     | 162.2785 / -11.6892                    | 53.1301 / 41.1121               |

Center is `-R.T @ t`, and optical axis is `R.T @ [0,0,1]` for the stored
world-to-camera transform. Azimuth is from model-world +X toward +Y; elevation
is above its XY plane. FOV uses continuous raster edges and zero skew. These
are supplied-model descriptions, not recovered historical filming positions,
verified player/target-line angles, calibrated focal lengths or measured heights.

Both saved definitions currently use the same authored stature/mass, 1.71 m and
78 kg, with arm scale 1.1, trunk scale 1.15 and shoulder scale 1.0. They are
generic population-model inputs, not player measurements.

Tiger's [official biography page](https://news.tigerwoods.com/about-tiger/) lists
6 ft 1 in, whose inch conversion is 1.8542 m. Its header is dated January 2014
and its body contains later wording; it is not a 2000 measurement. The
[PGA TOUR Hogan profile](https://www.pgatour.com/player/01528/Ben-Hogan/bio)
primary-domain search excerpt lists 5 ft 8 in and rounded 1.73 m; exact inch
conversion is 1.7272 m. Direct retrieval returned 403, so full-page verification
remains pending. Neither source supplies individualized segment geometry or
capture-date body mass. Later or ambiguous weight listings are not substituted.

Reuse `motion_matching.anthropometric_candidate` and `segment_scaling` for a
separately versioned candidate, after reviewing their documented native topology
and trunk limitations. Population-table segment parameters remain unqualified
for an individual. Reuse public `historical_fit.initialize_camera_hypothesis`
for pose-conditioned extrinsics; its low residual does not calibrate anatomy,
optics, depth or metric scale.

## Ordered Geometry and Motion Checks

1. Run the reviewed read-only geometry wrapper against a clean committed and
   published producer. Bracket source/runtime, driver/config and whole library
   identities; use exclusive journals and preserve failures. Preparation tests
   alone are not execution evidence.
2. Resolve authored shaft/grip/head solid centers, named frame origins and full
   left-wrist/right-hand attachment placements from exact saved definition bytes.
   Keep a named head frame distinct from the head solid COM and from a measured
   physical club-head point.
3. At every saved source pose compare public marker FK with body-pose plus local
   offset FK. Compare signed world grip separation to the native plant's metre
   vector, and separate position/rotation norms to public IK diagnostics. Do not
   combine metre and radian norms or refit the weld during an audit.
4. At first/middle/last poses compare canonical central differences at 1e-6 and
   2e-6 for world and pixel outputs. Report club references and the 13 body
   attachments separately, partitioned by free/locked coordinate and native unit.
5. Review native coordinate axes, sign, placement and rotation sequence before
   mapping them to anatomical ROM. The current two forearm Rz coordinates are
   locked at zero despite authored limits of ±90 degrees. Existing 32 finite
   ranges and 12 unbounded coordinates are authored model constraints, not
   clinical limits or evidence of observed forearm rotation.
6. Only after these checks, prepare a controlled forearm-unlocking comparison
   through the existing fitting pipeline, with exact inherited q/v/a trajectory,
   same camera/geometry/evidence/priors/budget and separately frozen recipes.
   Review sensitivity and body/club outcomes before changing further variables.

## Executed Authored Geometry Audit

The read-only four-fit audit ran at clean published producer
`7228fabe9b47813e60fe74d2f581acda1ff09b3f`. Its exclusive Desktop receipt,
`Authored Club Geometry Diagnostic V1/geometry-diagnostic-receipt.json`, has
SHA256 `2cddb400a82d41c270857de6e26c571e013b597f4209a9720c5df6f4c8bffd7e`
and 23,586,614 bytes. Exact whole tracked-source, runtime, library,
config, driver and helper snapshots agreed before and after. All 1,920 saved
poses across the four fits passed public marker/body-pose FK and separate
native closure agreement checks. No optimizer or library write ran.

Marker FK agreement was within 8.89e-16 m. Maximum authored right-grip
translation closure across source poses was 1.8303/1.8681 mm for Tiger
control/variant and 2.1867/2.1852 mm for Hogan. These are source-pose authored
attachment residuals, not measured physical grip error or the same sampling
as every stored fit diagnostic.

At first/middle/last source poses, both forearm coordinates were locked at zero.
The `LFInput` club-reference pixel Jacobian norm ranged approximately
178.47–480.76 px/rad for Tiger and 83.97–143.45 px/rad for Hogan. `RFInput`
had zero direct club-reference sensitivity in this authored kinematic tree but
nonzero body-marker sensitivity; closed-grip effects must be evaluated jointly.
These local two-step finite differences support a separately controlled
31-to-33-free-coordinate experiment. They do not identify observed pronation,
clinical ROM, global sensitivity or an accepted historical swing.

The authored club-head solid COM differs from its named frame by 0.5 mm;
its tested projection difference was at most 0.231 px for Tiger and 0.061 px
for Hogan. That convention difference is too small to explain the current
gross shaft mismatch. Wrist/body marker derivatives alone do not measure
grip-closure influence or establish absence of a joint effect. The independent
geometry verification receipt has SHA256
`2dc1fb9397224e4103d8a2edeed95072c132afdd7e2388861eecb3145b979ab4`.

Preserve the complete old Hermite motion with shared
`expand_image_spline_coordinates`, validate source and interior q/v/a, record
an explicitly unoptimized coordinate-expansion seed, then use the standard
native refit session. Keep model, camera, source/evidence roles, knot clock,
priors, geometry times and budget unchanged. Compare body, shaft and closure
outcomes before further adjustments.

## Shutter, Exposure and Flexure Hypotheses

Keep an exact zero-readout baseline. A proposed row-time model must expose its
readout direction, original sensor-row mapping, time origin and duration.
Evaluate both moving object geometry and camera pose at row time. Cropping,
resizing, field processing and playback retiming may make that mapping or its
physical units unidentifiable in historical encoded footage.

Treat exposure integration separately from row timing. Motion smear can occur
with simultaneous or sequential row exposure. A rigid authored shaft cannot
represent true flexure, but a curved visible feature does not by itself establish
flexure or rolling shutter. Do not add a readout parameter merely to absorb
disagreement from generic morphology, a locked coordinate or an incorrect rig.

[Oth et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2013/html/Oth_Rolling_Shutter_Camera_2013_CVPR_paper.html)
use a known calibration pattern; [Liao et al.](https://openaccess.thecvf.com/content/CVPR2023/html/Liao_Revisiting_Rolling_Shutter_Bundle_Adjustment_Toward_Accurate_and_Fast_Solution_CVPR_2023_paper.html)
jointly model camera motion and scene geometry. [RSL-BA](https://arxiv.org/abs/2408.05409)
informs rolling-shutter line geometry. A moving/deforming golf shaft is not
automatically a stationary world line for these methods.

Any field-separated, deinterlaced, stabilized or corrected derivative requires
an immutable version recording source bytes, filter/tool versions, declared
time/row mapping and fresh observation review. Compare it against originals;
never silently replace evidence or interpret visual improvement as validity.

## Trial and Display Acceptance

The canonical execution fingerprint must include shared `body_part_viz` source,
including projection, opacity and shape-transform code. Model/visual-description
hashes alone cannot identify renderer behavior. Fixed-commit mutation regressions
verify that changes in these dependencies invalidate the same source digest used
by native video jobs and publication guards. Earlier geometry diagnostics keep
their actual producer and full tracked-source brackets; no retroactive identity
change is claimed.

Declare train/evaluation roles, assumptions, priors, parameter bounds, gauge,
uncertainty and stopping rules before each comparison. Current V1/V2 shaft labels
have already been inspected; later adaptive tuning cannot call them unseen
validation. Reserve genuinely new reviewed footage or labels for later
generalization claims.

Keep skeleton and original footage visible in shape overlays. Reuse shared visual
geometry and saved native FK/camera; record authored hints versus derived model
proxies, exact identities and opacity. Native body visual solids are absent in
the current plain model, so derived surfaces are not measured anatomy. Opacity
changes presentation, not geometry, residuals, evidence roles or qualification.

Report convergence, rejection, body error, raw shaft error and closure/contact
units separately. Camera fit, subject geometry, coupled ROM, physical endpoints,
time, dynamics, impact and torque qualification remain explicit acceptance work.
