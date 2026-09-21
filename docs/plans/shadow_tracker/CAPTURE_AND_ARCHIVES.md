# Capture and Archive Workflows

## New Capture

Use existing Capture Rig sessions. Prefer three or more distinct calibrated
views, with one reserved for evaluation; two views are a useful input mode but
do not provide a held-out camera. Select positions that reduce arm/torso and
club occlusion, and retain the complete golfer and club arc. Avoid near-collinear
camera baselines. Capture a calibration target, ground plane and scale reference
over the working volume; preserve calibration residuals, not just matrices.

Record actual exposure, frame timestamps, clock synchronization evidence, focal
settings, shutter mode, camera poses and any stabilization. A short exposure
and high frame rate help fast club motion, but acceptance is driven by measured
blur and timing error. Pilot 120–240 fps where available; do not impose this as
an unsupported universal requirement. Record synchronization flashes/events,
clock offset and drift. Never assume matching frame indices imply simultaneity.

For scientific reference sessions, collect independent marker mocap, force plates
and club measurements when available, with their calibration and synchronization
uncertainties. Keep reference measurements out of the fitting input for the
silhouette-only evaluation. Include both handednesses and varied proportions,
clothing, backgrounds and cameras. Obtain subject/source permissions through
the normal project process and record the resulting allowed uses.

## Historical Archive Pipeline

1. **Catalog Before Download:** Record provider, source URL, golfer attribution
   confidence, event/date if known, suspected duplicate IDs, rights/allowed-use
   status and access method. Unknown attribution remains unknown.
2. **Review Access:** Use permitted downloads/APIs or user-supplied files. No
   bulk scraping or inference that old footage is automatically redistributable.
   Cache original source and checksum; respect provider limits and access terms.
3. **Detect Shots and Swings:** Record cuts, replay sections, views, subject
   identity, visible phases and truncation. Keep different swings separate even
   if clothing and camera match. A broadcast cut may be a different take.
4. **Normalize With Provenance:** Detect interlace/telecine, repeated frames,
   variable rate, aspect-ratio changes, mirror/rotation, overlays and damage.
   Keep original frames and a reversible transform/timestamp history.
5. **Grade Suitability:** Flag golfer/club visibility, resolution, blur, static or
   moving camera, scale evidence and physical-time evidence. Route recoverable
   cases to correction; reject or abstain with reasons on unsupported cases.
6. **Review Masks and Cameras:** Track the golfer through occlusions; exclude
   spectators, broadcast graphics and cast shadows. Store separate body/club
   channels and unknown pixels. Review first, occlusion, transition and impact
   frames plus low-confidence intervals; preserve each manual edit.
7. **Fit Hypotheses:** Solve plausible camera/scale/timing families. A clip with
   unknown physical time may support a phase trajectory but cannot qualify
   absolute angular speeds, powers or torques.
8. **Audit and Catalog Results:** Store accepted and rejected runs, candidate
   families, assumptions, evidence tiers and compute/correction cost. Resume by
   immutable asset/run IDs; avoid refitting duplicate archive encodings blindly.

## Restoration Policy

Deblurring, denoising, colorization and frame interpolation can manufacture
apparent detail. Restoration is an optional aid with a recorded transform/model;
original frames remain the observation authority. Generated intermediate frames
are never independent samples. Compare sensitivity with and without restoration.
Do not infer confident club position from interpolated streaks.

## Evidence Modes

| Mode                                    | Permitted Interpretation                                     | Key Limit                                              |
| --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| Calibrated Synchronous Multiview        | Metric kinematics and model-conditioned dynamics after gates | Still needs timing, morphology and physical validation |
| Calibrated Single View                  | Metric candidate family under priors                         | Hidden depth and self-occlusion remain                 |
| Estimated Archive Camera and Known Time | Conditional reconstruction with camera/scale sensitivity     | Accuracy depends on scene evidence and priors          |
| Unknown Scale or Physical Time          | Phase/shape hypotheses and qualitative comparisons           | No validated SI speed, force, work or power claims     |
| Truncated or Inadequate Evidence        | Partial labeled interval or abstention                       | Never present a completed invented swing               |

## Pilot and Scaling Plan

Start with approximately 10 rights-reviewed historical clips spanning static
camera, moving camera, interlaced/telecine and severe blur/occlusion; include
unusable examples intentionally. This is a workflow pilot, not sufficient
scientific validation. Benchmark the same degradation classes on modern footage
with known ground truth. Split by golfer, recording session and source lineage
so re-encodings of the same film cannot leak between training and holdout.

Only expand mining after suitability, deduplication, rights tracking, correction,
resume, and uncertainty gates work. Report discovery-to-acceptance yield with
all exclusions, failure reasons, annotation minutes and compute minutes per
clip. No footage is acquired or model weights downloaded by the planning setup.
