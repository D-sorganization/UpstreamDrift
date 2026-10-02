# Necromatcher Authored Ground Placement

## Create a Separate Hypothesis

Call `workspace.author_ground_placement(library, source_fit_id, new_fit_id,
source_frame_index, clearance_m, description)` from the clean SDK worker context.
Choose an exact saved frame, a finite nonnegative foot clearance in metres and
an explicit description of the assumed ground. The operation compiles the exact
bound native model once and uses the existing `preload_feet(..., preload=False)`
helper to place the lowest contact sphere at that frame. It applies one root
translation to every saved pose, leaving all rotational coordinates unchanged.

The pinhole camera translation changes by minus its rotation times the world
translation. The operation verifies rigid marker translation and conserved image
projection for every stored frame, with 1e-8 m and 1e-7 pixel tolerances. It reports
whole-track maximum penetration before and after the revision; placing one anchor
does not prove that every other frame is ground-admissible.

The result is a new kinematic-fit version, preserving capture/frame/model bindings
and false physical-time/dynamics flags. The current review geometry contains the
revised camera and unchanged attachments/free coordinates. It carries no obsolete
optimizer spline coefficients or solver-result metrics. The verified immutable
source fit retains the complete original solver evidence. Root/world offsets,
anchor clearance, diagnostics and implementation/runtime fingerprints are stored
in placement_revision provenance. Subsequent refitting uses the revised camera
and saved pose samples through existing worker interfaces.

## Recall and Validation

Kinematic-fit admission, recall and export revalidate the placement source fit's
hash and swing session, exact frames and model/capture bindings, unchanged joint
samples and the recorded root/camera transformations. Invalid anchor indices,
clearance evidence, camera changes and altered parents fail. This data admission
checks integrity and algebraic consistency; producer world-translation and contact
assertions are not independently rerun by storage.

Thirteen red-first cases cover all-frame pixel conservation, unchanged source/joint
samples, immutable parent recall, unsupported anchor/clearance/authorship values
and tampered provenance/camera/pose samples. The combined native library, fitting,
profile, replay and handoff suite passes 136 tests with one unavailable real-Drake
check skipped. Scoped standard mypy and Ruff pass.

## Qualification and Remaining Work

This is an authored world-origin/ground hypothesis, not an observed calibration.
It does not infer physical source time, anatomy, joint efforts or grip closure.
Scientific qualification remains rejected research. Preserve the original v5
Hogan/Tiger fits. Apply the committed procedure to separate versions, inspect the
remaining whole-track penetration, then address closure and historical control
recovery. Replay execution controls, unit-aware impact/whole-analysis consumers,
AffineDrift integration and full player acceptance remain open under #11235 and
#11232. PR #11240 remains draft and its CI remediation budget remains exhausted.

## Actual Committed Player Revisions

The [Placement Receipt](historical_capture/ground-placement-receipt-v6.json) records
execution from exact source 2c5129eab97d1d0d6b9a3110a4164e35212a1a3b with source/runtime
fingerprints and immutable parent identities. New versions are
hogan-authored-ground-fit-v6 (750 frames) and tiger-authored-ground-fit-v6
(210 frames). Their root shifts are 0.850056 and 0.495267 m along world Z;
anchor clearance is zero within floating-point precision. All-frame maximum
pixel differences are 5.68e-14 and 2.84e-13 px.

Hogan's whole-track maximum penetration decreases from 1.319760 to 0.469704 m;
Tiger's decreases from 0.495267 m to floating-point zero. Initial grip separations
remain 0.088525 and 0.053893 m. These hypotheses preserve source image evidence
and do not correct anatomy, camera calibration, missing foot observations or
closure. Both remain rejected research.

Read-only source inspection at Hogan indices 0, 549 and 749 finds the worst
penetration at index 549, a follow-through with the feet cropped from the frame.
The reviewed horizon/crop also varies. This supports reviewing swing windows and
landmark missingness before forcing a fixed-ground whole-track fit; it does not
prove calibrated camera motion. Raw PNGs and the full receipt remain outside Git
in the historical-capture/native-fit-research directory. The next actual fitting
work is bounded closed motion for Tiger and source-window/visibility review for
Hogan, followed by authored controls and independent replay.
