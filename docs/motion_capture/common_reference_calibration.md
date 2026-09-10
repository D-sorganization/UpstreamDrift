# Calibrate With Paper or a Ruler

## Start With the Recording

Open or import a capture, then choose **Calibration → Paper / Ruler References…**.
The heading identifies the capture receiving the calibration. In **Camera Setup**,
select each camera's saved lens profile and record the lens, optical zoom, focus
and sensor/crop mode used for the recording. Use **Start a New Reference Session**.
You may save observations with unknown settings, but camera estimation requires
compatible calibrated lens profiles and fresh confirmation of the physical setup.

## Choose a Common Reference

Choose US Letter paper (8.5 × 11 inches), A4, a yardstick or a metre stick. Check
the actual object with a ruler; choose **Use Measured Reference Dimensions…** if
it differs from the nominal size. The measured dimensions are saved as a distinct
reference. A ruler's two endpoints provide scale evidence; they cannot establish
camera positions alone. Camera estimation needs identified, non-collinear paper
corners. Paper must remain flat, without curling, and be large enough in each
image to mark accurately.

## Mark the Same Physical Points

Mark one paper corner as **origin** and put an arrow along its long edge toward
**along-arrow**. The other two corners are **opposite** and **across-width**. Keep
those physical identities the same in every camera view; do not substitute the
screen's upper-left corner when the camera angle changes.

In **Reference Placements**, name the stationary position, select a camera and
an original recording frame number (starting at zero), then use **Open Frame and
Mark Points…**. Click the requested point or enter its original pixel coordinates
and choose **Set Point**. The selector advances to the next unmarked point.
Use Ctrl+mouse-wheel to zoom, Clear Point to remove a mark and standard Undo/Redo
to revise it. Save Points retains partial work; Cancel keeps prior observations.
Swing trims and crops do not alter these original calibration coordinates.

## Add Views and Placements

Keep the reference still while all cameras observe it. Reuse the same placement
name for its other views. Move only the reference, choose **Add Another Placement**,
name the new position, and repeat. Use at least two cameras and two paper placements
with overlapping views; each fitted observation needs four non-collinear points.
Spread placements through the capture area. Double-click an observation row to
select it for editing. Include / Exclude keeps rejected evidence in the history.

Reserve extra camera observations with **Reserve This View for Validation**.
Other fitted views must locate that same placement. A wholly reserved placement
cannot locate itself. A useful validation set checks observations the solver did
not fit; more points alone do not prove a precise physical calibration.

## Establish the Ball and Target Direction

In **Camera Positions**, choose one paper placement as the world anchor. Its face
must lie flat facing up, and the marked long-edge arrow must point toward the
target. Measure the origin corner's offset from the ball: +X toward the target,
+Y upward, +Z the golfer's right. Enter metres, including the correct signs.
Confirm the offset and orientation and recheck the recorded camera settings.
Choose **Estimate Camera Positions**. The cancellable calculation runs outside
the UI; its status remains visible.

## Review and Use the Estimate

Review each camera/placement's mean and maximum pixel errors, distinguishing Fit
from Validation rows, and read all limitations. Poor errors, ambiguous markings
or uncertain reference dimensions need correction and another estimate. Small
pixel errors are not a certificate of millimetre accuracy. Only after review,
choose **Use Reviewed Camera Layout**. The app selects that camera file for Match;
return to the Capture Wizard and refresh status to continue detection and fitting.
Lens distortion is retained for original-video overlays and corrected once before
the pinhole reconstruction fit. Calibration does not perform pose detection.

## Save, Reopen and Recalibrate

**Save New Revision** keeps observations, notes, reference dimensions and original
frame hashes. **Open Saved Revision…** restores a revision belonging to this
capture. In Camera Positions, **Open a Saved Camera Estimate…** reopens its
original UUID-named result; open the matching reference revision first. Review
it again before use. Changed frames, observations, profiles or result bytes
require correction or a new estimate. Earlier results remain available.

Use the always-available **Calibrate Again** control to start a fresh session;
earlier saved revisions remain. Start a new session after moving a camera or changing zoom, focus, resolution or
sensor crop. Optical zoom changes the lens calibration, not just the framing.
Use a lens profile calibrated at those settings, or recalibrate the lens first.
The application records manual settings; it cannot detect unreported ring or
camera movement. Repeated placements refine one fixed setup, not changing zoom.
Lens profiles can be reused when compatible, but reference sessions and reviewed
layouts currently belong to their capture. Guided transfer to another capture
is not yet available.

## Storage and Recovery

Calibration evidence lives under the capture's **reference_calibration** folder:
lossless frames and metadata, observation revisions, original estimates and
separate reviewed copies. Archive or move the entire capture through Library so
its evidence travels together. Linked lens calibration files must remain available.
Cancelling point editing can leave an unused archived frame; these are retained
for recovery and are not automatically deleted. Do not rename internal sidecars.

If a command fails, read the status message, correct the selection and retry.
Stop Operation cancels an active worker. Close prompts before discarding unsaved
observations. A missing calibration provider needs a compatible application
installation; Reload Reference Choices retries availability after correction.
