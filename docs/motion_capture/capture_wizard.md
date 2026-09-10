# Guided Swing Capture

Open **Capture Wizard** in the Capture Rig header. Choose your outcome: edit a
swing, add coaching drawings, compare expert video, analyze one camera,
reconstruct multiple cameras, fit a body model, export motion, or project a
reference into reconstructed cameras. Select several compatible outcomes to
share preparation steps.

## Start With a Recognizable Capture

Open **Capture Library** to import videos or select an existing take. A capture
groups the camera recordings for one take; give it a recognizable title and
swing notes. Every wizard step displays that title and its stable capture ID.
Importing and editing existing videos does not require connected cameras.

To record a new take, choose **Set Up Cameras to Record** on the first wizard
page, or **Camera Setup** in the main header. Name views, select cameras and
save a reusable setup without editing JSON. Follow the
[camera setup guide](camera_setup.md) before Preview and Plan Check.

Use **Open Swing Editor** to select the first and last swing frames and crop the
image. Save even when keeping the full recording. When analysis already exists,
the editor offers a separate editable copy so earlier results remain available.
The wizard follows the selected copy and rechecks its evidence.

## Follow the Status and Controls

Each step opens the existing tool. Analysis pages open controls and instructions;
they do not start a job automatically. Return with **Capture Wizard** after
working in those controls, then **Refresh Status**. Inspection runs in the
background so the window remains responsive.

When a prerequisite needs attention, its named **Go to …** button takes you
directly to that wizard page. Complete or review the input, then return to the
next step. These links disappear once the input is satisfied and pause while an
operation is running. A missing reviewed calibration blocks calibrated analysis;
editing-only routes remain available without loading calibration evidence.

- **Ready for Your Action:** inputs are available; open the tool and complete the step.
- **Needs Attention:** the explanation identifies missing or changed inputs.
- **Available / Reviewed:** the required saved artifact or reviewed input is present.
  This does not certify a detector, model fit or physical measurement.
- **Skipped:** you explicitly skipped an optional step. My Clubs can be skipped
  when equipment information is unknown.

Next becomes available when the current step has its required evidence. Back,
Cancel and closing a window preserve recordings and saved artifacts. The existing
activity log reports running, failed, stopped and completed jobs. **Help** opens
searchable workflow instructions; **Capture Status** in the main window shows
detector, model and capture associations.

## Save and Resume

**Save and Close** stores a bookmark in the selected capture. Reopen the wizard
and choose **Resume This Capture's Saved Workflow**. The bookmark retains the
outcomes, step and optional skips. Changed inputs, model/match settings or
capability metadata require reviewing the outcomes again. Another capture's
bookmark is rejected. Corrupt metadata is reported and retained.

Calibration confirmation is renewed after closing the wizard. Open **Review or
Repeat Calibration**, verify the recorded camera identities and image sizes,
and confirm lens, optical zoom, focus and sensor/crop mode. Select compatible
revisions or recalibrate. Optical zoom can change lens intrinsics; matching image
dimensions alone does not make an older calibration compatible. This review does
not prove physical camera placement or reconstruction accuracy. Everyday-object
and moving-placement calibration remain tracked in #9897.

## Plan From the Capability Map

In the [capability atlas](../../ui/public/capability-atlas/index.html#capture-goals),
select outcomes and **Save Selected Capture Plan**. In the desktop wizard, use
**Open a Plan from the Capability Map** to open that JSON file. The app validates
goal IDs and the map revision before changing your choices. A plan contains
selections, not commands or recording paths.

## Current Route Scope

In Calibration, **Paper / Ruler References…** opens the
[common-reference workflow](common_reference_calibration.md). After using a
reviewed camera layout, return to the wizard and refresh status. Paper placement
estimation requires a compatible lens calibration for the recorded zoom and
focus; it does not replace lens calibration or joint detection.

Single-view timing and multi-view reconstruction require separate sessions.
Guided reconstruction/model/export routes use the default triangulated match,
all views and the default observation set. Named variants and image-space
matching remain in Match; incompatible selections receive an explanation rather
than another variant's completion status. Expert projection uses the existing
comparison tool's default reconstructed cameras.

After changing your selection, camera settings or model choices, refresh status
and review the affected analysis steps. For video edited outside the application,
import the edited file as a new capture. Body models retain club information as
context; they currently do not fit or constrain a club segment.

## Example: Prepare a Swing for Review

1. Choose **Trim and Crop a Swing** in Capture Wizard.
2. Open **Capture Library**, import the recording and give the take a clear title
   and swing notes. Cameras and calibration are not needed for imported video.
3. Open **Swing Editor**, mark the first and last swing frames, adjust the crop
   if needed, and save. The original video remains available.
4. Return to the wizard and choose **Refresh Status**. The saved selection is
   recognized; simply visiting the editor does not complete this step.
5. Use **Save and Close**. Later, select the same take in Library and choose
   **Resume This Capture's Saved Workflow** in the wizard.

## Example: Compare a Player With an Instructor Reference

1. Choose **Compare with an Expert Video**. You can also select **Trim and Crop
   a Swing** and **Add Coaching Lines and Shapes** for the same take.
2. Select the player's capture and save its swing selection.
3. Import the expert video in the reference library, then open **Expert
   Comparison** and select the player's camera view and the reference.
4. Align the swing events with the time controls, adjust visibility or opacity,
   and save the comparison. Refresh the wizard to see the saved alignment.
5. If you selected coaching drawings, open that step, add the reference shapes
   and save them. Use the comparison or drawing tool's export controls when
   preparing a video or still for the player.

A video comparison is an image alignment. It does not reconstruct the expert's
3-D swing or create a new camera angle. For that workflow, choose **Project an
Expert into Reconstructed Cameras** and follow its additional requirements.

## Example: Prepare Calibrated Body-Model Analysis

1. Choose **Fit a Body Model to a Reconstruction**. Import synchronized camera
   recordings or use **Camera Setup** to prepare a new take.
2. Review **My Clubs** and assign the club used for this swing. Enter measurements
   you know and leave the rest unknown. A saved capture keeps its own club
   snapshot even if the player's bag is edited later.
3. Open **Review or Repeat Calibration**. Confirm the actual camera identities,
   lens, optical zoom, focus and image settings. Use compatible profiles or
   recalibrate; the [common-reference guide](common_reference_calibration.md)
   explains paper/ruler placements and their limits.
4. Save the swing selection, run the selected detector from the analysis
   controls, and review the observations before reconstructing. The wizard's
   **Go to …** links return you to missing prerequisites.
5. Run reconstruction and body-model fitting using the existing controls.
   Return to the wizard and refresh status after each job. An unavailable
   runtime or failed job needs attention; opening a page does not run it.
6. Save your workflow before leaving. On reopening it, review calibration again.
   Changed zoom or camera settings require renewed review or recalibration;
   editing and video comparison remain available independently.

These steps organize the workflow. A completed software job does not by itself
establish physical camera accuracy or the quality of a model fit. Review the
result diagnostics before using measurements.

## Interface Qualification

The interface uses [Qt QWizard](https://doc.qt.io/qt-6/qwizard.html) with its
standard Classic style so Back/Next stay inside the application window on
Windows. Choice pages were reviewed at 760 × 610 and step pages at 660 × 560.
Qt tests cover small-window navigation, optional skipping, no-camera library
access, map validation, save/resume and stale-input explanations. These checks
qualify the UI and artifact associations, not camera hardware accuracy.
