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

Use **Open Swing Editor** to select the first and last swing frames and crop the
image. Save even when keeping the full recording. When analysis already exists,
the editor offers a separate editable copy so earlier results remain available.
The wizard follows the selected copy and rechecks its evidence.

## Follow the Status and Controls

Each step opens the existing tool. Analysis pages open controls and instructions;
they do not start a job automatically. Return with **Capture Wizard** after
working in those controls, then **Refresh Status**. Inspection runs in the
background so the window remains responsive.

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

The source is `src/config/capability_connections.json`. Executable prerequisites
are separate from informational architecture arrows. Run
`python3 -m scripts.generate_capability_atlas` to regenerate the browser reference,
graph data and Mermaid diagrams; CI checks freshness.

## Current Route Scope

Single-view timing and multi-view reconstruction require separate sessions.
Guided reconstruction/model/export routes use the default triangulated match,
all views and the default observation set. Named variants and image-space
matching remain in Match; incompatible selections receive an explanation rather
than another variant's completion status. Expert projection uses the existing
comparison tool's default reconstructed cameras.

Progress fingerprints inspect bounded metadata and media size/modification time,
avoiding video decoding or hashing during navigation. They are UI invalidation
tokens, not media-integrity certificates. Replacing media while preserving its
size and modification time requires manual review. Reference alignment retains
its existing camera, clock and asset-binding checks. Body models record club
context without applying unsupported club constraints.

## Interface Qualification

The interface uses [Qt QWizard](https://doc.qt.io/qt-6/qwizard.html) with its
standard Classic style so Back/Next stay inside the application window on
Windows. Choice pages were reviewed at 760 × 610 and step pages at 660 × 560.
Qt tests cover small-window navigation, optional skipping, no-camera library
access, map validation, save/resume and stale-input explanations. These checks
qualify the UI and artifact associations, not camera hardware accuracy.
