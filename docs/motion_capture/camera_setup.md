# Camera Setup

Open **Camera Setup** in the Capture Rig header, or **Set Up Cameras to Record**
on the capture wizard's first page. Importing videos remains available without
camera hardware or a saved setup.

## Create a Setup

1. Choose **Scan for Cameras**. Live discovery supports the ELP USB rig on Windows.
   Scanning runs in the background and does not start a recording stream.
2. Give each camera a recognizable view name, such as `face_on`, `down_line`, or
   `overhead`. Names start with a letter and may contain letters, digits,
   underscores and hyphens. Each view and camera must be unique.
3. Choose the resolution and frame rate. Presets can be edited using
   `1280x720@120:MJPG` syntax. A selected mode still needs verification on hardware.
4. Add placement notes and choose **Check Connections**. This checks discovered
   identities, capture indices and USB topology; it does not certify stream modes
   or calibration accuracy.
5. Choose **Save and Use**, then run **Plan Check** and **Preview** in Capture Rig.
   Confirm the expected views before recording a take.

## Reuse and Recover

**Load Plan** opens a saved JSON plan for editing, including offline editing.
Every save creates a separate revision under `camera-plans` in the configured
capture library. The selected revision is remembered after restarting the app.
Selecting a setup clears previous temporary mode and camera-control overrides;
it preserves the selected capture and recordings. Setup changes are unavailable
while recording, counting down or running another capture command.

**Cancel Scan** discards discovery results and preserves your choices. An active
operating-system probe may take up to its timeout to stop; scanning becomes
available again when it finishes. If no cameras are found, check USB connections
and permissions, scan again, load a saved plan, or import existing video. Saved
camera identities remain visible when disconnected; they are never silently
reassigned to another camera.

## Review Calibration

A saved setup describes camera identities and requested recording modes. It is
separate from calibration. Use **Review or Repeat Calibration** when cameras move
or lens, optical zoom, focus, resolution or crop settings change. Matching view
names alone cannot establish that an earlier calibration remains valid.
