---
title: Necromatcher
tile_id: necromatcher
status: partial
---

# Necromatcher

## Purpose

Recall historical players, swings, source captures, model candidates and authored
joint torque profiles from one local persistent library. Open the Necromatcher
tile, select a player and swing, then select a capture and move the frame slider.
The image is the stored source PNG. Its timestamp belongs to the source video;
unknown physical time and missing landmarks remain explicit.

## Library and Versions

The desktop and local web server use the same user configuration directory.
Set `NECROMATCHER_LIBRARY_ROOT` before launch to choose another library.
Player, swing and version IDs are permanent; existing versions cannot be replaced.
Both hosts can import a completed historical capture directory, a native model
candidate or a torque profile using a local filesystem path. Choose Import
Version in the desktop view or use the version form in the web view.

Model imports need an engine name and an ordered list of degrees of freedom.
Torque profiles use `necromatcher/torque-profile/1`, physical seconds and N\*m,
and must reference the exact saved model and its degree-of-freedom order.
They represent authored controls; saving them does not infer historical forces.

## Export and Qualification

Export Swing creates a portable ZIP with player/swing identities, asset versions
and hashes. The export verifies the packaged bytes and preserves qualification
metadata. A saved model is an unqualified candidate until actual native fitting,
reprojection checks and dynamics acceptance succeed. Source capture observations
alone cannot qualify a historical 3D reconstruction or simulator replay.

## Current Limits

Native and web review support player/swing creation, immutable version imports,
original imagery, image-space landmark overlays and package export.
Historical model fitting and accepted simulation handoffs are tracked in #11235.
See [Historical Capture Procedure](../development/historical-capture-procedure.md)
for source provenance, capture procedures and the Tiger/Hogan processing record.
