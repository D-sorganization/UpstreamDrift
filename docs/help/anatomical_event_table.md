---
title: Anatomical Event Table
tile_id: anatomical_event_table
status: active
---

# Anatomical Event Table

## Purpose

Show the wrist, forearm, trunk and pelvis terms at address, top, pelvis reversal, impact and finish as one read-only table. The desktop dock and the web Analysis Tools page render the same event-table document through one shared view, so the cells are identical for the same bundle.

## Inputs and Method

The desktop dock opens an event-table export (`anatomical_angles.event_table.to_json`, the body `POST /api/biomechanics/anatomical-event-table` returns) with **Open Event Table JSON…**. The web page posts calibrated frames to that route. `anatomical_angles.event_table_view` turns the document into rows; a missing value shows "missing" with its reason.

## Limitations

The table formats the export; it does not detect events or change any angle. An unreadable or invalid file is reported and the table keeps what it showed; no surface substitutes sample data. Angles are software outputs, not clinical measurements.

## See Also

- [Analysis Tools](analysis_tools.md)
