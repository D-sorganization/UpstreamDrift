---
title: Anatomical Event Table
tile_id: anatomical_event_table
status: active
---

# Anatomical Event Table

## Purpose

Show the wrist, forearm, trunk and pelvis terms at the swing events (address, top, pelvis reversal, impact and finish) as one read-only table. The desktop dock and the web Analysis Tools page render the same event-table document through one shared view, so the cells are identical for the same bundle.

## Inputs

| Input                | Units | Description                                                                                                                         |
| -------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Event table export   | JSON  | Desktop: an event-table document from `anatomical_angles.event_table.to_json`, the body `POST /api/biomechanics/anatomical-event-table` returns |
| Calibrated frames    | JSON  | Web: the calibrated frame bundle with event indices and provenance, posted to the same route                                        |

## Outputs

| Output | Units | Description                                                                 |
| ------ | ----- | --------------------------------------------------------------------------- |
| Value  | deg   | The term at the event, or "missing" with its reason as tooltip and Note     |
| Note   | text  | Why a value is missing, or the observability of a measured value            |

## Method

`anatomical_angles.event_table` samples the anatomical terms at the detected events and exports the document; `anatomical_angles.event_table_view` turns it into the rows both surfaces show. The desktop panel opens a saved export with **Open Event Table JSON…**; an unreadable or invalid file is reported and the table keeps what it showed.

## Limitations

The table formats the export; it does not detect events or change any angle. Neither surface substitutes sample data. Angles are software outputs from calibrated frames, not clinical measurements.

## See Also

- [Analysis Tools](analysis_tools.md)
- [Grip Wrench Plots](grip_wrench_plots.md)
