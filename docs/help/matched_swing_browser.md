---
title: Matched Swing Browser
tile_id: matched_swing_browser
status: active
---

# Matched Swing Browser

## Purpose

The Matched Swing Browser catalogs, filters, and previews certified swing trajectories across the entire fleet of physics engines and biomechanical humanoids. It serves as the primary library for exploring precomputed forward-dynamics and inverse-kinematics solutions.

## Inputs

| Input | Units | Description |
| --- | --- | --- |
| Filter Criteria | metadata | Engine selection, player category, club type, swing speed |
| Search Query | string | Identifier, player name, or experiment tags |
| Sort Attribute | key | Peak speed, impact consistency, or parity score |

## Outputs

| Output | Units | Description |
| --- | --- | --- |
| Matched Swing Records | table | Qualified swing trajectories matching search criteria |
| Summary Metrics | m/s, deg | Club speed, ball speed, smash factor, launch metrics |
| Artifact Provenance | hash | SHA-256 hashes of models, datasets, and solver receipts |

## Method

Scans the local and shared model repository manifests and qualified baseline receipt caches. Validates data integrity against schema contracts and surfaces qualified status badges.

## Limitations

Displays precomputed records; running new optimizations requires launching the respective engine or motion matching workbench.

## See Also

- [Tour Matching Viewer](tour_matching_viewer.md)
- [Motion Matching](motion_matching.md)
- [Data Explorer](data_explorer.md)
