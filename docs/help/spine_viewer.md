---
title: Spine Viewer
tile_id: spine_viewer
status: active
---

# Spine Viewer

## Purpose

Display the per-level spinal loads of a finished segmental-spine study. The viewer is read-only: it opens a study receipt, verifies the hash of its artifact and draws a heatmap of one load component over time and spinal level. Nothing is recomputed.

## Inputs

| Input         | Units | Description                                                                                     |
| ------------- | ----- | ----------------------------------------------------------------------------------------------- |
| Study receipt | file  | JSON receipt written by the segmental-spine study runner; opened with **Open Study Receipt...** |
| Run artifact  | file  | Read from the receipt's folder; it must match the SHA-256 recorded in the receipt               |
| Quantity      | label | One of `compression`, `tension`, `ap_shear` or `ml_shear`, chosen in the **Quantity** list      |

## Outputs

| Output              | Units | Description                                                                                                  |
| ------------------- | ----- | ------------------------------------------------------------------------------------------------------------ |
| Status banner       | text  | `QUALIFIED` or `UNQUALIFIED`, the run verdict, and any status reasons; always visible                        |
| Distribution source | text  | How the trunk motion was split across levels (`distribution_source` from the receipt); always visible        |
| Load heatmap        | N     | The chosen joint-reaction force component per level and time; the figure title repeats the scientific status |

## Method

`src.tools.spine_viewer.core.open_study` loads the receipt with `load_verified_run`, which checks the artifact's SHA-256, and wraps the joint-reaction wrench in a `SpinalLoadSeries`. The heatmap comes from the shared `spine.viz_data.load_heatmap` and `spine.plots.plot_load_heatmap`. Loads follow ADR-0055: the force acts on the superior vertebra, is expressed in the inferior vertebra's ISB frame, and compression is +F_y; the tension view is the negative of compression.

## Limitations

The status is fail-closed: a study shows `QUALIFIED` only when its receipt declares a qualified scientific status and its run verdict is `pass`. A passing run is not a qualification. A missing, tampered or inconsistent artifact is reported as an error and nothing is drawn. The loads are model outputs from OpenSim JointReaction analysis, not measured spinal forces, and they depend on the distribution source named in the banner.

## See Also

- [OpenSim Golf](opensim_golf.md)
- [Analysis Tools](analysis_tools.md)
