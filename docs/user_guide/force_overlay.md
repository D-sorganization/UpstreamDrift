# Force and Torque Overlays User Guide

This guide details the engine-agnostic physical vector and moment overlay system (FTO-1–30, ADR-0052, #11285).

## Overview

UpstreamDrift visualizes true 3D forces and joint torques across physics engines (MuJoCo, Drake, Pinocchio, OpenSim, Simscape) and video footage compositors.
Force overlays preserve physical realism:

- Forces are drawn as arrows aligned with the vector's line of action.
- Torques and moments are drawn as circular moment arcs curving around the moment axis according to the right-hand rule.
- Unavailable or non-finite values report unavailable rather than defaulting to zero.
- All spatial coordinates follow ADR-0041 ($Z$-up world frame in meters, forces in newtons, torques in newton-meters).

---

## Palette and Wrench Kinds

Each force or torque wrench is categorized by its physical source using the authoritative palette (`FORCE_KIND_PALETTE` in `src/shared/python/force_overlay/palette.py`):

| Wrench Kind      | Hex Code  | Visual Color   | Physical Meaning                                               |
| :--------------- | :-------- | :------------- | :------------------------------------------------------------- |
| `joint_actuator` | `#E69F00` | Orange         | Active force or torque commanded by joint actuators or motors  |
| `joint_reaction` | `#CC79A7` | Reddish Purple | Constraint reaction force and torque transmitted across joints |
| `contact`        | `#009E73` | Bluish Green   | Ground reaction forces (GRF) and surface contact interactions  |
| `grip`           | `#56B4E9` | Sky Blue       | Hand-to-handle interface coupling forces and moments           |
| `external`       | `#000000` | Black          | Applied external loads or aerodynamic perturbations            |
| `gravity`        | `#999999` | Gray           | Gravitational body forces acting at segment centers of mass    |
| `muscle`         | `#D55E00` | Vermillion     | Muscle-tendon unit (MTU) line-of-action tension forces         |

Categorical wrench colors stay distinct from axial tension and compression colors.

---

## Tension and Compression Shading

Segment axial loads derived from proximal joint reactions color segments according to the shared policy in `src/shared/python/body_part_viz/`:

- **Tension:** Blue (`#0000FF`) for positive axial force ($F_{\text{axial}} > 0$).
- **Compression:** Red (`#FF0000`) for negative axial force ($F_{\text{axial}} < 0$).
- **Neutral Band:** Segments within the deadband around zero retain neutral or base mesh colors.
- **Unavailable Data:** Missing, non-finite, or uncalibrated segment forces retain base model colors and are never rendered as zero.

For details on segment axial load derivation and color scales, see [Segment Force Colors](body_part_viz/force_colors.md).

---

## Enabling Overlays in Desktop GUIs and Web

### 1. Desktop GUIs (PyQt6)

In `Model Explorer`, `Capture Rig`, `Simscape 3D Viewer`, and `Tour Matching Viewer`:

- **Force Glyphs Toggle:** Enable "Show Force Arrows" to display 3D arrow glyphs at joint and contact anchors.
- **Torque Arcs Toggle:** Enable "Show Torques" to render moment arcs around joint rotation axes.
- **Model Volumes (Shaded):** Enable shaded capsule or mesh volumes with tension/compression fill.
- **Scale Controls:** Adjust `force_scale_m_per_n` and `torque_scale_m_per_nm` sliders to scale visual glyph lengths for clear inspection.

### 2. Web Interface (React & Three.Js)

In `Scene3D.tsx` and `SimulationControls.tsx`:

- Open the **Visualization** tab in the left control sidebar.
- Toggle **Show Forces** and **Show Torques**.
- Live `GlyphSet` objects stream over WebSocket `/ws/overlays/force-torque/{model_id}` with Three.js rendering via `GlyphLayer.tsx`.

### 3. Web Video Analyzer (SVG)

In `VideoAnalyzer.tsx`:

- Toggle **Video Force Overlay** to display calibrated SVG projections aligned with video frames via `VideoForceOverlay.tsx`.
- Projected arrows and moment arcs include dark halos for high-contrast visibility against bright or noisy background footage.

---

## Video Compositing and Qualification Labels

When compositing physical overlays onto calibrated camera footage:

- Camera calibration requires intrinsic matrix $K$, rotation $R$, and translation $t$ (or `PinholeCamera`).
- The compositor generates a `VideoGlyphReceipt` recording:
  - `drawn`: Count of successfully projected and rendered glyphs.
  - `skipped_behind_camera`: Glyphs culled because anchor or tip falls behind the camera plane ($Z_{\text{cam}} \le 0$).
  - `skipped_out_of_frame`: Glyphs entirely outside image boundaries.
  - `unavailable_labels`: Joints or bodies where kinetic data was missing or non-finite.

---

## Ground Reaction Breakdown

The shared core `src/shared/python/biomechanics/ground_reaction.py` (GCV-1, #11707) turns contact wrenches into the complete ground-reaction breakdown. Every engine and surface should display these quantities instead of recomputing them.

| Quantity                        | Label                                   | Notes                                              |
| :------------------------------ | :-------------------------------------- | :------------------------------------------------- |
| Foot and net GRF                | `contact:grf_<foot>`, `contact:grf_net` | Force by the ground on the foot, drawn at the CoP  |
| Centre of pressure (CoP)        | Arrow anchor                            | On the ground plane `z = z_g`; needs `F_z >= 10 N` |
| Free moment                     | `contact:free_moment_<foot>`, `..._net` | Vertical torque about the CoP                      |
| Moment about the whole-body CoM | `contact:moment_com_<foot>`, `..._net`  | `M_O - c x F`; the net equals the sum of the feet  |

Behavior to know when reading the overlays:

- Below 10 N vertical force the CoP and free moment are unavailable (not zero); the force arrow remains, anchored at the contact centroid.
- A foot with no active contact shows nothing and reports zero force.
- The net free moment is computed from the net wrench about the net CoP. It is generally not the sum of the two foot free moments, because the foot CoPs differ and shear forces at those offset points contribute a vertical moment. The two agree when the CoPs coincide.
- Time series use `GroundReactionSeries` (NaN marks unavailable values, `to_dataframe()` for export).

---

## Known Limits per Engine

- **MuJoCo:** Native offscreen rendering supports true 3D geoms (`add_glyphs_to_scene`) and footage compositing with segmentation masks. GPU rasterization requires EGL or an active display server.
- **Drake & Pinocchio:** Provider frames are exported headlessly via the Matplotlib 3D glyph renderer (`draw_glyphs_3d`); MeshCat browser screenshots are not supported in headless CI.
- **OpenSim:** Animated force playback operates on `.mot` / `.sto` time-series files via `force_overlay/playback.py`.
- **Simscape Multibody:** Replay relies on pre-recorded dataset CSVs; live forward integration is governed by compiled block budgets (MMR-04).
