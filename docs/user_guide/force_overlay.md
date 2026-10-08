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

## Grip Wrench: Per-Hand Loading, Midpoint Net Force and Couple

The grip overlays come from one shared definition,
`src/shared/python/biomechanics/grip_wrench.py` (GCV-7, #11713). Every wrench
is what the **hand exerts on the club**, in the world frame, in SI units.

With hand grip points `r_L`, `r_R`, forces `F_L`, `F_R` and free torques
`tau_L`, `tau_R`, and the grip midpoint `r_M = (r_L + r_R) / 2`:

```
R   = F_L + F_R                                  drawn at r_M  (grip:net_midpoint)
M_M = (r_L - r_M) x F_L + (r_R - r_M) x F_R      contact-force moment
      + tau_L + tau_R                            applied free torque
                                                 total: grip:couple_midpoint
```

```
   F_L  (grip:hand_left)            F_R  (grip:hand_right)
    \                                /
     o r_L ---------- o r_M ---------- o r_R
                      |
        MOF_L = (r_L - r_M) x F_L    (grip:mof_left)
        MOF_R = (r_R - r_M) x F_R    (grip:mof_right)

 couple M_M  =  [MOF_L + MOF_R]  +  [tau_L + tau_R]
                 contact-force      applied free
                 moment             torque
```

- The decomposition is exposed as `contact_force_moment_nm` and
  `applied_free_torque_nm`; their sum is the equivalent couple. Only a pure
  couple (`R = 0`) has the same moment about every reference point; otherwise
  move the wrench with `GripAnalysis.net_wrench_at(point)`.
- The club-local components are `R^T M_M` (`couple_local_nm`); `about_axis`
  gives the component about any axis, such as the swing-plane normal.
- `split_method` (`constraint_multiplier`, `efc_force`, `allocation`, `logged`
  or `unavailable`) records how the left/right split was obtained. Show it
  next to the per-hand arrows: the split is solver-regularised.
- A missing hand, or a missing free torque, yields `None` plus
  `unavailable_reason`, never zero; `GripSeries` uses NaN and `to_dataframe()`.

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
- **Scale Mode:** Choose how force arrows are sized. `Fixed (Slider)` uses the slider scale. `Body Weight` draws one body weight (body mass times 9.80665) as the reference length, 0.5 m by default, so a 3 BW ground reaction is 1.5 m long whatever the golfer's mass. `Series Peak` draws the given peak force as the reference length. Native export defaults to `Body Weight` using the model mass.
- **Clamped Arrows:** An arrow longer than the maximum length is shortened and drawn with a second head (double tip, or a white marker in the MuJoCo viewport). The legend reports how many arrows are clamped.
- **Group Toggles:** Per-Foot GRF, Net GRF, Free Moment, Moment About CoM, Contact Points, Grip Per Hand, Grip Net, Grip Couple and Grip MOF. Contact Points (raw per-sphere contacts) and Moment About CoM are off by default.

### 2. Web Interface (React & Three.Js)

In `Scene3D.tsx` and `SimulationControls.tsx`:

- Open the **Visualization** tab in the left control sidebar.
- Toggle **Show Forces** and **Show Torques**.
- Use the same **Scale Mode**, body mass or peak force, length per reference and **Groups** controls as the desktop Visualization tab.
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

## Known Limits per Engine

- **MuJoCo:** Native offscreen rendering supports true 3D geoms (`add_glyphs_to_scene`) and footage compositing with segmentation masks. GPU rasterization requires EGL or an active display server.
- **Drake & Pinocchio:** Provider frames are exported headlessly via the Matplotlib 3D glyph renderer (`draw_glyphs_3d`); MeshCat browser screenshots are not supported in headless CI.
- **OpenSim:** Animated force playback operates on `.mot` / `.sto` time-series files via `force_overlay/playback.py`.
- **Simscape Multibody:** Replay relies on pre-recorded dataset CSVs; live forward integration is governed by compiled block budgets (MMR-04).
