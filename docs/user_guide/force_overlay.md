# Force and Torque Overlay User Guide

The Force and Torque Overlay system ([ADR-0052](file:///home/dieterolson/Repositories/UpstreamDrift-worktrees/antigravity-11315/docs/adr/0052-force-torque-overlay-contract.md), epic [#11285](https://github.com/D-sorganization/UpstreamDrift/issues/11285)) provides renderer-neutral, engine-agnostic 3D physical vector, moment arc, and segment load visualizations across all UpstreamDrift simulation backends, native viewers, and web interfaces.

---

## 1. Wrench Kinds and Color Palette

Overlay wrenches are classified into standardized semantic categories with distinct, colorblind-accessible categorical hues. Missing vector halves are serialized as `None`/`null` and omitted rather than zero-filled.

| Wrench Kind | Palette Swatch | Hex Code | Meaning and Semantic Role |
| :--- | :--- | :--- | :--- |
| `joint_actuator` | <span style="display:inline-block;width:14px;height:14px;background:#E69F00;border:1px solid #333;border-radius:2px;"></span> Orange | `#E69F00` | Internal actuator driving torques/forces applied across joints. |
| `joint_reaction` | <span style="display:inline-block;width:14px;height:14px;background:#CC79A7;border:1px solid #333;border-radius:2px;"></span> Purple | `#CC79A7` | Constraint reaction forces and moments transmitted through joint articulations. |
| `contact` | <span style="display:inline-block;width:14px;height:14px;background:#009E73;border:1px solid #333;border-radius:2px;"></span> Green | `#009E73` | Normal and frictional environmental contact forces (e.g. ground reaction, ball impact). |
| `grip` | <span style="display:inline-block;width:14px;height:14px;background:#56B4E9;border:1px solid #333;border-radius:2px;"></span> Sky Blue | `#56B4E9` | Multi-axis forces and moments measured or modeled at hand-grip interfaces. |
| `external` | <span style="display:inline-block;width:14px;height:14px;background:#000000;border:1px solid #333;border-radius:2px;"></span> Black | `#000000` | Unmodeled external disturbances or applied test perturbations. |
| `gravity` | <span style="display:inline-block;width:14px;height:14px;background:#999999;border:1px solid #333;border-radius:2px;"></span> Grey | `#999999` | Gravitational body forces acting at segment centers of mass. |
| `muscle` | <span style="display:inline-block;width:14px;height:14px;background:#D55E00;border:1px solid #333;border-radius:2px;"></span> Vermillion | `#D55E00` | Line-of-action active contractile forces along muscular paths. |

Categorical colors are defined authoritatively in [`src/shared/python/force_overlay/palette.py`](file:///home/dieterolson/Repositories/UpstreamDrift-worktrees/antigravity-11315/src/shared/python/force_overlay/palette.py).

---

## 2. Tension and Compression Conventions

Segment axial loads represent normal forces resolved along the proximal-to-distal anatomical axis of a segment:

- **Tension ($F_{\text{axial}} > 0$):** Highlighted in blue (`#0000FF` / `#0072B2`).
- **Compression ($F_{\text{axial}} < 0$):** Highlighted in red (`#FF0000` / `#D55E00`).
- **Neutral Band / Stale / Absent Data:** Renders in the unshaded default material color of the segment. Missing loads are never zero-filled.

For detailed segment mesh coloring and scale configuration, see [Segment Force Colors](file:///home/dieterolson/Repositories/UpstreamDrift-worktrees/antigravity-11315/docs/user_guide/body_part_viz/force_colors.md).

---

## 3. Controlling Overlays in GUIs and Web

### MuJoCo Viewer
- **Menu:** Navigate to **View → Force Overlays** or toggle via keyboard shortcut.
- **Controls:** Individual checkable categories for Actuators, Reactions, Contacts, and Gravity.
- **Rendering:** Uses offscreen/native `add_glyphs_to_scene` with cylinder shafts, conical tips, and circular torque arcs.

### Drake and Pinocchio MeshCat
- **Menu:** In the viewer control tree under **View → Force Overlays**.
- **Controls:** Sliders for force scale ($m/N$), torque scale ($m/(N\cdot m)$), and category filters.
- **Rendering:** Interfaced through `MeshcatGlyphRenderer` using cached geometry transforms.

### C3D / Simscape Viewer
- **Menu:** View menu provides **Force Overlays** along with dedicated **Grip Forces** toggle.
- **Controls:** Loaded automatically when opening telemetry CSVs (`load_simscape_force_series`).

### Web Three.Js (`Scene3D`)
- **UI:** A collapsible **Force / Torque Overlay** panel is embedded in the 3D viewport.
- **WebSocket:** Receives JSON-serialized `ForceTorqueFrame` streams (`force-torque-frame-v1`).
- **Toggles:** Separate toggles for Force Arrows, Torque Arcs, and Legend overlay.

---

## 4. Video Overlays and Qualification Labels

When projecting forces onto video footage using OpenCV (`draw_glyphs_on_frame`):
- **Camera Calibration:** Camera must provide projection under world frame convention `adr0041` ([ADR-0041](file:///home/dieterolson/Repositories/UpstreamDrift-worktrees/antigravity-11315/docs/adr/0041-coordinate-frames.md)).
- **Occlusion Handling:** Glyphs behind the camera plane or outside the image bounds are skipped cleanly and tallied in `VideoGlyphReceipt`.
- **Legend & Qualification:** An on-frame legend box renders the active force reference scale, engine provenance, and any qualification string (e.g. `Note: Simulated kinematics`).

---

## 5. Engine Capabilities and Known Limits

| Engine | Status | Provider Mechanism | Known Limits / Considerations |
| :--- | :--- | :--- | :--- |
| **MuJoCo** | Supported (Live) | `cfrc_int` constraint reaction queries | Tendon spatial routing is omitted from `cfrc_int`; see native MuJoCo docs. |
| **Drake** | Supported (Live) | `MultibodyPlant` reaction output ports | Requires populated geometry visualizer inspection for segment links. |
| **Pinocchio** | Supported (Live) | RNEA recursive dynamics solver | Axial reactions require mapped proximal joint coordinate frames. |
| **OpenSim** | Supported (Recorded) | `opensim_force_recording.py` | Headless execution operates via recorded playback series (`render_force_playback`). |
| **Simscape** | Supported (Recorded) | Multi-channel CSV telemetry ingestion | Requires synchronized time coordinates matching motion capture. |

---

## 6. Shared Infrastructure and Reference Links

- [Force Overlay Contracts and Schemas](file:///home/dieterolson/Repositories/UpstreamDrift-worktrees/antigravity-11315/docs/agents/shared-infrastructure.md#force-and-torque-overlay-pipeline)
- [Segment Force Colors User Guide](file:///home/dieterolson/Repositories/UpstreamDrift-worktrees/antigravity-11315/docs/user_guide/body_part_viz/force_colors.md)
- [ADR-0052: Engine-Agnostic Force/Torque Visual Overlay Contract](file:///home/dieterolson/Repositories/UpstreamDrift-worktrees/antigravity-11315/docs/adr/0052-force-torque-overlay-contract.md)
