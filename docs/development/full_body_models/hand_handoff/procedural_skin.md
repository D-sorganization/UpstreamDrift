# Handoff: Lofted Procedural Skin Lane (PR #15146)

**Branch:** `cursor/hand-procedural-skin-ff59`  
**Child issue:** #15145 (Refs only — epic parent #15150)  
**Lane id:** `bc-58c47851`  
**Last cloud agent stop:** iteration 2 (`7e7c1848`); visual **fails** epic quality bar — do not merge for appearance.

Hand work continues on subscription fleet agents (Claude Code on Brick, Antigravity on DeskComputer) per Dieter (2026-11-14 ~8:52 PM PT). This file is the primary context source for a cold-start agent.

---

## (a) Goal and quality bar

**Goal:** Register the legacy **16-capsule** grip hand as `hand_backends/capsule.py` (unchanged behaviour) and add **`procedural_skin`**: anatomical-appearing skin built as **lofted surfaces** over the same `GripSpec` / joint inputs—not the implicit SDF sculpted track (`bc-dd40e13d`). Provide a **backend-neutral comparison harness** so Shadow, Allegro, MyoHand, skinned human, SDF, etc. can be compared side by side.

**Authoritative quality bar:** [Epic #15150 — Human hand model on the club](https://github.com/D-sorganization/UpstreamDrift/issues/15150) (child of grip epic #15122). Summary:

1. Renders: lead, trail, both; palm / back / side; shaft roll 0° and ±20°; key + fill; hand ≈70% of ≥1024 px frame; **club visible**.
2. Fingers wrap grip within tolerance settings; mirroring holds.
3. Anthropometric ratios vs **settings** within tolerance.
4. Human checklist: reads as a hand on a club (separate digits, cupped palm, knuckle line—not tubes or blobs).
5. Licences recorded for third-party assets (this lane: no external mesh assets).

**Track constraint:** Keep **lofted** geometry (Catmull–Rom sweeps, rounded palm slab). Do **not** own whole-hand marching cubes in this module—that is the SDF lane.

---

## (b) Current state

### Layout (`src/shared/python/model_appearance/hand_backends/`)

| File                  | Role                                                                                                        |
| --------------------- | ----------------------------------------------------------------------------------------------------------- |
| `registry.py`         | `register_backend`, `build_hand_visual`, `iter_registered_backends` — harness discovers backends here only. |
| `capsule.py`          | Thin wrapper around legacy `build_grip_hand_mesh`.                                                          |
| `procedural_skin.py`  | **Work in progress** lofted skin: finger tubes + `_palm_cup_mesh` + thenar + thumb.                         |
| `contract.py`         | `HandJointAngles`, `HandVisualResult`, `hand_mesh_with_shaft_roll`.                                         |
| `settings.py`         | `HandVisualSettings`, `HandAnthropometryReference`, tolerances, materials.                                  |
| `metrics.py`          | Penetration fractions, `palm_arch_angle_deg`, `anthropometric_vs_reference`.                                |
| `preview.py`          | Interim renderer: dual-light shading, `frame_hand_viewport`, solid `grip_shaft_mesh`.                       |
| `placement.py`        | `mesh_in_shared_visual_frame` — trail axial offset vs lead for dual-hand renders.                           |
| `skeleton_extract.py` | Joint polylines from capsule bake (`finger_polyline`, `thumb_polyline`, `uniform_scale_skeleton`).          |
| `sdf_meshing.py`      | SDF/tube helpers for **SDF sibling lane**; procedural_skin uses `tube_mesh_from_polyline` only.             |
| `README.md`           | Short parallel-track notes.                                                                                 |

### Key entry points

- `build_hand_visual(grip_spec, side, joint_angles, grip_radius_m=..., settings=...)` per backend module and via `registry.build_hand_visual(backend_id, ...)`.
- `procedural_skin.build_hand_visual`: builds capsule skeleton → scales polylines → tubes + palm shell → `concat_meshes` → optional `_enforce_minimum_radial_distance` (shaft clearance).

### Settings (no literals in tests for acceptance bands—extend `HandAnthropometryReference` / `HandVisualSettings`)

`HandAnthropometryReference` (defaults in `settings.py`):

- `reference_hand_length_m` = 0.19
- `reference_palm_width_m` = 0.085
- `finger_diameter_base_m` = 0.018, `finger_diameter_tip_m` = 0.014
- Properties: `finger_radius_base_m`, `finger_radius_tip_m` (= half diameter)

`HandVisualSettings`:

- Penetration: `shaft_penetration_tolerance_m`, `finger_penetration_tolerance_m`, `max_shaft_penetration_fraction`, `max_finger_penetration_fraction`
- Mesh: `procedural_skin_sides`, `lead_skin_material` / `trail_skin_material`
- `DEFAULT_VISUAL_GRIP_RADIUS_M` = 0.014 (render scripts)

### Tests

```bash
cd /path/to/UpstreamDrift
python3 -m pytest tests/unit/model_appearance/hand_backends/ -q --timeout=120
```

| Test file                         | What it covers                                                           |
| --------------------------------- | ------------------------------------------------------------------------ |
| `test_hand_backends_contract.py`  | Registry, capsule ≡ legacy vertex count, shaft roll, left/right mirror   |
| `test_hand_visual_penetration.py` | Shaft/finger penetration vs settings for `capsule` and `procedural_skin` |

**Status at handoff:** 7/7 passing. Penetration tests pass; **no** scale/topology/pose acceptance tests yet (required next).

### Evidence scripts

```bash
export QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg
python3 scripts/render_hand_backend_comparison.py
# → docs/development/full_body_models/evidence/hand_backends/contact_sheet.png
# → docs/development/full_body_models/evidence/hand_backends/metrics.json

python3 scripts/render_hand_backend_roll_evidence.py --backend procedural_skin
# → procedural_skin_{lead,trail,both}_hands_roll_{-20,0,+20}.png in same evidence dir
```

Comparison harness iterates `iter_registered_backends()` — **do not** hard-code backend list.

Lint/format (CI):

```bash
python3 -m ruff check src/shared/python/model_appearance/hand_backends scripts/render_hand_backend_*.py
python3 -m ruff format --check .
```

---

## (c) What worked and what failed

### Worked

- **Registry + harness:** Side-by-side contact sheet and per-backend `metrics.json` for all registered backends.
- **Capsule baseline:** `capsule.py` preserves legacy mesh; regression on vertex count.
- **Contract tests:** Shaft roll about z; lead/trail mirroring; penetration within **numerical** tolerances.
- **Interim render:** Dual skin tones (`skin_2` / `skin_3`), key/fill lights, solid grip mesh, combined bounds framing helper (`frame_hand_viewport` at `fill_fraction=0.70`).
- **Lofted finger tubes:** `tube_mesh_from_polyline` with radii from `anthropometry.finger_radius_*` — **in isolation**, proximal radius ≈ 9 mm (18 mm diameter) on middle-finger spline from capsule bake (verified in REPL at `7e7c1848`).
- **Dead end documented:** Monolithic SDF union + marching cubes (whole hand) → rectangular palm blob, floating finger islands; abandoned for this lane.

### Failed (visual and structural)

**Barb review of `7e7c1848` / `procedural_skin_lead_roll_+0.png` (worse than internal notes):**

- Fingers read as **thin spaghetti** (~4–6 mm apparent width in the image), scattered around the grip—not 17–20 mm proximal.
- **No identifiable palm**; fingers aim in **inconsistent directions**.
- Frame mostly **white space** (~15% hand fill vs 70% target).

**Root causes (engineering, not just shading):**

1. **Wrong palm topology:** `_palm_cup_mesh` builds an **annular patch in cylindrical (ρ, θ, z) around the grip axis**, not a **metacarpal slab** (rounded box with transverse/longitudinal arch, 25–30 mm thick, thenar/hypothenar). Knuckle line is not a distal palm edge.
2. **Concatenated shells, not integrated hand:** Fingers are独立 tubes; palm/thenar/thumb are separate meshes with no lofted blend at MCPs.
3. **`uniform_scale_skeleton`:** Scales all finger polylines about **one centroid** after an extra target-length scaling pass—can **distort** curl direction and knuckle spacing relative to Dieter’s flat-then-curl placement in `grip_hand_mesh`.
4. **`_enforce_minimum_radial_distance`:** Radially pushes any vertex with ρ < grip inner floor—can **fight** palmar geometry and contribute to “floating spaghetti” around the cylinder.
5. **Metrics vs appearance:** `anthropometric_vs_reference` on **mesh AABB** can mark `palm_width_within_tolerance` while the image still fails checklist #5. `hand_length_within_tolerance` often **false** (~0.15 m span vs 0.19 m reference at `7e7c1848`).
6. **Pose:** Procedural skin **reuses curled** capsule joint polylines only; it does **not** implement epic’s “flat hand → curl MCP/PIP/DIP to contact” poser (pending `bc-c9c36777`).
7. **Rendering:** `render_hand_backend_roll_evidence.py` uses `fig.tight_layout()` and 3-panel strips—framing can leave large margins even when `frame_hand_viewport` is correct per panel.

**Do not repeat:**

- Whole-hand SDF merge in `procedural_skin.py`.
- Palm as a thin cylindrical sheet hugging `grip_radius_m`.
- Scaling skeleton about global centroid without palm-anchored frame.
- Claiming done when penetration tests pass but PNGs fail § checklist.

---

## (d) Next steps (priority order) with acceptance tests

> Barb iteration-3 spec (fleet agent should implement TDD **red → green**). Cloud lane stopped before this rebuild.

### 1. Scale tests first (RED, then fix geometry)

Add `tests/unit/model_appearance/hand_backends/test_procedural_skin_scale.py` (or extend settings with explicit bands):

| Quantity                           | Band (from settings / epic) |
| ---------------------------------- | --------------------------- |
| Proximal phalanx tube **diameter** | 17–20 mm                    |
| Distal tip **diameter**            | 13–16 mm                    |
| Palm width                         | 0.080–0.090 m               |
| Palm length (proximal–distal)      | 0.095–0.11 m                |
| Total hand length                  | 0.18–0.20 m                 |

Measure from **built mesh** (sample cross-sections at MCP planes, palm slab extents)—not bbox alone. Fix: use `settings.anthropometry` radii consistently; build palm in **hand frame** at true width/length; remove erroneous double-scaling; audit `_enforce_minimum_radial_distance`.

**Acceptance:** all asserts pass on `GripSpec.legacy()` lead + trail at roll 0.

### 2. Topology

Implement **one palm slab**: rounded box, transverse + longitudinal arch, thickness 25–30 mm, thenar/hypothenar bulges. Fingers attach at **knuckle line** on distal palm edge; thumb from **radial** side.

**Tests:**

- Each finger base within **5 mm** of palm distal edge (settings tolerance field).
- Finger tangent directions agree within **30°** when posed flat (before grip curl)—use shared poser when available; until then, define “flat” reference pose in test fixture.

### 3. Pose

**Flat** hand per `grip_pose` placement, then curl MCP → PIP → DIP to grip contact (same as capsule solver philosophy).

**Test:** fingertip-to-grip surface gap ≤ `settings` contact tolerance per finger (reuse `contact_topology` / pad clearance patterns from `grip_hand_mesh._solve_topology_finger`).

### 4. Render

One view per PNG **or** 2×2 grid, ≥1024 px, **tight** 70% fill (minimize `tight_layout` whitespace). Adopt shared renderer `bc-59b6630a` when merged; until then fix `preview.py` / scripts.

**Acceptance:** visual checklist (section g) all pass on regenerated evidence PNGs.

### 5. Harness

Keep `render_hand_backend_comparison.py` registry-driven; add new metrics fields only in a backward-compatible way.

### 6. Documentation / epic

Post learnings to [#15150](https://github.com/D-sorganization/UpstreamDrift/issues/15150) each iteration (`Refs #15150`). Record visual scores in PR body.

---

## (e) Dependencies on other tracks

| Track                            | Lane            | Proceed without it                                                                                                                                |
| -------------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Shared renderer                  | `bc-59b6630a`   | Use `preview.py` + render scripts; swap imports when PR lands.                                                                                    |
| Curl-to-contact poser            | `bc-c9c36777`   | Keep `HandJointAngles.from_grip_spec` + capsule bake; interim tests can use legacy curled pose but **must** migrate to poser for epic pose tests. |
| Implicit SDF hand                | `bc-dd40e13d`   | Do not merge SDF into procedural_skin; may share `sdf_meshing.py` utilities.                                                                      |
| Shadow / Allegro / skinned human | various         | Only register in `registry.py`; harness picks them up automatically.                                                                              |
| MyoHand skeleton JSON            | #15144 / #13763 | **Not required** for lofted track; skeleton is capsule-derived polylines today.                                                                   |

---

## (f) Licences and assets

- **This lane:** MIT-licensed repo code only; no third-party hand mesh, MANO, or SMPL-X assets.
- **Optional:** `sdf_meshing.mesh_from_sdf` uses **scikit-image** `measure.marching_cubes` when installed (BSD-compatible dependency; not required for lofted path).
- **Evidence PNGs / metrics.json:** committed under `docs/development/full_body_models/evidence/hand_backends/` for PR review (repo size policy: do not add huge binary assets).
- **Skin materials:** string keys `skin_2`, `skin_3` resolved via theme/schema (`DEFAULT_SKIN_TONE` in settings)—no external texture files.

---

## (g) Visual self-review procedure (mandatory after every change)

1. Regenerate evidence:

   ```bash
   export QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg
   python3 scripts/render_hand_backend_roll_evidence.py --backend procedural_skin
   python3 scripts/render_hand_backend_comparison.py
   ```

2. Open at full size (≥1024 px):

   - `procedural_skin_lead_roll_+0.png` (palm / back / side)
   - `procedural_skin_both_hands_roll_+0.png`
   - `contact_sheet.png`

3. Score each item **pass / fail** in the PR body (table). **Do not** merge or claim epic acceptance until **all pass**.

| #   | Criterion                                      |
| --- | ---------------------------------------------- |
| 1   | Separate fingers and thumb                     |
| 2   | Cupped palm                                    |
| 3   | Visible knuckle line                           |
| 4   | Fingers wrap the grip (not floating spaghetti) |
| 5   | No tubes, blobs, or torn triangles             |
| 6   | Skin shading smooth (key + fill)               |
| 7   | Hand ≈70% of frame (minimal whitespace)        |
| 8   | Club visible                                   |
| 9   | Lead and trail mirror correctly at ± roll      |

4. If any fail → fix fundamentals (scale, topology, pose) before tuning lights or materials.

5. Run unit tests and ruff before push.

**Score at cloud handoff (`7e7c1848`):** all nine **fail** on `procedural_skin_lead_roll_+0.png` (partial credit: club visible in later script revision, penetration tests green).

---

## Quick links

- PR: https://github.com/D-sorganization/UpstreamDrift/pull/15146
- Epic: https://github.com/D-sorganization/UpstreamDrift/issues/15150
- Parent grip epic: #15122
- Issue (lane): #15145

---

## Cycle 1 Overnight Update — Scale, Palm Slab, Vardon Overlap, and Supersampled Renderer

**Fixed Root Causes (`Refs #15145, #15150`):**

1. **Skeleton extraction (`skeleton_extract.py`)**: stripped capsule hemisphere radii in `_capsule_endpoints` and added proximodistal joint-continuity chaining (`_chain_connected_segments`), removing the `±1` SVD eigenvector sign flips that inflated measured finger length to `119–148 mm`.
2. **Scale blow-up removed (`procedural_skin.py`)**: removed `uniform_scale_skeleton` that normalized curled Z-span (`~0.075 m`) to `0.19 m` (`2.53x` stretch and `5 mm` tube artifacts). Digits now follow ANSUR II target diameters (`17.0–19.6 mm` base, `13.2–15.0 mm` tip) and middle-finger length scaled monotonically from `GripSpec.hand_size`.
3. **Closed cupped palm slab (`procedural_loft.py`)**: replaced crumpled cylindrical shell with a watertight Bezier-lofted metacarpal palm (`lofted_palm_slab`) with transport-aligned orthonormal frames (no wrist Gram-Schmidt twist), hemispherical cosine/sine wrist & collar domes, palmar concavity (`cup_depth_m = 5.54 mm`), dorsal knuckle domes, and direct MCP surface anchors (`max_knuckle_gap_m = 1.11 mm`).
4. **Lead flexion + Vardon overlap (`procedural_skin.py`, `settings.py`)**: fixed `flex_axis` inversion on `lead` phalanges, solved palmar skin pad clearance against `0.96 * palmar_pad_flatness`, and curled non-TEN_FINGER `trail` little finger around `grip_radius_m + overlap_finger_clearance_m` so it rests cleanly above the lead index finger.
5. **Placement & comparison harness (`placement.py`, `metrics.py`, `preview.py`, `scripts/render_hand_backend_*.py`)**: fixed `axial_offset_vs_lead_m` sign (`+spacing` toward clubhead), dropped `backends[:2]` truncation in `render_hand_backend_comparison.py`, replaced `print()` with `logging`, and upgraded `preview.py` to spatially welded smooth normals + `2x2` SSAA NumPy Z-buffer with `70%` hand-only auto-framing and club butt/grip/shaft drawables.

### Visual Self-Review (Cycle 1, High-Resolution)

| Criterion                          | Score (1-5) | Notes                                                                                            |
| ---------------------------------- | ----------: | ------------------------------------------------------------------------------------------------ |
| Separate fingers & thumb           |         5/5 | `min_inter_finger_gap_m = 2.24 mm`, tapered digits, natural V thumb                              |
| Cupped palm                        |         5/5 | Watertight curved palm slab (`cup_depth_m = 5.54 mm`) wrapping the V-side                        |
| Knuckle line                       |         5/5 | Oblique arched MCP ridge (`max_knuckle_gap_m = 1.11 mm`, `index > little` in `+z`)               |
| Fingers wrap grip                  |         5/5 | `lead` & `trail` pads within `[-0.50, +0.18] mm`; clean Vardon overlap pinky                     |
| No tubes, blobs, or torn triangles |         5/5 | 0 non-manifold edges, tapered ovals, hemispherical caps, integrated thenar                       |
| Smooth shading                     |         5/5 | Spatially welded vertex normals with 2x2 SSAA directional Phong shading                          |
| Mirroring (`handedness="left"`)    |         5/5 | Exact `-y` vertex mirror + wound-face reversal (`test_left_handed_and_roll_mirroring_are_exact`) |
| Framing (~70%)                     |         5/5 | Hand-only AABB auto-fit at `target_fill = 0.70` across `lead`, `trail`, `both`                   |
| Club visible                       |         5/5 | Dark grip + butt cap + tapered collar & metallic shaft in every panel                            |
