# Skinned Human Hand Backend — Agent Handoff

**Track:** CC0 MakeHuman linear blend skinning (`hand_backends/skinned_human`)  
**Branch:** `cursor/skinned-human-hand-backend-751f` → base `cursor/hrg-gb1b-shaft-roll-2882`  
**Issue / PR:** [#15148](https://github.com/D-sorganization/UpstreamDrift/issues/15148) / [#15149](https://github.com/D-sorganization/UpstreamDrift/pull/15149)  
**Epic / quality bar:** [#15150 — Human hand model on the club](https://github.com/D-sorganization/UpstreamDrift/issues/15150)  
**Fleet:** Lead candidate per Dieter (2026-11-14/15); continue on subscription agents (Brick / DeskComputer).

---

## (a) Goal and quality bar

Deliver a **swappable visual backend** that returns a skinned CC0 human hand mesh in the **hand-grip frame**, driven by the same `GripSpec` / joint angles / shaft roll as other backends, with contact still on capsules.

The binding acceptance bar is **[epic #15150](https://github.com/D-sorganization/UpstreamDrift/issues/15150)**:

1. Evidence stills: lead, trail, both; butt / side / top; shaft roll 0° and ±20°; ≥1024 px; hand ~70% of frame; **club in every view**; smooth skin shading (not wireframe facets).
2. Fingers **wrap** the grip: fingertip and middle phalanx within contact tolerance; no penetration beyond tolerance; mirroring holds.
3. Anthropometric ratios within settings tolerances.
4. **Human checklist** (Dieter sign-off): reads as a hand **holding** a club — separate fingers and thumb, cupped palm, knuckle line, no tubes/blobs/torn triangles.
5. CC0 (or permissive) assets with NOTICE; MANO/SMPL-X research-only, not vendored.

**Not done yet:** Items (2) and (4) — fingers still read open; wrap is weak; Barb iteration-2 review flagged torn palm slab from vertex clamp, flat shading, tight crop.

---

## (b) Current state

### Layout

| Path | Role |
|------|------|
| `src/shared/python/model_appearance/hand_backends/skinned_human.py` | Backend entry: `build_hand_visual`, `register()` |
| `src/shared/python/model_appearance/hand_backends/registry.py` | `register_hand_visual_backend`, `build_hand_visual` dispatcher |
| `src/shared/python/model_appearance/hand_backends/joint_angles.py` | `HandJointAngles`, `joint_angles_from_grip_spec` |
| `src/shared/python/model_appearance/hand_backends/skeleton.py` | `bone_poses_from_joint_angles` (flat bone list, **not** MH hierarchy yet) |
| `src/shared/python/model_appearance/hand_backends/skinning.py` | `SkinnedHandAsset`, `pose_vertices`, `apply_palm_correctives` |
| `src/shared/python/model_appearance/hand_backends/hand_asset.py` | Loads `assets/hand_cc0/hand_cc0.npz` |
| `src/shared/python/model_appearance/hand_backends/qa.py` | Penetration / self-penetration helpers |
| `src/shared/python/model_appearance/hand_backends/types.py` | `HandVisualResult` |
| `src/shared/python/model_appearance/grip_preview_matplotlib.py` | Evidence renderer (Lambert + vtx normals; interim until shared renderer) |
| `assets/hand_cc0/build_hand_asset.py` | Build `hand_cc0.npz` from MakeHuman `base.obj` |
| `assets/hand_cc0/mh_rig_io.py` | **WIP** — parse `default.mhskel` / `default_weights.mhw` (not wired into build yet) |
| `scripts/config/skinned_human_hand.json` | Skin tone, palm arch, grip wrap, anthropometric refs |
| `scripts/config/hand_visual_quality.json` | Test tolerances (penetration, mirror, self-penetration) |
| `docs/development/full_body_models/evidence/hand_skinned_human/render_skinned_human_evidence.py` | Generates 21 PNGs |
| `tests/unit/model_appearance/hand_backends/test_skinned_human.py` | Unit tests (9) |

### Settings

**`scripts/config/skinned_human_hand.json`**

| Key | Purpose |
|-----|---------|
| `skin_rgb` | Evidence / mesh tint |
| `palm_arch_gain` | Palm corrective strength in `apply_palm_correctives` |
| `grip_wrap_translation_m` | Pad beyond grip radius when placing hand |
| `length_reference_m` / `breadth_reference_m` | Anthropometric scale vs `GripSpec.hand_size` |

**`scripts/config/hand_visual_quality.json`**

| Key | Typical value | Used by |
|-----|----------------|---------|
| `grip_cylinder_penetration_m` | `0.002` | Grip penetration test |
| `self_penetration_fraction` | `0.08` | Inter-digit heuristic |
| `shaft_roll_median_error_m` | `0.003` | Shaft roll test |
| `mirror_median_error_m` | `0.008` | Trail mirror test (dense mesh) |

**Iteration 3 targets (not in JSON yet):** `grip_radius_min_m` `0.0105`, `grip_radius_max_m` `0.0135`, per-finger contact tolerance — add when curl-to-contact lands.

### Public API

```python
from src.shared.python.model_appearance.hand_backends import build_hand_visual
from src.shared.python.model_appearance.hand_backends.joint_angles import (
    joint_angles_from_grip_spec,
)

result = build_hand_visual(
    grip_spec,
    "lead",  # or "trail"
    joint_angles_from_grip_spec(grip_spec, "lead"),
    grip_radius_m=0.0145,
)
# result.mesh — vertices in hand-grip frame; result.skin_rgb
```

### Tests

```bash
cd /path/to/UpstreamDrift
python3 -m pytest tests/unit/model_appearance/hand_backends/test_skinned_human.py -m unit --timeout=60
python3 -m ruff check src/shared/python/model_appearance/hand_backends assets/hand_cc0
```

**Last known result on branch tip before handoff:** **9 passed** (commit `8a0f21a96…`).

### Evidence / PNGs

```bash
python3 docs/development/full_body_models/evidence/hand_skinned_human/render_skinned_human_evidence.py
```

Output: `docs/development/full_body_models/evidence/hand_skinned_human/` — 21 files (`lead_*`, `trail_*`, `both_*`, roll `±20` / `0`).

Rebuild asset (requires MakeHuman tree or pinned `base.obj`):

```bash
git clone --depth 1 https://github.com/makehumancommunity/makehuman.git /tmp/mh
python3 assets/hand_cc0/build_hand_asset.py \
  /tmp/mh/makehuman/data/3dobjs/base.obj \
  --source-commit "$(git -C /tmp/mh rev-parse HEAD)"
```

---

## (c) What worked and what failed

### Worked

| Item | Detail |
|------|--------|
| Real CC0 mesh | Downloaded MakeHuman on VM; joint-bbox extract → **1606 verts / 3168 faces** (iteration 1 had **~115 verts** from wrong `y≈5.5` crop on MH coords ~2.4) |
| First readable hand | Barb: first track that **reads as a human hand** (iteration 2) |
| Shaft roll | Tests pass via `resolved_shaft_roll_deg` + `apply_shaft_roll` |
| Penetration tests | Pass after radial placement fix |
| Asset provenance | `PROVENANCE.json`, `NOTICE` with commit `a8bc2d54ff0ac92e78ff71431b1023eda42bf482` |

### Failed / open (with root causes)

| Symptom | Root cause | Dead end — avoid |
|---------|------------|------------------|
| Fingers straight, no wrap | Pose from **normalized curl scalars** in `joint_angles_from_grip_spec`, not curl-to-contact; LBS uses **independent bone matrices**, not MH hierarchy | Don't tune `curl_scale` alone expecting grip wrap |
| Torn palm slab | **`_clamp_grip_clearance`** / **`_place_on_grip` radial scale** moved verts near `ρ→0` (exploded x,y) | **Do not** revive all-palm radial inflate; Barb iteration 3: remove vertex-pushing clamp entirely |
| Flat / faceted look | Matplotlib **per-triangle** shading; tight `margin_fraction` crops | Wireframe edges fixed (alpha 0); need **Gouraud/Phong** per vertex + wider framing |
| Camera crop | `apply_equal_axis_limits` with aggressive margin on subset of points | Frame on **all hand + shaft** verts with comfortable margin (~30–40%) |
| Weak LBS at curl | `skeleton._about_joint` × flat `rest_poses` with only translations; not MH weights/parent chain | Proximity-weight assignment in old build was wrong topology for dense mesh |

### Numbers (iteration 2, post-fix placement)

- Hand span after placement: **z ≈ 0.17 m**, **ρ_min ≈ 0.016 m** at `grip_radius_m=0.0145`
- Neutral LBS max vertex displacement from rest: **~0**; curled **~22 mm** max (not enough for wrap)

---

## (d) Next steps (priority order)

Each step should add/adjust tests first (TDD), then implementation, then **visual self-review** (section g).

### 1. MakeHuman skeleton + weights rig (P0)

**Goal:** Build `hand_cc0.npz` from `base.obj` + `data/rigs/default.mhskel` + `data/rigs/default_weights.mhw` (CC0).

**Tasks:**

- Wire `assets/hand_cc0/mh_rig_io.py` into `build_hand_asset.py`
- Right-hand bones: `wrist.R`, `metacarpal1–4.R`, `finger1–3.R` (thumb), `finger2–3.R` … `finger5–3.R`
- Map full-mesh vertex indices → hand subset; pack top-4 weights per vertex from `.mhw`
- Store `bone_parents`, `inv_bind_poses` (4×4), `bone_heads` in npz; extend `SkinnedHandAsset` + `hand_asset.py` loader
- Hierarchical LBS: `pose_vertices(asset, global_poses)` with `palette[i] = global[i] @ inv_bind[i]`

**Acceptance tests:**

- `hand_cc0.npz` loads; **≥1500 faces**, **≥1500 verts**
- Neutral pose: `pose_vertices` matches rest within **1e-6 m** max
- `max_self_penetration_fraction` ≤ setting; no **degenerate** faces (`test_mesh_no_degenerate_triangles`)

**Files on disk (clone repo):**

- `makehuman/data/rigs/default.mhskel`
- `makehuman/data/rigs/default_weights.mhw`
- `makehuman/data/3dobjs/base.obj`

### 2. Joint-rotation curl-to-contact (P0)

**Goal:** Remove `_place_on_grip` radial inflation and **`_clamp_grip_clearance`** entirely. Pose by **joint rotation** (MCP → PIP → DIP; thumb CMC/MCP/IP); increment flex until skin **surface** meets grip cylinder within tolerance.

**Tasks:**

- New module `hand_backends/curl_to_contact.py` (or adopt **bc-c9c36777** when merged; keep thin adapter)
- Settings: `grip_radius_min_m` **0.0105**, `grip_radius_max_m` **0.0135**, `contact_surface_tolerance_m`
- Flat-hand placement: index proximal on club axis; pinky base slightly below (Dieter); bake as rigid pre-curl transform in grip frame
- Metacarpal arch via bone rotation, not vertex correctives only

**Acceptance tests:**

- Fingertip + middle-phalanx sample verts for **each** finger within contact tolerance (new test using bone-weight masks)
- `max_grip_cylinder_penetration_m` ≤ `hand_visual_quality.json`
- Visual: fingers visibly curled around shaft in side + butt views

### 3. Smooth normals + framing (P1)

**Goal:** Gouraud or Phong vertex shading; hand + grip **fully in frame** at ~70% fill, 1024 px.

**Tasks:**

- Upgrade `grip_preview_matplotlib` (or adopt **bc-59b6630a** shared renderer)
- Evidence script: render **lead and trail separately**; `margin_fraction` **≥ 0.35**
- Optional: split “both hands” into two panels

**Acceptance tests:**

- Subjective checklist all **pass** (section g)
- PNG dimensions **≥ 1024** on shortest side

### 4. Tests parity with epic (P1)

Add to `test_skinned_human.py`:

- Contact on all five digits (tip + mid phalanx)
- No flipped/degenerate triangles after pose (`signed area` / volume > 0)
- Shaft roll ±20° (existing)
- Mirror (existing)

---

## (e) Dependencies on other tracks

| Lane | ID | Adopt when | Until then |
|------|----|------------|------------|
| Shared curl-to-contact | bc-c9c36777 | Module on branch/`main` | Local `curl_to_contact.py`; keep API compatible with `HandJointAngles` |
| Shared renderer | bc-59b6630a | Same | Keep `grip_preview_matplotlib` + evidence script |
| MyoHand skeleton JSON | bc-761b1bab / #15150 | JSON posted | Use MH `default.mhskel` joint positions |
| Comparison harness | #15146 | `render_hand_backend_comparison.py` | Per-backend evidence folder |

**Registry:** Expect trivial merge conflicts in `registry.py` with parallel PRs #15143, #15146, #15147.

---

## (f) Licences and assets

| Asset | Licence | Source |
|-------|---------|--------|
| MakeHuman `base.obj` (hm08) | CC0 | `https://github.com/makehumancommunity/makehuman/blob/a8bc2d54ff0ac92e78ff71431b1023eda42bf482/makehuman/data/3dobjs/base.obj` |
| `default.mhskel` | CC0 (ship with MH data) | `.../makehuman/data/rigs/default.mhskel` |
| `default_weights.mhw` | CC0 (explicit in file) | `.../makehuman/data/rigs/default_weights.mhw` |
| Generated `hand_cc0.npz` | Derivative; record SHA256 in `PROVENANCE.json` | `assets/hand_cc0/` |

Repo copies: `LICENSE.CC0.txt`, `SOURCE_LICENSE_STATEMENT.txt`, `NOTICE` — **do not** commit full MakeHuman clone (`assets/hand_cc0/.gitignore` has `download/`).

**Research-only:** MANO, SMPL-X — noted in NOTICE, **not vendored**.

---

## (g) Visual self-review procedure (mandatory every change)

1. Run tests (section b).
2. Regenerate evidence:

   ```bash
   python3 docs/development/full_body_models/evidence/hand_skinned_human/render_skinned_human_evidence.py
   ```

3. Open at minimum:

   - `lead_side_roll_+0.png`, `lead_butt_roll_+0.png`, `lead_top_roll_+0.png`
   - `trail_side_roll_+0.png`
   - `both_side_roll_0.png`
   - One roll variant: `lead_side_roll_+20.png`

4. Score each image **pass / fail** against this checklist:

| # | Criterion |
|---|-----------|
| 1 | Separate fingers visible |
| 2 | Thumb identifiable |
| 3 | Cupped palm (not flat sheet) |
| 4 | Knuckle line readable |
| 5 | Fingers wrap the grip (not open / parallel to shaft) |
| 6 | No tubes, blobs, or torn triangles |
| 7 | Skin shading smooth (no heavy facet / wireframe look) |
| 8 | Hand ≈70% of frame; not cropped |
| 9 | Club shaft visible |
| 10 | Lead vs trail mirror plausibly (for paired views) |

5. Record scores in the PR body under **Learnings** or **Visual review** with date and commit SHA.
6. **Do not** claim #15150 bar met until **all 10 pass** on the standard set.

---

## Related coordination

- Learnings for epic relay: put in PR **## Learnings** (GitHub issue comments fail for cloud bot; Barb relays to #15150).
- Child issue: **Refs #15148**, epic **Refs #15150**, parent grip **Refs #15122**.
- Do **not** arm auto-merge; target base `cursor/hrg-gb1b-shaft-roll-2882` until stack merges.

---

*Handoff written at end of cloud lane bc-1714e768 / cursor/skinned-human-hand-backend-751f. Iteration 3 (MH rig + curl-to-contact) intentionally partial — `mh_rig_io.py` only.*