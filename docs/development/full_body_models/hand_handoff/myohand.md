# MyoHand Hand Backend — Lane Handoff

**Branch:** `cursor/myohand-visual-backend-dcca`  
**PR:** [#15147](https://github.com/D-sorganization/UpstreamDrift/pull/15147)  
**Child issue:** [#15144](https://github.com/D-sorganization/UpstreamDrift/issues/15144)  
**Last cloud agent run:** bc-761b1bab (Cursor Cloud, 2026-11-15)

Subscription fleet (Claude Code on Brick, Antigravity on DeskComputer) continues this lane via normal dev channels.

---

## (A) Goal and Quality Bar

**Goal:** Provide a **research-only anatomical skeleton reference** for human hands on a golf club — separate MyoHand **bone** meshes in the grip hand-grip frame, plus a **bilateral skeleton JSON** (joint centres and axes) for the MakeHuman skinned track ([#15149](https://github.com/D-sorganization/UpstreamDrift/pull/15149)), SDF, and posing lanes.

This is **not** a production skinned grip visual. Myo_sim ships STL bones only; MuJoCo `class="skin"` entries are discrete capsules/ellipsoids, not a continuous skin mesh.

**Authoritative quality bar:** [Epic #15150 — Human hand model on the club](https://github.com/D-sorganization/UpstreamDrift/issues/15150) (parallel tracks, evidence views, contact tolerances, human checklist).

**Contract:** `build_hand_visual(grip_spec, side, joint_angles)` in `grip_hand_frame`; contact physics stays on procedural capsules.

---

## (B) Current State

### Submodule / Assets

- **Model:** `shared/models/myosuite/myo_sim/` (pinned submodule). Hand XML: `hand/myohand.xml` (see `model_xml_relpath` in settings JSON).
- **NOTICE:** `src/shared/python/model_appearance/hand_backends/assets/myohand/NOTICE`

### Source Modules (`src/shared/python/model_appearance/hand_backends/`)

| Module               | Role                                                                                                                                                                                                                                                        |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `myohand.py`         | `build_hand_visual` (ADR-0057 contract, one mesh), `build_myohand_bone_meshes` (per-bone dict), `register()`, `penetration_depth_m`, `finger_pair_penetration_m`, `mirror_y_vertices`; exports `_model_and_data`, `_pipeline_transform` for skeleton export |
| `registration.py`    | `kabsch_rigid_transform`, `registration_to_grip_hand_mesh` — aligns posed bones to **curled capsule** from `build_grip_hand_mesh`                                                                                                                           |
| `skeleton_export.py` | `export_myohand_skeleton`, `export_bilateral_myohand_skeleton` — JSON in grip hand frame                                                                                                                                                                    |
| `joint_mapping.py`   | `grip_spec_to_myohand_joint_targets`, `anatomical_mirror_for_side`                                                                                                                                                                                          |
| `mujoco_extract.py`  | `load_myohand_model`, `apply_joint_targets`, `extract_hand_bone_meshes`, `anthropometric_scale_matrix`                                                                                                                                                      |
| `render_evidence.py` | Interim lit matplotlib renderer (`LitRenderSettings`, `render_scene_png`) — **temporary** until shared renderer lane lands                                                                                                                                  |
| `registry.py`        | Shared ADR-0057 registry (#15197): `myohand` registers with `default_enabled=False` when myo_sim assets exist; build it by name                                                                                                                             |
| `settings.py`        | `MyoHandVisualSettings`, `default_myohand_settings()` (cached, loads JSON)                                                                                                                                                                                  |
| `types.py`           | Shared `HandSide`, `HandVisualResult` (ADR-0057 A2)                                                                                                                                                                                                         |

### Settings

**File:** `scripts/config/hand_backend_myohand.json`

Important fields:

- `grip_radius_m`, `club_cylinder_radius_m` (0.014 m typical)
- `penetration_tolerance_m`, `finger_pair_tolerance_m`
- `bone_rgb`, `grip_rgb` (evidence rendering)
- `hand_body_names` — bodies exported as meshes
- `registration_mode`: `kabsch_to_capsule_curl`
- `research_only`, `license_gate_issue`: **13763**
- `model_xml_relpath`: relative to repo root

### Skeleton JSON (Ready for Skinned Track)

**Committed example:**

`docs/development/full_body_models/evidence/hand_backend_myohand/myohand_skeleton_grip_spec_neutral.json`

**API:**

```python
from src.shared.python.grip_spec import GripSpec
from src.shared.python.model_appearance.hand_backends.skeleton_export import (
    export_bilateral_myohand_skeleton,
    export_myohand_skeleton,
)

export_bilateral_myohand_skeleton(GripSpec.neutral())
export_myohand_skeleton(GripSpec.legacy(), "trail")
```

**Schema (`schema_version: 1`):**

- Top: `grip_spec_sha256`, `lead`, `trail`
- Per side: `side`, `handedness`, `research_only`, `license_gate_issue`, `joint_angles_rad`, `joints`
- Per joint: `origin_m` [x,y,z], `axis_unit` [x,y,z] in **grip hand-grip frame** (sagittal mirror applied on lead for right-handed specs)

Joint names follow `MYOHAND_CANONICAL_JOINTS` in `myohand_bilateral_scene`.

### Tests

**File:** `tests/unit/model_appearance/hand_backends/test_myohand_visual_backend.py` (14 tests)

**Requires:** `myo_sim` submodule initialized (`myohand_assets_ready()`).

```bash
cd /path/to/UpstreamDrift
git submodule update --init shared/models/myosuite/myo_sim   # if needed
python3 -m pytest tests/unit/model_appearance/hand_backends/test_myohand_visual_backend.py -m unit
python3 -m ruff check src/shared/python/model_appearance/hand_backends/
```

| Test                                                       | Intent                                                           |
| ---------------------------------------------------------- | ---------------------------------------------------------------- |
| `test_myohand_is_registered_but_disabled_by_default`       | Listed only with `include_disabled=True`                         |
| `test_disabled_backend_still_builds_by_name`               | `build_hand_visual(..., backend="myohand")` builds the bone mesh |
| `test_skeleton_export_has_all_canonical_joints`            | JSON export completeness                                         |
| `test_meshes_have_finite_watertight_parts`                 | Mesh sanity                                                      |
| `test_palm_finger_depth_moves_cmc_and_wrist_joints`        | GripSpec coupling                                                |
| `test_thumb_straightness_moves_thumb_mp_ip`                | Thumb mapping                                                    |
| `test_lead_trail_mirroring_under_right_handed_spec`        | `firstmc` centroid separation > 12 mm                            |
| `test_shaft_roll_rotates_hand_about_grip_z`                | Roll after registration (Kabsch seed excludes roll)              |
| `test_club_penetration_within_tolerance`                   | Radial penetration metric (see §c)                               |
| `test_finger_parts_do_not_interpenetrate_beyond_tolerance` | Inter-bone clearance                                             |
| `test_left_handed_mirrors_shaft_roll_on_build`             | Left-handed smoke                                                |
| `test_mirror_y_swaps_hand_side`                            | Mirror helper                                                    |
| `test_joint_angle_override_is_honored`                     | Overrides                                                        |
| `test_anthropometric_scaling_changes_extent`               | Scale matrix monotonicity (not final mesh extent)                |

### Evidence PNGs

**Directory:** `docs/development/full_body_models/evidence/hand_backend_myohand/`

**Script:** `docs/development/full_body_models/evidence/hand_backend_myohand/render_myohand_evidence.py`

```bash
export MPLBACKEND=Agg
python3 docs/development/full_body_models/evidence/hand_backend_myohand/render_myohand_evidence.py
```

Produces:

- Per hand (`lead`, `trail`): `front`, `side`, `top`, `grip_oblique` × roll `p0`, `m20`, `p20`
- Both hands: `grip_oblique`, `front`, `side` × same rolls
- `myohand_skeleton_grip_spec_neutral.json`, `receipt.json`

Naming: `{side}_{view}_club_roll_{p0|m20|p20}.png` or `both_{view}_club_roll_*.png`

**Placement:** Hands built in grip hand frame, then transformed with `grip_hand_frame` + `ClubAssembly` grip mesh (see script).

---

## (C) What Worked and What Failed

### Worked

1. **Bone-only positioning** — Separate phalanges, carpal bones, thumb chain read as anatomy; bone RGB + Lambert key/fill in evidence.
2. **Shared curl joint targets** — `pose_to_myohand_joint_targets` via `grip_spec_to_myohand_joint_angles` (same path as capsule intent).
3. **Kabsch to curled capsule** — `registration_to_grip_hand_mesh` replaces static neutral calibration matrix; reproducible if Kabsch seed uses **joint targets + side**, not full `spec_sha256` (shaft roll must not re-seed registration).
4. **Skeleton export** — MuJoCo `data.xanchor` / `data.xaxis` pushed through placement transform; bilateral JSON committed for #15149.
5. **Registry safety** — `research_only`, default remains `capsule`, explicit `allow_research=True`.
6. **Club in every evidence frame** — Grip mesh from `ClubAssembly` + `assembly_meshes`.

### Failed / Partial (With Numbers)

| Issue                               | Detail                                                                                                                                                                                                                                                                             |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Not a grip visual**               | No skin; first iteration used skin-tinted bones without club — rejected.                                                                                                                                                                                                           |
| **Fingertip air gap**               | Bones curl in joint space but STLs do not hug rubber; visually ~10–15 mm gap. `penetration_depth_m` measures **inward** penetration (positive when verts inside cylinder radius); **air gap yields ~0** in that metric — so unit test can pass while visuals fail “wrap the grip”. |
| **Frame fill < 70%**                | Matplotlib 3D ortho + tight limits: content bbox ~25–45% of 1024×1024 on sampled PNGs after iteration 2.                                                                                                                                                                           |
| **Kabsch vs anthropometric extent** | Final mesh extent does **not** track `HandSize.length_m` monotonically (registration absorbs scale to capsule). Test now checks `anthropometric_scale_matrix` only — **do not** reintroduce full-mesh extent assertion without fixing pipeline.                                    |
| **Random Kabsch subsample**         | Seeding from `spec_sha256` broke shaft-roll test (RMS ~0.04 m). Fixed: stride subsample; seed from `(side, sorted joint targets)`.                                                                                                                                                 |
| **Lead/trail mirroring**            | Whole-hand centroid useless after Kabsch; use **`firstmc`** separation (> 12 mm) for legacy spec.                                                                                                                                                                                  |
| **Epic comments**                   | Cloud bot got **403** posting to #15150; learnings live on PR #15147 comment until subscription agent posts.                                                                                                                                                                       |

### Dead Ends (Avoid)

- Rendering MuJoCo `class="skin"` geoms as “skin” — not a surface mesh.
- Static `model_to_grip` matrix from JSON — superseded by Kabsch to capsule.
- Skin-tinted bone colour as faux skin — use `bone_rgb`.
- `bbox_inches="tight"` on 1024 renders — shrinks output below 1024 px.
- Asserting lead/trail mirror via global mesh centroid after registration.
- GB-2 `GripBuilder` — not in repo; use `build_grip_hand_mesh` + joint targets.

---

## (D) Next Steps (Priority Order)

### 1. Stabilize Skeleton JSON for MakeHuman Lane (P0)

**Acceptance:**

- `export_bilateral_myohand_skeleton(GripSpec.neutral())` matches committed evidence JSON keys for all `MYOHAND_CANONICAL_JOINTS` on lead and trail.
- `python3 -m pytest tests/unit/model_appearance/hand_backends/test_myohand_visual_backend.py::test_skeleton_export_has_all_canonical_joints`
- Document consumer path in #15149 / skinned_human code when wired.

### 2. Adopt Shared Curl-to-Contact (P1) — Epic Track Bc-C9c36777

**When:** PR lands on stack from #15150.

**Acceptance:**

- Replace or augment Kabsch-only placement with shared module; finger pad contact within settings from epic (fingertip/middle phalanx).
- Visual self-review: “fingers wrap the grip” → **pass** on `lead_grip_oblique_club_roll_p0.png` and trail analogue.
- `penetration_depth_m` and/or new contact metrics match epic tolerances (settings, not hard-coded).

### 3. Adopt Shared Renderer (P1) — Epic Track Bc-59B6630a

**When:** PR lands on stack.

**Acceptance:**

- Delete or thin `render_evidence.py`; evidence script calls shared helper.
- PNGs ≥ 1024 px; subject fill ≥ **0.65** of frame (automate bbox metric in script or test).
- Re-render full matrix; update `receipt.json`.

### 4. Tighten Visual Contact Without Shared Module (P2 Fallback)

**Acceptance:**

- Per-finger radial distance to grip cylinder ≤ configured tolerance in **grip_hand_frame** (new unit test).
- Regenerate evidence; human checklist ≥ 7/10 applicable items (skeleton lane: skip skin shading).

### 5. Licence / Default Backend (P3)

**Acceptance:**

- Owner sign-off on [#13763](https://github.com/D-sorganization/UpstreamDrift/issues/13763) recorded.
- Only then: `research_only: false` in JSON + registry; update NOTICE.

### 6. Merge Coordination

**Acceptance:**

- Rebase on `cursor/hrg-gb1b-shaft-roll-2882` (or `main` when stack merges).
- Resolve registry conflicts with #15143, #15146, #15149 without breaking `allow_research` gate.
- CI green: ruff, pytest unit file, file size budgets.

---

## (E) Dependencies on Other Tracks

| Track                  | ID              | This lane without it                                                |
| ---------------------- | --------------- | ------------------------------------------------------------------- |
| Shared renderer        | bc-59b6630a     | Keep `render_evidence.py`; tune `half`, `frame_fill`, `dpi` locally |
| Shared curl-to-contact | bc-c9c36777     | Keep `grip_spec_to_myohand_joint_angles` + Kabsch to capsule        |
| Capsule baseline       | #15141 / #15142 | Already depends on `build_grip_hand_mesh`                           |
| Skinned human          | #15149          | **Consume skeleton JSON now** — no blocker on renderer              |
| Shaft roll GB1b        | #15138 / #15139 | Base branch includes per-hand roll                                  |

**Proceed without shared modules:** Skeleton JSON and bone meshes are usable today; visual polish waits on renderer/posing.

---

## (F) Licences and Assets (#13763)

| Asset                   | Licence                                    | Notes                                              |
| ----------------------- | ------------------------------------------ | -------------------------------------------------- |
| `myo_sim` hand STLs     | Apache-2.0                                 | Submodule pin; no new vendoring in PR              |
| MoBL / 2nd-Hand lineage | BSD + **non-commercial** (upstream README) | Recorded in `NOTICE`                               |
| Backend default         | **Research-only**                          | `research_only: true`, `license_gate_issue: 13763` |
| MANO / SMPL-X           | Not vendored                               | Cited as research-only alternatives                |

**Production use** requires explicit owner sign-off on #13763; do not clear `research_only` without it.

---

## (G) Visual Self-Review Procedure (Mandatory After Every Visual Change)

After **any** change to posing, registration, rendering, or evidence scripts:

1. **Regenerate** full evidence set:

   ```bash
   MPLBACKEND=Agg python3 docs/development/full_body_models/evidence/hand_backend_myohand/render_myohand_evidence.py
   ```

2. **Open** at minimum:

   - `lead_grip_oblique_club_roll_p0.png`
   - `trail_grip_oblique_club_roll_p0.png`
   - `both_grip_oblique_club_roll_p0.png`
   - `lead_side_club_roll_p20.png`, `trail_side_club_roll_m20.png`

3. **Score** each checklist item **pass / fail / N/A** (skeleton lane: N/A for skin-only items).

   | #   | Criterion                       | Skeleton lane | Current score (2026-11-15)                             |
   | --- | ------------------------------- | ------------- | ------------------------------------------------------ |
   | 1   | Separate fingers and thumb      | Required      | **pass**                                               |
   | 2   | Cupped palm                     | Required      | **partial** (curl via joints; palm not rubber-wrapped) |
   | 3   | Knuckle line readable           | Required      | **pass**                                               |
   | 4   | Fingers wrap the grip           | Required      | **fail** (air gap)                                     |
   | 5   | No tubes/blobs/torn triangles   | Required      | **pass**                                               |
   | 6   | Skin shading smooth             | Skin backends | **N/A**                                                |
   | 7   | Hand ~70% of frame              | Required      | **fail** (~30–45%)                                     |
   | 8   | Club visible                    | Required      | **pass**                                               |
   | 9   | Lead and trail mirror correctly | Required      | **pass** (both-hand views)                             |

4. **Record scores** in PR **## Learnings** and honest verdict; iterate until all **Required** items pass before claiming epic quality bar met.

5. **Optional metric:** bbox fill ratio on PNG (luminance < 240):

   ```python
   # Quick check — paths under hand_backend_myohand/
   from PIL import Image
   import numpy as np

   im = Image.open("lead_grip_oblique_club_roll_p0.png")
   arr = np.array(im.convert("L"))
   ys, xs = np.where(arr < 240)
   w, h = im.size
   fill = max((xs.max() - xs.min() + 1) / w, (ys.max() - ys.min() + 1) / h)
   assert fill >= 0.65, fill
   ```

6. **Run unit tests** before push:

   ```bash
   python3 -m pytest tests/unit/model_appearance/hand_backends/test_myohand_visual_backend.py -m unit
   ```

---

## Quick Reference

| Item                     | Value                                                                                                     |
| ------------------------ | --------------------------------------------------------------------------------------------------------- |
| PR                       | [#15147](https://github.com/D-sorganization/UpstreamDrift/pull/15147)                                     |
| Epic                     | [#15150](https://github.com/D-sorganization/UpstreamDrift/issues/15150)                                   |
| Skeleton JSON            | `docs/development/full_body_models/evidence/hand_backend_myohand/myohand_skeleton_grip_spec_neutral.json` |
| Base branch (at handoff) | `cursor/hrg-gb1b-shaft-roll-2882`                                                                         |
| Default drawn hand       | capsule mesh (outside the registry, ADR-0057 A3)                                                          |
| Opt-in myohand           | `build_hand_visual(spec, side, grip_radius_m=r, backend="myohand")`                                       |

## Cycle 1 Calibration Update

### What Changed Since Previous Handoff

- Replaced coarse Kabsch-to-capsule registration with `anatomical_curl_to_contact` (`contact_solver.py`), optimizing finger and 4-DOF thumb flexion directly against the `GripSpec` grip cylinder (`0.002 m` penetration guard, `0.0105 m` middle-phalanx tolerance, `0.0055 m` thumb-tip tolerance, `0.0028 m` palm seat clearance).
- Fixed mirrored triangle winding via `(det < 0) ^ mirror_y` in `mujoco_extract.py` and cofactor axial transformation in `skeleton_export.py` so exported `myohand_skeleton_grip_spec_neutral.json` joint origins/axes track the exact posed `lead` and `trail` bones under `0°` and `±20°` shaft roll.
- Upgraded `render_evidence.py` to vectorized `uint64` packed `ss=2` Gouraud Z-buffer rendering (`1024 px`, `0.7` frame fill, full dark grip sleeve + silver shaft visible in `lead`, `trail`, and `both` views).
- Added `tests/unit/model_appearance/hand_backends/test_myohand_contact_and_framing.py`; `19 passed` across `tests/unit/model_appearance/hand_backends/`.
- Kept `research_only=True` (`allow_research=True` required) pending `#13763` sign-off.

## Cycle 2 Calibration Update (2026-11-15)

### What Changed

- Added thumb segment XY chord guard in `contact_solver.py` (`_ThumbEvalCtx`, `_xy_segment_min_r`, `_thumb_pose_cost`, and `_relax_thumb_penetration` invoked from `_relax_club_penetration`) so `firstmc -> proximal_thumb -> distal_thumb` clears the `GripSpec` grip cylinder without chord pass-through across `0°` and `±20°` shaft roll.
- Exposed `render_frame_fill: 0.74` as a setting in `scripts/config/hand_backend_myohand.json` and `MyoHandVisualSettings` (`settings.py`), wired into `docs/development/full_body_models/evidence/hand_backend_myohand/render_myohand_evidence.py`.
- Added Cycle 2 TDD suite `tests/unit/model_appearance/hand_backends/test_myohand_cycle2_thumb_cup_framing.py` verifying `research_only is True`, `license_gate_issue == 13763`, `0.72 <= render_frame_fill <= 0.76`, thumb bone-centroid XY chord clearance on `lead`/`trail` across `(-20.0, 0.0, 20.0)` deg roll, `distal_thumb` radial wrap band, and healthy `export_bilateral_myohand_skeleton(GripSpec.neutral())`.
- Regenerated 33 `1024 px` PNG evidence images, `receipt.json` (`"frame_fill": 0.74`, `"research_only": true`), and `myohand_skeleton_grip_spec_neutral.json`, and visually verified `lead`, `trail`, and `both` across `0°` and `±20°` roll.

### Verification Command and Result

- Evidence render: `PYTHONPATH=. python docs/development/full_body_models/evidence/hand_backend_myohand/render_myohand_evidence.py` -> regenerated 34 artifacts (`33` PNGs + `receipt.json` + `myohand_skeleton_grip_spec_neutral.json`).
- Unit tests: `run_pytest.py ... tests/unit/model_appearance/hand_backends/ -v` -> **`27 passed, 4 warnings in 23.28s`**.

### Remaining Known Issues

- On the `trail` hand, `distal_thumb` flares slightly wider away from the grip sleeve than on `lead` (still within the `0.0055 m` `thumb_contact_tolerance_m`).
- Remains gated by `research_only=True` (`allow_research=True` required) pending `#13763` license sign-off.
- Still uses standalone `render_myohand_evidence.py` until shared `posing.py` (`#15164`) and `render.py` (`#15154`) land and are pulled in.
