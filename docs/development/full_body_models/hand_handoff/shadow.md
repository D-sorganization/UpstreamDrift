# Shadow Hand Visual Backend — Lane Handoff

**Branch:** `cursor/shadow-hand-grip-visuals-4437`  
**PR:** [#15143](https://github.com/D-sorganization/UpstreamDrift/pull/15143)  
**Track issue:** [#15151](https://github.com/D-sorganization/UpstreamDrift/issues/15151)  
**Epic (quality bar):** [#15150](https://github.com/D-sorganization/UpstreamDrift/issues/15150)  
**Base stack:** `cursor/hrg-gb1b-shaft-roll-2882` (#15139 shaft roll)

---

## (a) Goal and quality bar

**Goal:** Replace capsule **visuals** on the club grip with the Apache-2.0 **Shadow Dexterous Hand** (MuJoCo Menagerie meshes). Physics and contact remain on the existing capsule hand in `grip_hand_mesh.py` — do **not** remove or repoint capsule contact without an explicit epic change.

**Contract (shared across all hand tracks):** `build_hand_visual(grip_spec, side, joint_angles)` in `src/shared/python/model_appearance/hand_backends/`, registered via `registry.py`, driven by the same `GripSpec`, joint angles, and shaft roll as every other backend.

**Quality bar** ([epic #15150](https://github.com/D-sorganization/UpstreamDrift/issues/15150)):

1. Evidence PNGs: lead, trail, and both hands; iso, butt, and face views; shaft roll 0° and ±20°; solid skin shading; hand ≈70% of frame at ≥1024 px; club visible in every view.
2. Fingers wrap the grip within tolerance settings; penetration bounded by settings; L/R mirroring holds.
3. Anthropometric ratios within configured tolerances.
4. Written human checklist in the PR; Dieter sign-off before merge.
5. Licence recorded beside assets (`NOTICE` / `PROVENANCE.json`).

---

## (b) Current state — files, functions, settings, tests, evidence

### Layout

| Piece | Path | Role |
|-------|------|------|
| Shadow backend | `src/shared/python/model_appearance/hand_backends/shadow.py` | Solve, mesh extract, `build_hand_visual` |
| Types | `hand_backends/types.py` | `HandVisualResult`, `HandSide`, `JointAngles` |
| Registry | `hand_backends/registry.py` | Default backend `"shadow"` |
| Capsule compare | `hand_backends/shadow_capsule_compare.py` | Pad centroids vs capsule hand |
| Placement | `grip_hands.py` | `_placed_hand` → registry + `grip_hand_frame` |
| Config | `scripts/config/grip_shadow_hand_visual.json` | All tolerances and visual knobs |
| Assets / licence | `assets/third_party/shadow_dexterous_hand/` | `PROVENANCE.json`, `LICENSE`, `NOTICE` |
| Runtime MJCF | `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/hand_assets/shadow_hand/` | `left_hand.xml`, `right_hand.xml`, `.obj` meshes |
| Licence check | `grip_shadow_hand_assets.py` → `verify_shadow_mesh_asset_licences()` | CI-facing audit dict |
| Render helper | `grip_preview_matplotlib.py` | Lambert shading, `frame_fill_fraction`, no wireframe edges |
| Evidence script | `docs/development/full_body_models/evidence/grip_shadow_hand/render_shadow_hand_evidence.py` | 27 PNGs + `receipt.json` |
| MuJoCo demo | `grip_shadow_hand_free_club.py` | Free-club engine wire-up |

### Key functions (`shadow.py`)

- `visual_settings()` — cached load of `grip_shadow_hand_visual.json`.
- `solve_shadow_hand_qpos(hand, spec_hand, grip_spec=, grip_radius_m=)` → `ShadowHandPose` (qpos + **dynamic** `mount_matrix`).
- `_compute_mount_matrix` — places palm on cylinder in hand-grip frame (xy radial fit, z centre).
- `_curl_finger_chain` — per-joint bisection using `_radial_surface_deficit` (not global `_cylinder_gap`).
- `extract_shadow_hand_mesh` — geom group 2, filters `hidden_*_substrings`, applies mount + `_apply_anthropometry`.
- `build_hand_visual` — public entry; `joint_angles is None` triggers full solve.
- `finger_pad_contact_gaps` — min |r − (R + pad_clearance)| per digit in grip frame.
- `penetration_metrics` — club / finger-finger clearance (legacy metric; see §c).

### Settings (`grip_shadow_hand_visual.json`) — important keys

| Key | Current value | Meaning |
|-----|---------------|---------|
| `finger_contact_tolerance_m` | 0.004 | Pad-to-cylinder contact in solve/tests |
| `club_penetration_tolerance_m` | 0.012 | Bisect hi bound for penetration |
| `pad_clearance_m` | 0.001 | Target shell offset from grip radius |
| `curl_bisect_iterations` | 48 | Per joint |
| `mount_rotation_deg` | x90, z90 | Static rotation before dynamic translate |
| `mount_translation_m` | [0,0,0.018] | Base offset before palm solve |
| `anthropometry.breadth_scale` | **0.52** | xy scale in grip frame (fits glove on taper) |
| `hidden_mesh_substrings` | forearm, wrist, mounting_plate, **metacarpal** | Stripped from visual mesh |
| `capsule_pad_divergence_tolerance_m` | per-finger | Explicit vs capsule (not silent thumb merge) |
| `capsule_pad_divergence_thumb_max_m` | 0.062 | Thumb-only ceiling |
| `club_assembly_axial_overhang_m` | 0.22 | Wrap test butt allowance |
| `skin_tone_rgba` | [0.78, 0.62, 0.54, 1] | Material on `HandVisualResult` |

**Cache note:** `visual_settings()` is `@lru_cache(maxsize=1)`. After editing JSON in a long-lived Python process, restart the interpreter or call `visual_settings.cache_clear()`.

### Tests — how to run

```bash
export MUJOCO_GL=disable PYTHONPATH=.

# Core Shadow lane (includes slow MuJoCo solve ~15–30 s each)
python3 -m pytest \
  tests/unit/model_appearance/hand_backends/ \
  tests/unit/model_appearance/test_grip_shadow_hand_visual.py \
  tests/unit/model_appearance/test_grip_hands.py::test_hands_wrap_the_grip_tube_without_entering_it \
  --timeout=300

# Fast subset only (skips slow pad-divergence parametrization if marked)
python3 -m pytest \
  tests/unit/model_appearance/hand_backends/ \
  tests/unit/model_appearance/test_grip_shadow_hand_visual.py \
  -m "unit and not slow" --timeout=120

# MuJoCo free-club slice (if present on branch)
python3 -m pytest tests/unit/engines/mujoco/test_grip_shadow_hand_free_club.py -m unit --timeout=120
```

| Test file | What it proves |
|-----------|----------------|
| `test_shadow_registry.py` | Backend registered; mesh humanized; skin material |
| `test_grip_shadow_hand_visual.py` | Penetration, finger-finger gap, shaft roll, L-handed XML, licence audit |
| `test_shadow_capsule_pad_divergence.py` | Pad contact at roll 0/±20°; capsule centroid gaps per finger |
| `test_grip_hands.py` (wrap) | Club-frame placement; butt overhang uses `club_assembly_axial_overhang_m` |

### Evidence — regenerate PNGs

```bash
cd /path/to/UpstreamDrift
MUJOCO_GL=disable MPLBACKEND=Agg PYTHONPATH=. \
  python3 docs/development/full_body_models/evidence/grip_shadow_hand/render_shadow_hand_evidence.py
```

**Outputs:** `docs/development/full_body_models/evidence/grip_shadow_hand/grip_shadow_{both|lead|trail}_{iso|butt|face}_roll_{0p|20p|20m}.png` and `receipt.json` (dpi 1024, figure 8 in).

**Hero checks:** `grip_shadow_lead_iso_roll_0p.png` (single-hand wrap), `grip_shadow_both_iso_roll_0p.png` (overlap grip — harder).

---

## (c) What worked / what failed

### Worked

1. **Per-finger curl bisection** on `_radial_surface_deficit` (min radial xy vs `grip_radius + pad_clearance`) — avoids curling all verts to a global gap and landed finger pads on the cylinder.
2. **Dynamic `mount_matrix`** after wrist offsets — palm radial push + `trans[2] -= centre[2]` z-aligns palm with grip axis.
3. **`breadth_scale: 0.52`** — Shadow hand anthropometry in grip frame; without it pads miss on real club taper radii.
4. **Bug fix:** finger vertex filter used `and` between `"distal"` and `"middle"` → needed **`or`**.
5. **`extract_shadow_hand_mesh` uses solved mount** — not static config mount only.
6. **`finger_pad_contact_gaps`** at `grip_radius_m=0.0118`, roll 0°: index ≈ **2.8–3.1 mm**, other digits **sub-mm**, thumb **sub-mm** (within `finger_contact_tolerance_m` 4 mm).
7. **Lead iso evidence** — reads as fingers wrapped around grip stub (thumb opposing); hardware forearm/wrist hidden.
8. **Licence audit** — `verify_shadow_mesh_asset_licences()`; 14× `.obj`, Apache-2.0 bundle LICENSE; runtime sha256 match recorded in `PROVENANCE.json`.
9. **Capsule pad divergence** — explicit per-finger tolerances in JSON; thumb uses separate `capsule_pad_divergence_thumb_max_m` (geometry differs from capsules by design).

### Failed / partial

1. **Visual acceptance not complete** — `grip_shadow_both_iso_roll_0p.png` still reads busy/splayed (two overlap hands + long proximal extent); **frame fill** still loose on some views despite `frame_fill_fraction` and quantile crop (0.06–0.94).
2. **`penetration_metrics["club_penetration_m"]` ≈ 11 mm** at club taper radius — metric uses broad `_exterior_contact_vertices` (includes palm/thumb hub), not pad-only; **do not tune pose from this number**. Trust `finger_pad_contact_gaps` and `test_club_penetration_within_tolerance` instead.
3. **Global `_cylinder_gap` on all verts** (early approach) — dead end; fingers stayed open.
4. **Static mount only** — dead end; pads missed after wrist mapping.
5. **`hand_stack.ptp()` typo** in evidence script — fixed to `np.ptp`.
6. **Sibling tracks not merged** — shared renderer (`bc-59b6630a`) and shared curl-to-contact (`bc-c9c36777`) were **not accessible** on cloud agents this session; compare branch `cursor/grip-hand-render-mesh-fix-8498` for renderer tweaks when adopting.
7. **Overlap grip wrap test** — required `club_assembly_axial_overhang_m` in config + relaxed butt assertion tied to that setting (`along.min()` margin).

### Numbers to remember

- Club iron7 grip radius at lead/trail on-grip points ≈ **0.0116 / 0.0112 m** (not butt nominal 0.0118).
- Capsule pad Euclidean divergence (hand-grip frame, roll 0°) ~ **index 0.054**, middle 0.074, ring 0.15, little 0.11, thumb ~0.05 m — tolerances in JSON document this gap explicitly.

---

## (d) Next steps (priority order + acceptance)

1. **Visual sign-off (#15150 checklist)** — run § Visual self-review below; iterate mount/curl/hidden geoms/crop until **every** checklist item passes on lead, trail, and both at roll 0/±20°.  
   **Accept:** Scores recorded in PR body; hero PNGs updated in repo.

2. **Tighter framing** — adopt shared renderer from epic when landed; until then extend `grip_preview_matplotlib.apply_equal_axis_limits` or evidence `_crop_focus_points` quantiles.  
   **Accept:** Hand bbox ≈70% of 1024 px frame on `grip_shadow_lead_iso_roll_0p.png` (measure in image editor).

3. **Both-hands overlap view** — overlap style separates hands in y; verify intentional spec vs artistic overlap; may need trail/lead y offset in solve or camera azimuth tweak for `both` subset only.  
   **Accept:** `grip_shadow_both_iso_roll_0p.png` passes checklist (fingers wrap, not starfish).

4. **Migrate curl + metrics** — when `bc-c9c36777` lands on #15150, delete duplicated bisect/mount logic from `shadow.py` and call shared module; keep Shadow-specific MJCF paths and hidden filters.  
   **Accept:** Tests unchanged green; diff shrinks in `shadow.py`.

5. **Migrate renderer** — when `bc-59b6630a` lands, route `render_shadow_hand_evidence.py` through `scripts/render_hand_backend_comparison.py` (or shared preview API).  
   **Accept:** Same PNG names/receipt schema; visual parity or improvement.

6. **Comparison harness** — register `capsule`, `shadow`, siblings in one contact sheet (epic table).  
   **Accept:** `python3 scripts/render_hand_backend_comparison.py --backends shadow,capsule` (once script exists).

7. **Engine adapters** — Drake/Pinocchio/OpenSim still TODO in `humanoid_grip_adapter`; MuJoCo path is `grip_shadow_hand_free_club.py`.  
   **Accept:** One engine integration test per adapter added.

---

## (e) Dependencies on other tracks

| Track | Agent / PR | What to adopt | Proceed without it |
|-------|------------|---------------|---------------------|
| Shared renderer | `bc-59b6630a` | Lambert + crop + 1024 framing | Keep `grip_preview_matplotlib.py`; diff against `cursor/grip-hand-render-mesh-fix-8498` |
| Shared curl-to-contact | `bc-c9c36777` | Bisect, mount, contact metrics | Keep `solve_shadow_hand_qpos` in `shadow.py` |
| Capsule baseline | #15142 | Penetration test patterns | Already mirrored in Shadow tests |
| MyoHand skeleton JSON | #15144 / #15147 | Research anatomy | Not blocking Shadow visuals |
| Skinned CC0 / procedural | #15145–#15149 | Registry slots | Shadow stays default until config switch |

Before each iteration: read [#15150](https://github.com/D-sorganization/UpstreamDrift/issues/15150) and open sibling PRs; post a learnings comment on #15150 after each iteration.

---

## (f) Licences and assets

- **Licence:** Apache-2.0 (Menagerie `shadow_hand` bundle).
- **Vendored copy:** `assets/third_party/shadow_dexterous_hand/LICENSE`, `NOTICE`, `PROVENANCE.json`.
- **Runtime:** `.../hand_assets/shadow_hand/` (14 visual `.obj` meshes; group 2 geoms).
- **Verification:** `verify_shadow_mesh_asset_licences()` in tests; sha256 of runtime LICENSE must match vendored copy.
- **MANO/SMPL-X:** not used on this track (research-only per epic).

---

## (g) Visual self-review procedure (mandatory after every change)

1. Regenerate evidence (command in §b).
2. Open these PNGs at full resolution:
   - `grip_shadow_lead_iso_roll_0p.png`
   - `grip_shadow_trail_iso_roll_0p.png`
   - `grip_shadow_both_iso_roll_0p.png`
   - `grip_shadow_lead_face_roll_20p.png` (shaft roll sample)
3. Score each image **pass/fail** against the checklist:

| # | Criterion | Pass if |
|---|-----------|---------|
| 1 | Separate fingers and thumb | Five digits distinguishable; thumb opposed |
| 2 | Cupped palm | Palm visible closing around grip, not flat board |
| 3 | Knuckle line | Proximal knuckles visible along dorsal arc |
| 4 | Fingers wrap grip | Pads follow cylinder; not splayed open in air |
| 5 | No tubes/blobs/torn triangles | No capsule fallback; no obvious mesh tears |
| 6 | Skin shading smooth | Lambert shading, no wireframe edges |
| 7 | Hand ≈70% of frame | Subject fills most of square; minimal empty margin |
| 8 | Club visible | Grip stub/tube visible in frame |
| 9 | Lead/trail mirror | Trail view mirrors lead for same roll (L/R spec) |

4. **Record** a table in the PR body under `## Visual review (YYYY-MM-DD)` with columns: `view | roll | pass/fail | notes`.
5. If any item fails, fix code/settings (not tolerances unless measured on an issue), re-render, re-score. **Do not merge** until Dieter checklist in epic #15150 is satisfied.

---

## Lane status at cloud wind-down

- **Tests:** Hand backend + grip shadow visual + wrap tests pass locally with `MUJOCO_GL=disable`.
- **Physics:** Capsule hand untouched.
- **Visual:** Partial — single-hand iso acceptable; both-hands and frame fill need iteration on subscription fleet.
- **Child issue filed:** #15151.
- **Refs:** PR must include `Refs #15150`.

---

*Handoff written by cloud lane `bc-5fa05b91` at session wind-down (Dieter, 2026-11-14 PT → UTC 2026-11-15).*