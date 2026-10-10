# Design Decisions for the Full-Body Showpiece, Volume 2

Continuation of [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md), which reached the 51,200-byte documentation budget at section 17. Numbering continues across volumes, and `load_design_decisions` in `src/shared/python/motion_matching/pipeline/design_decisions.py` reads `DESIGN_DECISIONS.md` and then `DESIGN_DECISIONS_<n>.md` in order. Add new sections to the highest-numbered volume; start the next volume when this one reaches the budget.

---

## 18. MyoSuite Neck Retarget: NeckInputY to `neck_flexion` (OSV-3, #11729)

### What

The anthro document neck joint is `Rx(NeckInputX) Ry(NeckInputY) Rz(NeckInputZ)` in the COMRod frame (`full_body_spec_anthro_driver.json`, joint `GolfSwing3D_Kinetic/Neck Joint`), and the head forward axis (towards `HeadFront`, offset `[0.1, 0, 0.16]` m on `Head`) is +x. So `NeckInputX` is lateral bending, `NeckInputY` is flexion/extension (head pitch) and `NeckInputZ` is axial rotation. The pinned myo_sim head chain (`head/assets/myohead_simple_chain.xml` at `33f3ded9`) has only `neck_rotation` (hinge about `[0.2, 1, 0]`) and `neck_flexion` (hinge about head z), and no lateral bending.

`src/engines/physics_engines/myosuite/python/coordinate_map_anthro.json` now maps:

| Source       | Target                                  | Sign |
| ------------ | --------------------------------------- | ---- |
| `NeckInputY` | `neck_flexion`                          | -1   |
| `NeckInputZ` | `neck_rotation`                         | +1   |
| `NeckInputX` | none (in `omitted_source`, with a note) | n/a  |

**Sign derivation (forward kinematics, MuJoCo).** Head forward-axis pitch `p = asin(f . up)` (positive = above the horizontal), central difference at `+/-0.2` rad from the neutral pose, all other coordinates zero:

- Native plant (`get_plant("mujoco", spec)`, `f` = `Head[0.1, 0, 0.16]` minus `Head[0, 0, 0.16]`, up = +z): `dp/dNeckInputY = -11.459 deg` per 0.2 rad (head pitches down). `NeckInputX` and `NeckInputZ` give 0. Analytically `Ry(b) x = (cos b, 0, -sin b)`, so the pitch change is `-b`.
- MyoSuite (`myobody_simpleupper.xml` at the pin, `f` = head body +x, up = -gravity): `dp/dneck_flexion = +11.459 deg` per 0.2 rad (head pitches up); `neck_rotation` gives 0.002 deg. The head +x axis is anterior: its cosine with the right foot calcn-to-toes direction is 0.99992.

Opposite pitch sensitivities of equal magnitude give `neck_flexion = -NeckInputY`. The magnitudes agree to 2e-7 deg, so the mapping is one-to-one on pitch at the neutral pose. Away from neutral the two chains differ: the hinge axes and joint centres are not co-located, and MyoSuite applies rotation before flexion.

### Why

The previous map sent `NeckInputX` (lateral bending) to `neck_flexion`, so a sideways head tilt was replayed as a nod. It also carried a secondary `NeckInputY -> neck_rotation` entry with weight 0.5. `retarget_frame` assigns rather than accumulates, so that entry overwrote the `NeckInputZ` contribution, and `neck_rotation` was effectively `0.5 * NeckInputY` (unit test `test_neck_input_z_alone_drives_neck_rotation` was red on the old map). Removing it makes `NeckInputZ` the only `neck_rotation` driver.

### What Was Tried and Rejected

- **Keeping `NeckInputX -> neck_flexion`** (the previous map). `NeckInputX` is lateral bending, so this replayed a sideways head tilt as a nod.
- **Keeping the 0.5-weighted `NeckInputY -> neck_rotation` entry.** It overwrote `NeckInputZ`, the only axial-rotation input.

### Limitations

- MyoSuite cannot represent neck lateral bending. `NeckInputX` is dropped, not absorbed into another coordinate.
- Other secondary (weighted) entries in the map have the same overwrite problem (for example `RScapInputY` 0.5 over `RSInputY` on `arm_flex_r`). That is outside this decision and tracked as a follow-up.
- Earlier MyoSuite replay receipts (`evidence/matched/driver_g1_myosuite/receipt.json`, 11 mapped coordinates, historical map) are historical and were not regenerated. They neither record nor depend on a hash of the map file.
- The map remains diagnostic (`qualification.diagnostic_only`). This is a kinematic sign/axis correction, not dynamics or gaze qualification for #11729.

### Reproduction

```bash
git clone https://github.com/MyoHub/myo_sim.git /path/to/myo_sim
git -C /path/to/myo_sim checkout 33f3ded946f55adbdcf963c99999587aadaf975f
MUJOCO_GL=egl python3 -m scripts.myosuite_neck_flexion_sign --myo-sim /path/to/myo_sim
python3 -m pytest tests/unit/engines/myosuite/test_retarget.py -q
```

### Evidence Receipt

- [`evidence/myosuite_neck/neck_flexion_sign.json`](evidence/myosuite_neck/neck_flexion_sign.json)
- Tests: `tests/unit/engines/myosuite/test_retarget.py` (`test_neck_input_y_drives_neck_flexion_with_fk_sign`, `test_neck_input_x_is_unmapped_and_documented`, `test_neck_input_z_alone_drives_neck_rotation`, `test_fixture_map_partitions_source_coordinates`).

---

## 19. MyoSuite Retarget Map Is One-to-One: Weighted Secondary Entries Dropped (#11729)

### What

`load_retarget_map` (`src/engines/physics_engines/myosuite/python/retarget.py`) now rejects any map in which two sources share a target, a source appears twice, or a sign is zero, and raises `ValueError`. The six weighted secondary entries in `coordinate_map_anthro.json` were removed, and their sources were added to `omitted_source` with notes:

| Source                                | Former entry       | Why it has no MyoSuite target                                                                                                                                                                            |
| ------------------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `HipInputX`, `HipInputY`, `HipInputZ` | 0.25 x `lumbar_*`  | They give the `LowerTorso` orientation relative to `world` (the pelvis root), not a lumbar angle. Adding them to the lumbar joints moves the trunk relative to the pelvis that the capture did not move. |
| `RScapInputY`, `LScapInputY`          | 0.5 x `arm_flex_*` | The scapula joint is `Rx(ScapInputX) Rz(ScapInputY)`, so `ScapInputY` is a rotation about z (protraction/retraction). `arm_flex` follows `SInputY`, a y-axis rotation, so the axes do not match.         |
| `RScapInputX`                         | 0.5 x `arm_add_r`  | `myobody_simpleupper` has no scapular DOF. The left side never had this entry, so the absorption was asymmetric.                                                                                         |

`LScapInputX` was already omitted. Its note used to say it was "absorbed via `arm_add_l`", but no such entry existed, so the note was corrected.

`retarget_frame`, `project_to_source` and `interpolate_unmapped` keep their one-to-one assignment semantics. `project_to_source` is now an exact inverse on mapped sources, because the loader guarantees a one-to-one map with non-zero signs.

### Why

`retarget_frame` assigns `out[target] = sign * q[source]` for each entry in order. Every secondary entry came after its primary, so on the shipped map it overwrote the primary. `SpineInputX`, `SpineInputY`, `TorsoInput`, `RSInputX`, `RSInputY` and `LSInputY` had no effect, and their targets were replayed as 0.25 x pelvis angle or 0.5 x scapula angle. Section 18 removed the same defect for `NeckInputY -> neck_rotation`.

Each mapped target is now driven only by its axis-matched primary. Inputs that MyoSuite cannot represent are listed as omitted rather than silently blended in, and the loader check stops the same overwrite from coming back.

### What Was Tried and Rejected

- **Accumulating weighted contributions** (`out[target] += weight * q[source]`). This would keep the intent of the old entries. But none of the weights (0.25, 0.5) was derived from kinematics, and the Hip and Scap axes do not correspond to the targets they were added to (see the table). Accumulation would also make `project_to_source` under-determined.
- **Reordering the entries so the primary is written last.** This hides the overwrite without removing it, and it silently discards the secondary input.

### Limitations

- MyoSuite replay still has no shoulder-girdle motion, and the pelvis root orientation is not driven. `_qpos_from_retarget` reads `pelvis_r*`, which is not in the target list, so the free-joint quaternion stays at identity. Absorbing scapula motion into the shoulder needs a forward-kinematics fit of the scapula and shoulder chain against the glenohumeral joint. It should not be added back as a fixed weight.
- Earlier MyoSuite replay receipts were produced with the overwriting map and are historical. They were not regenerated.
- The map remains diagnostic (`qualification.diagnostic_only`).

### Neck Torque Capacities

`src/shared/python/myofullbody/neck.py` `CAPACITY_NM` labelled `NeckInputX` as flexion/extension (30 N m) and `NeckInputY` as lateral bending (36 N m). Section 18 established that the anthro neck joint is `Rx(X) Ry(Y) Rz(Z)` with the head forward axis on +x, so X is lateral bending and Y is flexion. The capacities were swapped to `NeckInputX` 36 N m (lateral bending) and `NeckInputY` 30 N m (flexion, the smaller of flexion and extension). `NeckInputZ` stays at 15 N m. The MyoFullBody receipts under `evidence/myofullbody/` record the old `capacity_nm` and are historical until they are regenerated.

### Reproduction

```bash
python3 -m pytest tests/unit/engines/myosuite/test_retarget.py tests/unit/engines/myofullbody/test_myofullbody_neck.py -q
```

### Evidence Receipt

- LaTeX reference: [`myosuite_retarget_map.tex`](../../research/myosuite_retarget_map/myosuite_retarget_map.tex). The MyoFullBody neck capacities are also in [`myofullbody_swing.tex`](../../research/myofullbody_swing/myofullbody_swing.tex).
- Tests: `tests/unit/engines/myosuite/test_retarget.py` (`test_primary_source_alone_drives_its_target`, `test_fixture_map_targets_are_one_to_one`, `test_secondary_sources_are_omitted_and_documented`, `test_fixture_map_round_trips_mapped_sources`, `test_loader_rejects_two_sources_on_one_target`, `test_loader_rejects_duplicate_source`, `test_loader_rejects_zero_sign`); `tests/unit/engines/myofullbody/test_myofullbody_neck.py` (`test_capacities_follow_anthro_neck_axes`). All of them failed before the fix.
