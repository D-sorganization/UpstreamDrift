# OpenSim Golf-Humanoid Model

This directory holds the **generated** OpenSim humanoid `.osim` model that
the OpenSim engine integrates into the cross-engine motion-matching
pipeline. The model is the joint-torque-actuated MVP body from
[`OPENSIM_PARITY_SPEC.md`](../OPENSIM_PARITY_SPEC.md) §3.

| File                 | Description                                    |
| -------------------- | ---------------------------------------------- |
| `golf_humanoid.osim` | Committed, generated model — do not hand-edit. |

## Provenance

- **Base model:** `Rajagopal2015_opensense.osim` from
  `opensim-org/opensim-models` (git submodule at
  `shared/models/opensim/opensim-models/`, commit `d9b05d4`).
  Path within the submodule:
  `Models/Rajagopal_OpenSense/Rajagopal2015_opensense.osim`.
- **Original publication:** Rajagopal et al. (2016),
  _Full-Body Musculoskeletal Model for Muscle-Driven Simulation of Human
  Gait_, IEEE TBME 63(10): 2068–2079.
  doi: [10.1109/TBME.2016.2586891](https://doi.org/10.1109/TBME.2016.2586891).
- **Why the OpenSense variant?** The OpenSense Rajagopal2015 model is a
  kinematics-focused descendant of the muscle-driven Rajagopal2016 model
  with the muscle force-set already removed. Starting from this variant
  avoids a brittle muscle-removal pass and keeps the build deterministic
  without requiring the OpenSim Python bindings at build time.

## Licence Audit

The upstream `opensim-org/opensim-models` repository is published by the
OpenSim Development Team. The repository itself does not ship a
top-level `LICENSE` file, but per the repository
[README](https://github.com/opensim-org/opensim-models) and the per-model
credits embedded in each `.osim` file, the models distribute under the
SimTK / OpenSim community licence. Concretely:

- The `Rajagopal2015_opensense.osim` `<credits>` field cites
  Rajagopal et al. (2016) as the model authors.
- The `Rajagopal2015_opensense.osim` model file does not contain any
  embedded restrictive licence header.
- The base Rajagopal model has historically been redistributed with
  modification (e.g. the OpenSense variant in this same submodule, the
  `RajagopalLaiUhlrich2023.osim` variant, scaling pipelines used by
  third-party labs); modification + redistribution as part of an open
  research codebase is the standard practice for these models.
- Sibling models in this same submodule (e.g. `arm26.osim`, see its
  `<credits>` field) are explicitly published under
  Creative Commons CC-BY 3.0.

**Net licence assessment for this MVP commit:** safe to redistribute as
part of UpstreamDrift, with attribution to Rajagopal et al. (2016).
The upstream submodule carries the canonical licence terms and is
referenced (not vendored) so any future licence changes upstream remain
visible. If a stricter per-model licence surfaces during the
peer-review-driven scientific roll-out, the parity spec § 7.2 fallback
plan (drop to `gait2392.osim` + hand-authored upper body) is unblocked
and tracked.

The licence-review note for this PR is recorded above; the project owner
sign-off is captured in the merging PR's review.

## Modifications Applied to the Base Model

The generator `scripts/build_humanoid_osim.py` reads
`Rajagopal2015_opensense.osim` and emits `golf_humanoid.osim` with the
following modifications:

1. **Model rename.** `OpenSense_Subject` → `golf_humanoid`.
2. **Shared club, two hands (OSV-9 #11756).** `msk_club.attach_club`
   adds a `Body name="Club"` built from the shared driver specification
   (`docs/development/full_body_models/full_body_spec_anthro_driver.json`):
   mass and inertia from `ClubDynamics.from_spec` (0.313 kg, club solids
   only), the shared head, shaft and grip meshes from
   `geometry/club/` (the same STLs as the generated models, head rolled by
   `ADDRESS_SQUARE_FACE_ROLL_DEG`), and grip frames at the shared
   `GripInterface` lead and trail grip points. Frames are `<components>` of
   their bodies: `Club/club_grip_offset` (lead), `Club/club_trail_grip_offset`,
   `Club/club_head_offset` (face centre, for FK) and
   `hand_l/hand_l_grip_offset`, `hand_r/hand_r_grip_offset`.
3. **Grip attachment.** `--grip-model weld` (default, matching the generated
   models' topology) welds the lead hand (`WeldJoint hand_l_to_club`) and
   closes the trail hand with `WeldConstraint hand_r_to_club`.
   `--grip-model bushing` hangs the club on `FreeJoint ground_to_club` with
   two `BushingForce`s at the grips (shared `GripInterface` stiffness,
   damping for zeta 0.7). The hand frames and address pose come from
   `msk_club_grip_calibration.json`, written by
   `python3 -m src.engines.physics_engines.opensim.python.msk_club_calibration
<models>`: inverse kinematics puts each hand's palm grip point on its
   shaft grip point (gap under 0.4 mm), lays the shaft diagonally across the
   palm, keeps the wrists in their physiological deviation range, follows
   the generated model's hip, shoulder and elbow centres at the captured
   address and plants the feet (the generated feet are unobserved). The
   default coordinates are that address pose, so the face is square there
   (0.18 deg by OpenSim FK).
4. **Joint-torque actuators on every DOF.** A `CoordinateActuator` is
   added to the `ForceSet` for every `Coordinate` in the model
   (39 in total: 6-DOF pelvis root, lower-limb chains incl. knee*beta
   coupled coordinates, lumbar 3-DOF, both shoulder/elbow/wrist chains).
   Naming convention: `tau*<coordinate_name>`. Each actuator has
`optimal_force=1`, `min_control=-Inf`, `max_control=+Inf`; the
polynomial torque controller (issue `OPENSIM-SIMULATE`) writes
   torques in N·m directly into the controls vector.

Muscles are **explicitly stripped for the MVP** — the OpenSense base is
muscle-free, and we do not re-introduce the Rajagopal2016 muscle set
here. The post-MVP muscle path is tracked by issue **#4134** and by
`OPENSIM_PARITY_SPEC.md` §8; the modifications in this directory are
deliberately structured so that re-grafting the muscle force-set is a
single ForceSet additive operation, with no rip-and-replace.

## Coordinate-Name Alignment With the Simscape Body Chain

The cross-engine parity spec ([CROSS_ENGINE_PARITY_SPEC.md](../../CROSS_ENGINE_PARITY_SPEC.md)
§2.6) requires the OpenSim coordinate names to round-trip with the
Simscape body chain. The Rajagopal coordinate names (which we keep
unchanged) align with the Simscape chain as follows; the runtime mapping
table lives in `python/opensim_golf/coordinate_map.py` (issue
`OPENSIM-COORD-MAP`).

| Body-chain segment      | Simscape DOF naming         | OpenSim coordinate(s)                                   |
| ----------------------- | --------------------------- | ------------------------------------------------------- |
| Root translation        | `pelvis_tx,ty,tz`           | `pelvis_tx`, `pelvis_ty`, `pelvis_tz`                   |
| Root rotation           | pelvis tilt/list/rotation   | `pelvis_tilt`, `pelvis_list`, `pelvis_rotation`         |
| Lumbar 3-DOF            | torso ext/bend/rot          | `lumbar_extension`, `lumbar_bending`, `lumbar_rotation` |
| Right hip               | hip flex/add/rot R          | `hip_flexion_r`, `hip_adduction_r`, `hip_rotation_r`    |
| Right knee              | knee R (+ patellar coupler) | `knee_angle_r` (+ coupled `knee_angle_r_beta`)          |
| Right ankle / foot      | ankle / subtalar / mtp R    | `ankle_angle_r`, `subtalar_angle_r`, `mtp_angle_r`      |
| Left hip / knee / ankle | mirror of right             | `*_l` analogues                                         |
| Right shoulder 3-DOF    | shoulder flex/add/rot R     | `arm_flex_r`, `arm_add_r`, `arm_rot_r`                  |
| Right elbow + forearm   | elbow / forearm pron/sup R  | `elbow_flex_r`, `pro_sup_r`                             |
| Right wrist 2-DOF       | wrist flex/dev R            | `wrist_flex_r`, `wrist_dev_r`                           |
| Left arm chain          | mirror of right arm         | `*_l` analogues                                         |

The Rajagopal model is **39-DOF generalized-coordinate-rich**; the
patellar `knee_angle_*_beta` coordinates are coupled to the knee angle
by `CoordinateCouplerConstraint`s preserved from the upstream model, so
the **independent DOF count is 37**. Aligning to the 23-DOF Simscape
chain (cross-engine spec §2.6) is the role of `coordinate_map.py`; this
artifact deliberately preserves the upstream Rajagopal naming so the
mapping helper can be implemented without touching the OSIM model.

## Regeneration Command

```bash
python3 scripts/build_humanoid_osim.py            # golf_humanoid + club of the scaled model
python3 scripts/build_humanoid_osim.py --grip-model bushing
python3 scripts/render_msk_club.py --geometry <opensim Geometry dir>  # under xvfb only
```

The base model path can be overridden with `UPSTREAMDRIFT_RAJAGOPAL_OPENSENSE`.
The builder is deterministic — running it twice produces a byte-identical
`golf_humanoid.osim`. CI may re-run this command and `git diff --exit-code`
to enforce that the committed artifact matches the script.

## Validation

`tests/test_opensim_model_loads.py` exercises the model in two layers:

- **Pure-XML structural assertions** (always run): topology checks
  (Club body present, lead WeldJoint and trail WeldConstraint, one CoordinateActuator per
  Coordinate, canonical Simscape-chain coordinate names present).
- **OpenSim binding load test** (`@pytest.mark.requires_opensim`):
  `osim.Model(path).initSystem()` succeeds and the joint / actuator
  counts match the topology. Skipped automatically when the OpenSim
  Python bindings are not installed.

## Full-Body Anthropometric Models (MS-40 #10339)

The `generated/` subdirectory contains pure-XML generated 44-coordinate
full-body OpenSim models produced from the canonical anthropometric
specifications (`full_body_spec_anthro_driver.json` and `full_body_spec_anthro_iron7.json`):

| File                                     | Description                                                                                                               |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `generated/full_body_anthro_driver.osim` | Full-body golfer with driver club, 44 coordinates, weld grip closure, foot Hunt-Crossley contact spheres, 34 tour markers |
| `generated/full_body_anthro_iron7.osim`  | Full-body golfer with 7-iron club, 44 coordinates, weld grip closure, foot Hunt-Crossley contact spheres, 34 tour markers |
| `generated/export_receipt.json`          | Hash provenance receipt linking spec sha256 to osim sha256 and topology metadata                                          |

### Exporter Command

```bash
python -m src.engines.physics_engines.opensim.python.full_body_osim \
  --spec docs/development/full_body_models/full_body_spec_anthro_driver.json \
  --out src/engines/physics_engines/opensim/models/generated/full_body_anthro_driver.osim \
  --receipt src/engines/physics_engines/opensim/models/generated/export_receipt.json
```

### Club Visuals (OSV-1 #11727, GCV-11 #11717)

Every generated full-body model carries three `<Mesh>` visual geometries on
the `Clubhead` body: shaft, grip rubber and a parametric clubhead. They come
from the shared adapter `src/shared/python/model_appearance/club_head_mesh.py`
(Tools parametric builder, with committed STLs under `assets/club_heads/` as
fallback) and are written to `geometry/club/` with a `provenance.json` that
records the spec hash, assembly parameters and a sha256 per file.

- Visual only: the meshes add no mass, inertia or contact. The `Clubhead`
  body mass properties are identical to a bare export (checked by
  `tests/unit/motion_matching/test_full_body_osim_club.py`).
- The saved models reference the files by bare name. OpenSim 4.6 does not
  resolve relative paths containing `..`, so loaders call
  `club_visuals.register_geometry_path()` first. In the OpenSim GUI, add
  `models/geometry/club` to File > Preferences > Geometry Search Path, or
  re-export with `--club-geometry-ref <absolute directory>`.
- `tour_matching/club_geometry.py::attach_visual_club` applies the same three
  meshes to `golf_humanoid.osim` through a `club_visual_frame` offset frame.
- Limitation: the visual head follows the specification solid placement; the
  head centre of mass in the dynamics model is not moved to the mesh centre.
- Regenerate with the exporter command above; `--no-club-geometry` produces a
  bare club.
