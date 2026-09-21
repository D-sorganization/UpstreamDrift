# OpenSim golf-humanoid model

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

## Licence audit

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

## Modifications applied to the base model

The generator `scripts/build_humanoid_osim.py` reads
`Rajagopal2015_opensense.osim` and emits `golf_humanoid.osim` with the
following modifications:

1. **Model rename.** `OpenSense_Subject` → `golf_humanoid`.
2. **Club rigid body.** A new `Body name="Club"` is appended to the
   `BodySet` with driver-class mass / inertia
   (`mass=0.32 kg`, `Ixx=Izz=0.139 kg·m²`, length `1.14 m`). These
   values are MVP placeholders sized to keep the integrator stable; the
   canonical golf-club anthropometric YAML (issue
   `PARITY-DIMENSIONS`) will replace them once that lands.
3. **Rigid grip attachment.** A `WeldJoint name="hand_r_to_club"`
   joins `hand_r` → `Club` via two `PhysicalOffsetFrame`s
   (`hand_r_grip_offset` on the hand, `club_grip_offset` on the club
   origin). Per `OPENSIM_PARITY_SPEC.md` §3.4 we use a `WeldJoint`
   (Option A) rather than a `WeldConstraint`: same rigid-attachment
   semantics, zero added DOFs, faster integration. A third frame
   `club_head_offset` exposes the clubhead position for FK extraction.
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

## Coordinate-name alignment with the Simscape body chain

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

## Regeneration command

```bash
python3 scripts/build_humanoid_osim.py
```

The builder is deterministic — running it twice produces a byte-identical
`golf_humanoid.osim`. CI may re-run this command and `git diff --exit-code`
to enforce that the committed artifact matches the script.

## Validation

`tests/test_opensim_model_loads.py` exercises the model in two layers:

- **Pure-XML structural assertions** (always run): topology checks
  (Club body present, WeldJoint to hand_r, one CoordinateActuator per
  Coordinate, canonical Simscape-chain coordinate names present).
- **OpenSim binding load test** (`@pytest.mark.requires_opensim`):
  `osim.Model(path).initSystem()` succeeds and the joint / actuator
  counts match the topology. Skipped automatically when the OpenSim
  Python bindings are not installed.
