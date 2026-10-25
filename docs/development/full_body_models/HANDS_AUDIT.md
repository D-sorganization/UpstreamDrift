# Hands Audit

This audit covers every model id in [`grip_capability_inventory.json`](../../../src/config/grip_capability_inventory.json). The table reports the per-asset capability record; `No` means the capability is not available for that model asset. A rigid proxy does not imply articulated fingers. `Finger DOF` counts independent finger joints, excluding wrist joints.

The specification for a future full-body articulated-finger hand option,
including the capsule, frozen-phalanx, and MyoHand tradeoffs and required
acceptance tests, is [ADR-0057](../../adr/0057-hrg-hx5-articulated-hand-option.md).
It is a docs-only decision; it does not qualify the separate MyoHand research
fixture for product use.

## Per-Asset Hand Capability Matrix

| Model ID | Hands and Finger DOF | Club Attachment | Pad-Grip Contact | Physical Hand-Hand Contact | Native Visual Styles | Asset Path |
| --- | --- | --- | --- | --- | --- | --- |
| `mujoco/driver` | Club-attached hand meshes; 0 finger DOF in source spec | Weld | Yes | No | No | `docs/development/full_body_models/full_body_spec_anthro_driver.json` |
| `mujoco/iron` | Club-attached hand meshes; 0 finger DOF in source spec | Weld | Yes | No | No | `docs/development/full_body_models/full_body_spec_anthro_iron7.json` |
| `mujoco/legacy-golfer` | Left and right hand bodies; 1 hinged finger proxy per hand | Weld | No | No | No | `src/engines/physics_engines/mujoco/models/generated/golfer.xml` |
| `mujoco/shadow-allegro-lab` | Shadow Hand; 22 finger DOF (24 joints total including 2 wrist DOF) | Mocap weld | No | No | No | `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/hand_assets/shadow_hand/right_hand.xml` |
| `drake/driver` | Club-attached hand meshes; 0 finger DOF in source spec | Weld | Yes | No | No | `docs/development/full_body_models/full_body_spec_anthro_driver.json` |
| `drake/iron` | Club-attached hand meshes; 0 finger DOF in source spec | Weld | Yes | No | No | `docs/development/full_body_models/full_body_spec_anthro_iron7.json` |
| `drake/legacy-golfer` | Hand links with 1 hinged finger proxy per hand | Weld | No | No | No | `src/engines/physics_engines/drake/models/generated/golfer.urdf` |
| `pinocchio/driver` | Club-attached hand meshes; 0 finger DOF in source spec | Weld | Yes | No | No | `docs/development/full_body_models/full_body_spec_anthro_driver.json` |
| `pinocchio/iron` | Club-attached hand meshes; 0 finger DOF in source spec | Weld | Yes | No | No | `docs/development/full_body_models/full_body_spec_anthro_iron7.json` |
| `pinocchio/legacy-golfer` | Hand links with 1 hinged finger proxy per hand | Weld | No | No | No | `src/engines/physics_engines/pinocchio/models/generated/golfer.urdf` |
| `opensim/driver` | Club-attached hand meshes; 0 finger DOF in source spec | Weld | Yes | No | No | `src/engines/physics_engines/opensim/models/generated/full_body_anthro_driver.osim` |
| `opensim/iron` | Club-attached hand meshes; 0 finger DOF in source spec | Weld | Yes | No | No | `src/engines/physics_engines/opensim/models/generated/full_body_anthro_iron7.osim` |
| `opensim/native-golf-humanoid` | No hand bodies or finger DOF | Free club with bushing | No | No | No | `src/engines/physics_engines/opensim/models/golf_humanoid.osim` |
| `myosuite/driver` | Hand grip sites; no finger DOF | Site weld | No | No | No | `shared/models/myosuite/golf/body/golfer_myobody_driver.xml` |
| `myosuite/iron` | Hand grip sites; no finger DOF | Site weld | No | No | No | `shared/models/myosuite/golf/body/golfer_myobody_iron.xml` |
| `myosuite/myohand` | Myo arm hand; no finger DOF in this asset | None | No | No | No | `src/engines/physics_engines/mujoco/myo_sim/arm/myoarm_simple.xml` |
| `simscape/driver` | No hand bodies; club meshes carry the hands; 0 finger DOF | Bushing | No | No | No | `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/mdl_reference/GolfSwing3D_Kinetic.mdl` |
| `simscape/iron` | No hand bodies; club meshes carry the hands; 0 finger DOF | Bushing | No | No | No | `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/mdl_reference/GolfSwing3D_Kinetic.mdl` |
| `jaxsim/backend` | No hand bodies or finger DOF | Bushing | No | No | No | `src/engines/physics_engines/jaxsim/jaxsim_backend.py` |
| `humanoid_character_builder` | Hand bodies; no finger DOF in inventory | Weld | Yes | No | No | `src/shared/python/humanoid_character_builder/spec_export.py` |
| `mujoco_humanoid` | Rigid hand proxies; no finger DOF | Weld | No | No | No | `src/shared/python/model_generation/library/bundled/mujoco_humanoid/humanoid.xml` |

The flagship anthropometric driver and 7-iron assets share the same base full-body specification across the MuJoCo, Drake, Pinocchio and OpenSim inventory entries. That specification contains the club and `GolfSwing3D_Kinetic/Grip/RHandStandoff`, but no separate hand bodies; the hand meshes are visual attachments on the club. The Simscape kinetic model has the same limitation. Its trail-wrist endpoint is `RHandStandoff`. These are visual representations and do not establish hand mass, finger dynamics or independent hand contact.

The legacy MuJoCo model has one hinge for each finger proxy and a club weld to `hand_right`; its inventory attachment label is `weld`. The legacy Drake and Pinocchio URDFs also contain one revolute finger-proxy joint per hand. The Shadow Hand file contains 24 named joints: 22 finger joints plus two wrist joints. Other records use the inventory's declared finger mode; `none` means no finger articulation is represented, and `rigid_proxy` denotes non-articulated finger geometry unless the asset path above shows a jointed proxy.

The contact columns distinguish pad-to-grip contact from physical hand-to-hand contact. The shared grip topology can describe hand-hand pairs, but the inventory reports no physical hand-hand contact implementation for any listed asset. `allocated_hand_hand` is a separate inventory capability and is also unavailable for every engine entry. Shared `GripPose` geometry does not imply that an engine's native exporter supports visual grip-style switching.

## Wrist Frame Contract

In the full-body specification, `child_to_follower` is a homogeneous transform. Let `N` be its native rotation block and `G(e)` the extrinsic x-y-z rotation matrix from `subject.grip_rotation_deg`. The builder composes the wrist rotation as:

```text
N @ R_x(25 deg) @ G(e)
```

The grip-only offset is `M = R_x(25 deg) @ G(e)`. Matrices act on column vectors and compose right to left, so `G(e)` acts first, followed by the fixed 25-degree x rotation and the native wrist rotation `N`. The translation block is preserved by the grip contract. The `25 deg` term is the fixed grip ulnar offset; it is part of the transform convention, not an extra fit parameter. See [`grip_contract.py`](../../../src/shared/python/motion_matching/grip_contract.py) for the executable convention and validation.

## Length Constants and GripPose Naming

Three source constants currently describe hand length or hand-segment geometry with different values and roles:

| Source | Value | Meaning |
| --- | ---: | --- |
| `src/shared/python/core/physics_constants.py:237` | `0.19 m` | Anthropometric average hand length |
| `src/shared/python/grip_contact/contact_topology.py:43` | `0.19 m` | Reference hand length for scaling contact sites |
| `src/shared/python/optimization/model_provider.py:180` | `0.08 m` | Optimization hand segment along the club axis from the wrist |

These constants are not interchangeable: the first two are reference hand lengths, while the third is a simplified optimization segment length. Consumers must not assume one is a substitute for another.

`GripPose.grip_strength_deg` is also a naming collision: it is a visual roll/azimuth offset around the shaft used by `GripPose.placement()`, not physical grip strength. Physical wrist rotation uses `subject.grip_rotation_deg` and the frame contract above. Do not interpret the visual field as force, torque or a physical contact parameter.

## Evidence and Limits

The row identifiers are checked against `src/config/grip_capability_inventory.json` by `tests/unit/full_body_models/test_hands_audit_inventory.py`. Capability statuses and asset paths are copied from that registry; joint counts noted above were checked in the listed MJCF/URDF/XML source files. This is a source audit, not evidence of runtime contact behavior or human grip performance.
