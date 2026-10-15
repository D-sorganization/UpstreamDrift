# Canonical Pose to Engine Adapters

## Responsibilities

`CanonicalPose` owns the static interchange record. `PoseConventionAdapter`
and concrete adapters own translation to and from native engine position vectors.
`JointSlot` records indices, angular units, signs and native joint limits.

## Data Contract

Static canonical pelvis translation is in metres. Pelvis intrinsic XYZ Euler
angles and joint angles are in degrees. Native joint slots declare `rad` or
`deg` and a sign of +1 or -1. Use adapters for conversion and ordering.
The dynamic canonical-v2 quaternion/velocity/acceleration layout is a separate
contract; do not treat the static pose as a complete simulation state.
Adapters that mix in `DynamicMapsMixin` also declare a `NativeDynamicLayout`
(base orientation encoding and the frames of the free-joint linear and angular
velocity) and expose canonical-v2 velocity, acceleration and generalized-force
maps. These maps leave static pose translation unchanged.

## Lifecycle and Failures

Validate the canonical record and model joint mapping before translating.
Round trips must preserve represented coordinates within test tolerances.
Invalid joint slot units, indices, signs or limits raise `ValueError`, as does
requesting a dynamic map for a multi-DOF joint slot.
Optional engine installation and model compatibility remain runtime concerns.

## Evidence

- [Static Pose](../../../src/shared/python/pose_interchange/canonical.py)
- [Adapter Protocol](../../../src/shared/python/pose_interchange/protocol.py)
- [MuJoCo Adapter Contract Tests](../../../tests/unit/pose_interchange/adapters/test_mujoco_protocol.py)
- [Convention Examples](../../user_guide/pose_studio/cross_engine_conventions.md)
- [Dynamic State Contract](../../conventions/canonical-v2.md)
- [Dynamic Maps](../../../src/shared/python/pose_interchange/dynamic_maps.py)

## Rationale

Keep index permutations and sign/unit conversions in adapters. Review affected
engine adapters and round-trip tests together when changing canonical fields.
