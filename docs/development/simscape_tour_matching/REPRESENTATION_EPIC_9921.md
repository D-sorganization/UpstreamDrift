# Native-Equivalent Multi-Engine Representation Epic

Parent: #9921. Native implementation: #9967. Related canonical convention
consolidation: #8867. User-expanded scope: preserve the Simscape R2025b model
while supporting native angles and numerically useful alternative representations
across Pinocchio, MuJoCo and Drake, with reliable motion/frame translations.

## Required Outcomes

1. One canonical conversion provider; reuse pose_interchange and existing frame
   utilities. Existing motion_pipeline consumers must not define competing conventions.
2. Explicit chart metadata: axis order, intrinsic/extrinsic convention, radians,
   quaternion coefficient order/sign policy, parent/child expression frames,
   winding/branch reference, coordinate order and physical model identity.
3. Convert orientations, angular velocities, accelerations and efforts. Include
   convective acceleration terms and dual torque maps preserving virtual work.
4. Support frame translations through existing rigid transforms and twist/wrench
   adjoints, including translation-induced moment terms and ordering conventions.
5. Expose native-angle and alternate-representation adapters in all three engines;
   preserve original constraints, inertias, attachments, gravity and actuator routing.
6. Explicitly distinguish storage/conversion adapters from alternative native
   integration models. Quaternion storage alone does not qualify spherical-joint
   replacement or eliminate a physical gimbal's noninvertible effort map.
7. Preserve native R2025b baseline and compare identical physical initial state and
   inputs on a representative trajectory, not only quaternion pose round trips.

## Sequential Acceptance Stages

- R0: inventory canonical APIs and engine-native bundles; identify overlaps with8867.
- R1: tested serial-rotation chart maps, inverse conditioning checks and native
  joint inventory. Root parallel agent owns this foundation and dedicated turnover.
- R2: engine adapters for Pinocchio, MuJoCo and Drake. Test per-engine native angle
  ordering, quaternion conversion, world/local conventions and scalar-rate semantics.
- R3: transformed velocities/accelerations/efforts pass manufactured motions,
  virtual-work invariance, finite-difference checks and cross-engine comparisons.
- R4: explicitly reviewed alternative integration representations. Qualify the
  model's actual gimbal input-axis maps; reject unsupported/singular inverses rather
  than silently setting rates, changing inertias or substituting joint mechanics.
- R5: replay representative source motions in every supported representation,
  compare marker transforms, constraints, physical velocities and energy/work;
  then verify against MATLAB R2025b with source/runtime hashes.
- R6: publish user-facing comparison/translation entry points, examples and
  executable turnover with exact artifact locations and known limitations.

## Verification and Handoff Rules

TDD, explicit finite/shape/unit/frame contracts, DRY canonical providers and narrow
adapter dependencies are required. Tests must include quaternion sign/winding,
all supported axis orders, nonzero angular rate convective terms, known transforms,
power invariance, masked capture data and explicit singularity behavior.
Every claim must identify whether it is mathematical conversion, adapter testing,
native-engine execution, or Simscape parity. Do not infer later stages from R1.
Archive immutable runtimes and evidence, save incremental commits, and update
HANDOFF plus DEVELOPMENT_LOG. Initial parallel ownership is joint_representations;
root owns matching experiments and shared root turnover.
