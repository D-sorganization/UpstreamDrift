# Feasibility and Research Basis

## What the Images Can Determine

A silhouette constrains the projection of the visible surface. It does not
directly reveal internal joints, hidden limbs, depth, segment mass, or torque.
Several poses, cameras, body shapes, and control histories can explain nearly
identical pixels. Clothing and hair are not rigid anatomy. Thin, blurred clubs
need their own evidence channel; body overlap can hide important wrist motion.

Multiple synchronized calibrated views reduce ambiguity but do not remove it.
Even a visual hull cannot recover concavities invisible in all outlines. A
moving camera viewing different times is not a simultaneous multiview capture
of a deforming golfer. Videos of different swings may inform subject-shape
priors, but must never be triangulated as one swing.

These are design constraints, not reasons to abandon the feature. The useful
claim is: “This family of simulated motions is consistent with these images
under these camera, timing, body, and contact assumptions.”

## Failure Modes and Responses

| Risk                           | Observable Symptom                          | Required Response                                                         |
| ------------------------------ | ------------------------------------------- | ------------------------------------------------------------------------- |
| Depth/Camera/Scale Tradeoff    | Different 3D motions fit the same mask      | Multiple starts; camera/scale sensitivity; retain alternatives            |
| Timing Ambiguity               | Unknown film speed or slow-motion replay    | Keep physical-time hypotheses; suppress absolute speed/force claims       |
| Clothing/Geometry Compensation | Body shape changes to hide bad pose         | Fix subject shape across a swing; bound appearance offsets                |
| Hidden Limb/Club               | Good body IoU but wrong arm or shaft        | Visibility weights, club residuals, auxiliary keypoints, uncertainty      |
| Camera Pan/Zoom/Crop           | Apparent body translation follows camera    | Estimate shot-level camera track from scene evidence; preserve transforms |
| Contact/Actuation Mismatch     | Fit needs flying feet or root forces        | Audit contacts, grip, joint limits and actuators; reject infeasible fits  |
| Model Mismatch                 | Skeleton cannot represent visible head/legs | Capability gate; improve model under a separate reviewed task             |
| Optimizer Degeneracy           | Lower loss with implausible controls        | Bounds, normalized residuals, holdouts, physics gates                     |
| Domain Shift                   | Historical masks or confidence fail         | Manual correction; degradation benchmark; abstain                         |
| Archive Selection Bias         | Only favorable angles survive               | Report eligible, attempted, rejected, and accepted denominators           |

## Literature and Design Implications

The following primary sources were checked on 2026-09-14. They motivate the
plan; none qualifies Shadow Tracker or proves golf-specific performance.

- [Laurentini, The Visual Hull Concept for Silhouette-Based Image Understanding,
  1994](https://doi.org/10.1109/34.273735): foundational shape-from-silhouette
  reference. Use the visual hull as a geometric constraint, not recovered anatomy.
- [Shimada et al., PhysCap, 2020](https://arxiv.org/abs/2008.08880): combines
  kinematic inference and physical constraints for monocular capture. Its use of
  residual root forces motivates an explicit force audit here; plausible motion
  is not necessarily an unassisted forward-dynamics reconstruction.
- [Yuan et al., SimPoE, 2021](https://ye-yuan.com/simpoe/): simulation-based
  character control supports combining image evidence and dynamics. Treat learned
  control as an optional initializer, not independent evidence of historical motion.
- [Gartner et al., Trajectory Optimization for Physics-Based Reconstruction of
  3D Human Pose From Monocular Video, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Gartner_Trajectory_Optimization_for_Physics-Based_Reconstruction_of_3D_Human_Pose_From_CVPR_2022_paper.pdf):
  supports optimizing physical parameters and motor controls against video.
  Shadow Tracker adds silhouette observations to existing engine pathways.
- [SAM 2 Paper](https://arxiv.org/abs/2408.00714) and
  [Official Implementation](https://github.com/facebookresearch/sam2): candidate
  interactive video-mask provider. Pin weights, code, and license at adoption;
  compare on blurred clubs, occlusion, and archival film before selecting it.

The optimization stages, error gates, and deployment choices in this plan are
engineering proposals inferred from those methods and this repository's APIs.

## Decisions Before Expensive Implementation

ST-01 must measure whether the chosen model can express the required full body,
head, hands, club, feet, and grip. Current full-body work includes a rigid
torso-attached head limitation; body visuals and dynamics fidelity need separate
assessment. An upper-body-only or double-pendulum model can test software but
cannot qualify a full golfer reconstruction.

Use a torque-driven MuJoCo full-body adapter as the initial candidate because
the repo already has that model pathway. This is a provisional selection, not
a claim of installed runtime, superiority, or successful convergence. Compare
actual rollout fidelity and cost before freezing it. Pinocchio and Drake provide
useful additional checks; Simscape must be qualified in MATLAB R2025b.

For initial optimization, favor a bounded, reproducible numerical baseline over
training a new motion policy. Evaluate finite differences and derivative-free
methods on small problems; adopt differentiable rendering/dynamics only if the
gradient and contact approximations are verified. A differentiable renderer
alone does not make a black-box engine differentiable.

## Scientific Interpretation

Kinematics are inferred. Torques and contact forces are additionally conditioned
on masses, inertias, physical timing, contact laws, actuator models, and priors.
Muscle forces and co-contraction require further assumptions and evidence.
Compare a historical golfer's robustly observable motion features, with ranges;
do not present a single optimized torque trace as how that golfer actually
generated the swing. Unknown scale or physical time blocks SI kinetic claims.
