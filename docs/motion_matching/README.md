# Motion Matching Documentation

Entry point for documentation covering UpstreamDrift's motion-matching
pipeline: target loading, cost terms, validation, and the desktop
preview tool.

## Contents

### Architecture decisions

- [ADR 0006 — Multi-Source Motion Targets](../adr/0006-multi-source-motion-targets.md)
  Decision record for the `ClubTarget` / `ClubBallTarget` / `BodyTarget`
  + `MultiSourceTarget` aggregator surface and the format-agnostic
  loader dispatchers.

### Specifications

- [CLUB_IK_SPEC](../../src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/CLUB_IK_SPEC.md)
  Inverse-kinematics specification for the club chain shared between
  the MATLAB and Python loaders.

### Tools

- [Starting-Pose Matcher README](../../src/tools/starting_pose_matcher/README.md)
  PyQt6 desktop tool for previewing motion targets and aligning
  starting poses before optimisation.

### User guides

- [Loading Motion Targets](../user_guide/motion_matching/loading_targets.md)
  How to load club, ball-aware, and full-body targets from xlsx,
  `.mat`, and `.c3d` sources, including a worked example.

### Related guides

- [Surrogate Training Guide](SURROGATE_TRAINING_GUIDE.md)
  Training the surrogate model that the matcher consumes.
