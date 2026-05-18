# Motion Matching Documentation

Entry point for documentation covering UpstreamDrift's motion-matching
pipeline: target loading, cost terms, validation, and the desktop
preview tool.

## Contents

### Architecture decisions

- [ADR 0006 — Multi-Source Motion Targets](../adr/0006-multi-source-motion-targets.md)
  Decision record for the `ClubTarget` / `ClubBallTarget` / `BodyTarget`
  - `MultiSourceTarget` aggregator surface and the format-agnostic
    loader dispatchers.
- [ADR 0008 — Body-Part Visualisation Toolkit](../adr/0008-body-part-viz-toolkit.md)
  Decision record for the shared `body_part_viz` package that backs
  the C3D Viewer's Segments tab, the starting-pose matcher, and the
  URDF generator with one shape / fitter / renderer stack.

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
- [Body-Part Viz — Quickstart](../user_guide/body_part_viz/quickstart.md)
  Load a C3D, open the Segments tab, swap a line for a library shape,
  save.
- [Body-Part Viz — Custom Mesh Import](../user_guide/body_part_viz/mesh_import.md)
  Bring in a custom STL / OBJ / PLY / GLB; sizing, rest-pose fitting,
  performance considerations.
- [Body-Part Viz — Asset Author Guide](../user_guide/body_part_viz/asset_author_guide.md)
  Add a new shape to the bundled default library; manifest schema and
  procedural-generation script.

### API reference

- [`body_part_viz`](../api/body_part_viz.md) — public API surface for
  the shared shape / fitter / renderer toolkit.

### Related guides

- [Surrogate Training Guide](SURROGATE_TRAINING_GUIDE.md)
  Training the surrogate model that the matcher consumes.
