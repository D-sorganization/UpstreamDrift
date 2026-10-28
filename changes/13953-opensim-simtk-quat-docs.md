---
issue: 13953
summary: "OpenSim: correct SimTK quaternion ordering documentation in coord_map (UD-12)"
branch: "fix/13953-opensim-simtk-quat-docs"
dl_state: "in_review"
next_step: "Merge PR"
---

Correct coord_map module and quat_eigen_to_canonical docstrings to clarify that native SimTK convertRotationToQuaternion produces scalar-first [w, x, y, z] quaternions rather than Eigen [x, y, z, w], and document the helper's purpose for external Eigen inputs (UD-12, #13953).
