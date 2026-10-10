---
issue: 11932
summary: "Add a bounded native multi-DOF manifold BoxFDDP candidate with complete-state torque replay and a matched SciPy comparison."
branch: "feat/f05d-multidof-boxfddp-11932"
---

Add a quaternion-aware native BoxFDDP candidate with bounded two-motor
inputs, MuJoCo Euler derivatives, nonlinear fallback admission and
independent complete-state replay of applied ZOH torque. Compare it with
matched-budget SciPy shooting on source-hashed synthetic trials while keeping
F05 full-body, contact, muscle and capture qualification open.
