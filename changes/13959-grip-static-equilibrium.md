---
issue: 13959
summary: "Enforce both upper bound and zero transverse force in check_static_equilibrium (UD-30)"
branch: "fix/13959-grip-static-equilibrium"
dl_state: "in_review"
next_step: "Merge the PR into main."
---

Enforce Newtonian static equilibrium in GripContactModel.check_static_equilibrium: require support_ratio within tolerance of 1.0 (rejecting upward acceleration) and ensure transverse contact force magnitude orthogonal to gravity is negligible.
