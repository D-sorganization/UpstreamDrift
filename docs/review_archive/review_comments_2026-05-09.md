# Review Comments Archive - 2026-05-09

Generated: 2026-05-09T15:45:04.294158

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #4934: src/shared/python/pose_interchange/services/_mock.py:110

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Apply pelvis SE(3) to mock landmark transforms**

`get_link_transforms()` currently derives landmarks from `forward_kinematics(angles)` and writes those points directly into each SE(3) translation, so any non-zero `pelvis_translation_m` / `pelvis_rotation_xyz_deg` in the last `set_pose()` is ignored. In scenarios where Pose Studio (or parity checks) places the canonical pose away from origin, the mock service...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4934#discussion_r3213770810)

---

### PR #4934: src/shared/python/pose_interchange/canonical.py:126

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Freeze joint-angle mapping to preserve CanonicalPose invariants**

`CanonicalPose` is declared frozen and validated on construction, but `__post_init__` stores `joint_angles_deg` as a plain mutable `dict`. Callers can mutate that mapping after construction and bypass all validation (including canonical field-name and finiteness checks), which undermines immutability guarantees and can produce invalid or non-d...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4934#discussion_r3213770814)

---

