# Review Comments Archive - 2026-05-09

Generated: 2026-05-09T11:48:37.758408

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #4924: src/shared/python/pose_interchange/canonical.py:47

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Decouple canonical pose from optional pandas dependency**

Importing `REFERENCE_GOLFER_FIELDS` via `src.shared.python.motion_matching.diagnostics.reference_pose` pulls in the `motion_matching` package initializer first, which eagerly imports loader modules that require `pandas` (`motion_matching/loaders/c3d.py`). Because `pandas` is not in the core dependency set, environments installed without extras cannot ...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4924#discussion_r3213721780)

---

### PR #4924: src/shared/python/pose_interchange/canonical.py:126

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Preserve immutability of joint angles in frozen pose**

`CanonicalPose` is documented and tested as a frozen value object, but `joint_angles_deg` is stored as a plain mutable `dict`. Callers can mutate `pose.joint_angles_deg[...]` after construction, bypassing the constructor’s validation (finite values, known keys) and silently breaking invariants that downstream adapters may rely on.

Useful? React with 👍 /...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4924#discussion_r3213721782)

---

