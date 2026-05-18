# Review Comments Archive - 2026-05-07

Generated: 2026-05-07T22:45:03.366487

## Reviewer (chatgpt-codex-connector[bot]) (3 comments)

### PR #4417: src/tools/starting_pose_matcher/**init**.py:16

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Avoid importing core dependencies at package import time**

Importing any provider via `src.tools.starting_pose_matcher.providers...` executes this package `__init__`, which eagerly imports `.core` and its heavy optional dependencies (for example `pandas`). In environments that only need observed-input providers (without `gui-tools`), this causes immediate import failure before provider code runs, so the new ...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4417#discussion_r3206539605)

---

### PR #4417: src/tools/starting_pose_matcher/providers/openpose.py:322

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Do not mark derived OpenPose joints as missing by default**

`get_missing_keypoints` checks `torso`, `hub`, and `mp` against `get_confidence_map`, but that map only contains directly observed OpenPose keypoints and never adds confidences for derived joints. As a result, these joints are always reported missing even when `get_skeleton` successfully derives them, which produces incorrect quality diagnostics for...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4417#discussion_r3206539607)

---

### PR #4417: src/tools/starting_pose_matcher/providers/mediapipe.py:332

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Honor presence threshold in MediaPipe missing-landmark checks**

`get_missing_landmarks` only compares visibility against `visibility_threshold`, but `get_skeleton` requires both visibility and presence. If a landmark has high visibility but low presence, it is excluded from the skeleton while still being reported as present here, causing false negatives in missing-landmark reporting and letting low-quality d...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4417#discussion_r3206539612)

---
