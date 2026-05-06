# Review Comments Archive - 2026-05-06

Generated: 2026-05-06T00:13:01.360025

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #4053: src/engines/Simscape_Multibody_Models/3D_Golf_Model/MachineLearning/optimize_torque_sequence_for_club.py:426

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Compute non-zero orientation loss when orientation mode is used**

In `optimize_sequence`, the orientation branch and its `else` both set `orientation_loss` to zero, so `--cost-mode position_orientation` (and `full` when quaternions are present) never contributes any gradient. This makes `--orientation-weight` ineffective and silently prevents the optimizer from reducing orientation error even when quaternion...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4053#discussion_r3193618120)

---

### PR #4053: src/engines/Simscape_Multibody_Models/3D_Golf_Model/MachineLearning/optimize_torque_sequence_for_club.py:53

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Keep direct script execution compatible with default workflow**

These new top-level imports require `src` to be importable as a package, but this script is run directly in the documented workflow (`py .../optimize_torque_sequence_for_club.py`). In that mode, Python resolves imports relative to the script directory, so `src.shared...` is not found unless users preconfigure `PYTHONPATH`/install the repo packag...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4053#discussion_r3193618127)

---

