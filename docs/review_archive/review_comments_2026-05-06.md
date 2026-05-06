# Review Comments Archive - 2026-05-06

Generated: 2026-05-06T15:32:19.385100

## Reviewer (chatgpt-codex-connector[bot]) (3 comments)

### PR #4090: src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option4_python_bridge/tests/conftest.py:21


Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Handle missing matlab package when checking marker availability**

`pytest_collection_modifyitems` calls `importlib.util.find_spec("matlab.engine")`, but on Python 3.10+ this raises `ModuleNotFoundError` when the parent `matlab` package is absent instead of returning `None`. On machines without MATLAB (the common CI/dev case this hook is meant to support), collection aborts with an INTERNALERROR before any sk...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4090#discussion_r3197765779)

---

### PR #4090: src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option4_python_bridge/fit_swing_python.py:104

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Validate theta0 before starting MATLAB engine**

`fit_swing_scipy` starts the MATLAB engine before checking whether `options.theta0` has the correct length. That means a simple user-input error can trigger an unnecessary (and expensive) engine startup, and on hosts without MATLAB it will raise an engine startup failure instead of the intended `ValueError` for bad `theta0`, making argument validation unusable ...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4090#discussion_r3197765789)

---

### PR #4105: src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/compute_skeleton_fk.m:203

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Remove extra torso-length step in angle fallback chain**

When `sensor_anchored` mode is unavailable (or `force_angle_chain=true`), the fallback now advances the torso axis three times by `UpperTorsoLength/2` (`hip->spine`, `spine->torso`, and `torso->hub`). This adds an extra `UpperTorsoLength/2` translation compared to the prior legacy geometry and systematically biases every downstream joint in fallback mo...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4105#discussion_r3197917606)

---
