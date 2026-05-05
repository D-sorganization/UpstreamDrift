# Review Comments Archive - 2026-05-05

Generated: 2026-05-05T12:33:21.616149

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #3965: src/engines/Simscape_Multibody_Models/3D_Golf_Model/MachineLearning/extract_dynamics_dataset.py:24

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Keep default dataset filenames consistent across pipeline**

The extraction script writes to `golf_dynamics_slim.parquet` by default, but `train_dynamics_surrogate.py` reads `golf_inverse_ready.parquet` by default. Running both scripts with defaults (as a typical first pass) makes training fail with a missing-file error unless users manually override one side. Aligning these defaults avoids a broken out-of-th...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3965#discussion_r3191027575)

---

### PR #3965: src/engines/Simscape_Multibody_Models/3D_Golf_Model/MachineLearning/README.md:232

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Invoke MATLAB helper directly instead of `matlab.` namespace**

The documented call uses `matlab.run_ml_polynomial_input_swing(...)`, but this function is stored in a normal `matlab/` folder, not a `+matlab` package. In MATLAB this namespace call does not resolve and users following the documented workflow will hit an undefined function/package error before simulation starts.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3965#discussion_r3191027578)

---

