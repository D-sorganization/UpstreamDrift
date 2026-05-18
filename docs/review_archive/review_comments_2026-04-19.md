# Review Comments Archive - 2026-04-19

Generated: 2026-04-19T20:52:43.804377

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #2830: src/robotics/sensing/noise_models.py:374

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Handle shape changes across `measure` calls**

The state initialization short-circuits after the first call, so `_bias` and `_scale_factor` keep their original shape forever. If later `measure()` calls use a different `raw.shape`, NumPy broadcasting can silently return the wrong output shape (for example, first call with shape `(3,)`, later call with `(1,)` yields a `(3,)` result) or apply incorrect shared bi...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2830#discussion_r3108141090)

---
