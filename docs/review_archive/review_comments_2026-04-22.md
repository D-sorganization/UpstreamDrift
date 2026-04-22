# Review Comments Archive - 2026-04-22

Generated: 2026-04-22T05:32:13.446175

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #2965: src/engines/physics_engines/putting_green/python/ball_roll_physics.py:81

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Handle non-1D velocity inputs before using math.hypot**

`BallState.__post_init__` converts inputs to `np.array` but does not enforce a 1-D shape, and callers like `PuttingGreenSimulator.set_state` can pass column vectors. With a `(2,1)` velocity, `self.velocity[0]`/`[1]` are 1-D arrays, so `math.hypot(...)` raises `TypeError` (`only 0-dimensional arrays can be converted to Python scalars`), whereas the previ...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2965#discussion_r3123828302)

---

