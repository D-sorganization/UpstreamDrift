# Review Comments Archive - 2026-04-29

<<<<<<< HEAD
Generated: 2026-04-29T06:48:42.047744

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #3440: src/robotics/planning/collision/_primitive_shapes.py:45

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Preserve ndarray input compatibility in contains_point**

Switching to `math.hypot(*(point - self.center))` narrows accepted input shapes and now throws `TypeError` when `point` is a non-1D ndarray (for example `(1, 3)` row vectors commonly produced by slicing/batching). The previous `np.linalg.norm(...)` accepted these ndarray forms, so this introduces a runtime regression for existing callers that pass 2D s...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3440#discussion_r3161491112)

---
=======
Generated: 2026-04-29T08:26:07.292160

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #3450: src/robotics/planning/collision/_primitive_shapes.py:45
>>>>>>> origin/main

Actionable: Yes
Has Suggestion: No

```
<<<<<<< HEAD
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Keep distance checks compatible with row-vector points**

Replacing `np.linalg.norm(point - self.center)` with `math.hypot(*(point - self.center))` changes accepted input shapes: a point shaped `(1, 3)` now raises `TypeError` because `math.hypot` receives a 1D array argument, while the previous implementation returned a valid scalar norm. This is a behavior regression for callers that pass row slices (e.g. `p...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3448#discussion_r3161810460)
=======
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Normalize point shape before unpacking into math.hypot**

This `contains_point` path no longer handles row/column vector inputs: `point` is only converted with `np.asarray`, so shapes like `(1, 3)` or `(3, 1)` reach `math.hypot(*(point - self.center))` and raise `TypeError` because unpacked elements are arrays, not scalars. Before this commit, `np.linalg.norm` accepted these common single-point layouts, so th...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3450#discussion_r3161871320)
>>>>>>> origin/main

---

>>>>>>> origin/main
