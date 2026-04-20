# Review Comments Archive - 2026-04-20

Generated: 2026-04-20T12:44:01.402230

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #2862: src/engines/physics_engines/drake/python/drake_physics_engine.py:452

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Keep suite spatial Jacobian row order as angular then linear**

Changing `spatial` to `np.vstack([jacp, jacr])` flips the suite row order to `[linear; angular]`, but the repository still defines the canonical contract as `[angular; linear]` (`src/shared/python/engine_core/capabilities.py:27-34`, `SPEC.md:519`). This makes the top 3 rows no longer represent angular terms for consumers that rely on the document...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2862#discussion_r3113241867)

---

