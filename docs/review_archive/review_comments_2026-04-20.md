# Review Comments Archive - 2026-04-20

Generated: 2026-04-20T04:01:33.453014

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #2848: src/shared/python/biomechanics/humanoid_urdf_contracts.py:204

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Validate mirror links for right-side limbs too**

The symmetry check currently skips every mass entry that is not prefixed with `left_`, so a URDF that defines only `right_*` limb links can incorrectly pass with no `asymmetric_limbs` violations. Because this validator is intended to enforce bilateral symmetry, this misses a real contract failure whenever the model has right-only limbs (or right-side inertials...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2848#discussion_r3110001753)

---

### PR #2848: src/shared/python/biomechanics/humanoid_urdf_contracts.py:70

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Parse raw XML safely before filesystem existence check**

The parser calls `Path(source).exists()` for all string inputs, but valid raw XML strings (especially longer ones) can trigger `OSError` (e.g., filename too long) before XML parsing is attempted. Since the public API explicitly accepts raw XML text, this can cause `validate_humanoid_urdf` to crash on valid string inputs instead of returning a parse res...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2848#discussion_r3110001759)

---
