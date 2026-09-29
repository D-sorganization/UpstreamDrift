# Review Comments Archive - 2026-04-17

Generated: 2026-04-17T21:48:41.209305

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #2746: scripts/assess_repository.py:186

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Restore repository-wide secret scan coverage**

`assess_F()` now limits the hardcoded-secret scan to `src/**/*.py`, so any Python code in `scripts/`, tooling, or other non-`src` paths is no longer evaluated. This creates false negatives in the security assessment output (it can report no findings even when a hardcoded secret exists outside `src`), which is a regression from the previous repository-wide behavi...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2746#discussion_r3099641884)

---

### PR #2746: scripts/assess_repository.py:185

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Match common `*_key` secret assignments in regex**

The new pattern requires exact identifiers `password`, `secret`, `api_key`, or `token` directly before `=`, so assignments like `secret_key = "..."`, `private_key = "..."`, or `access_key = "..."` are not detected anymore. Because `*_key` names are common for credentials, this change introduces a blind spot in the secret heuristic and can under-report real h...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2746#discussion_r3099641890)

---
