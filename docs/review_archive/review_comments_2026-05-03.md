# Review Comments Archive - 2026-05-03

Generated: 2026-05-03T11:13:38.450411

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #3932: src/shared/python/_contracts_decorators.py:None

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Route precondition eval failures through level policy**

This direct `ContractEvaluationError` raise bypasses `_handle_violation`, so `DBC_LEVEL=warn` no longer behaves as documented (`warn` should log without raising in `src/shared/python/contracts.py`). In practice, a malformed precondition (for example, a signature mismatch such as `lambda a, b: ...` on a single-arg function) now throws and aborts executio...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3932#discussion_r3178487123)

---

### PR #3932: src/shared/python/_contracts_decorators.py:None

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Preserve WARN semantics on postcondition eval errors**

Raising `ContractEvaluationError` here also skips `_handle_violation`, so evaluation-time postcondition errors now raise even when contracts are configured to `WARN`. That means callers running with warn-mode soft contracts can crash on condition bugs (e.g., `ZeroDivisionError` inside the postcondition) instead of getting a warning, which changes runtime...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3932#discussion_r3178487127)

---

