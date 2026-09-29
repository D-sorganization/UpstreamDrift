# Review Comments Archive - 2026-04-16

Generated: 2026-04-16T21:02:38.848540

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #2720: .pre-commit-config.yaml:44

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Use cross-platform interpreter for formatter hook**

The new `formatter-guidance-consistency` hook invokes `python3` via `language: system`, which will fail on setups that only provide `python` on PATH (a common Windows configuration). Since this hook is marked `always_run: true`, those contributors will be blocked from committing any change, not just formatter-related edits. Please make the hook interpreter-...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2720#discussion_r3097772501)

---
