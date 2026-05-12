# Review Comments Archive - 2026-05-12

Generated: 2026-05-12T06:39:18.616981

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #5301: src/config/models.yaml:84

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Point golf suite tile to an allowed launcher entrypoint**

Using `launch_golf_suite.py` at the repository root makes this tile unlaunchable through `ProcessManager.launch_script`: `validate_script_path()` only allows scripts whose first path segment is in the whitelist (`src`, `engines`, `launchers`, `tools`, `shared`, `examples`). With the current path, launches fail with `Script in disallowed directory`, so...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/5301#discussion_r3226831341)

---

