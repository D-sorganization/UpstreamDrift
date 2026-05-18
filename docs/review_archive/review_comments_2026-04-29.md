# Review Comments Archive - 2026-04-29

Generated: 2026-04-29T20:21:49.574348

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #3498: pyproject.toml:185

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Move Ruff ignore list back under lint section**

Adding `[tool.ruff.lint.mccabe]` before `ignore = [...]` changes the TOML scope so `ignore` is parsed as a `mccabe` option, which Ruff does not support. In this state, any Ruff command fails to load the config (for example, `ruff check .` errors with `unknown field 'ignore', expected 'max-complexity'`), so linting/CI is effectively broken until `ignore` is plac...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3498#discussion_r3165433456)

---
