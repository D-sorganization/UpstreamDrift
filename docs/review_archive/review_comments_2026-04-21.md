# Review Comments Archive - 2026-04-21

Generated: 2026-04-21T13:37:53.509928

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #2942: pyproject.toml:205

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Avoid enabling incomplete pinocchio stubs globally**

Adding `mypy_path = "stubs"` makes `stubs/pinocchio/__init__.pyi` the authoritative type surface for every `pinocchio` import, but that stub currently omits APIs used in `src` (for example `buildModelFromXML`, `Model.createData`, and others). Because CI runs `mypy src --config-file pyproject.toml` in `.github/workflows/ci-standard.yml`, this change causes ...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2942#discussion_r3120154795)

---

