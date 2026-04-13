# Review Comments Archive - 2026-04-12

Generated: 2026-04-12T15:09:49.953645

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #2635: .github/workflows/ci-standard.yml:None

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P0 Badge](https://img.shields.io/badge/P0-red?style=flat)</sub></sub>  Remove invalid top-level FLEET_MODE workflow key**

This adds `FLEET_MODE` as a root-level key in the workflow file, but GitHub Actions only accepts specific top-level fields (`name`, `on`, `env`, `jobs`, etc.), so the parser rejects the workflow as invalid ("Unexpected value 'FLEET_MODE'"). In this commit the same trailing line is appended across many workflow files, which can block those workflows from loading...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2635#discussion_r3070198513)

---

