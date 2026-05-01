# Review Comments Archive - 2026-05-01

Generated: 2026-05-01T03:36:23.210989

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #3625: scripts/ci/generate_portfolio_demo_output.py:42

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Expose non-default ball coefficients in the fixture contract**

This change replaces `BallProperties()` defaults with custom aerodynamic coefficients, but those new assumptions are not emitted in `golf_modeling_demo_output.csv` (which is presented as the reproducible input/output contract). As a result, consumers who rely on the documented fixture rows can no longer reproduce the same outputs, because critica...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3625#discussion_r3172882900)

---

