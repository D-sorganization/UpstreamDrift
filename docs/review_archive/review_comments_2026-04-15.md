# Review Comments Archive - 2026-04-15

Generated: 2026-04-15T07:15:29.899150

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #2684: src/shared/python/data_processing/processor.py:436

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Reinstate expression validation before calling DataFrame.eval**

This change removes `_validate_dataframe_expression(...)` and now forwards raw `expression` input directly to `df.eval(...)`, while also deleting the previous security rationale tied to issues #2065/#2349. That reopens the formula path to untrusted-expression abuse in any API/workflow that accepts user-supplied formulas, because the explicit AST...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2684#discussion_r3086987416)

---

