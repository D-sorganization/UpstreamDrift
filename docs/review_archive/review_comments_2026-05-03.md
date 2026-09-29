# Review Comments Archive - 2026-05-03

Generated: 2026-05-03T19:29:19.271110

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #3944: Dockerfile:135

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Expand FORWARDED_ALLOW_IPS before passing to uvicorn**

The `CMD` uses Docker’s exec form, so `"${FORWARDED_ALLOW_IPS:-127.0.0.1}"` is passed to uvicorn as a literal string instead of being shell-expanded. That makes the new proxy trust setting effectively non-configurable at runtime (setting `FORWARDED_ALLOW_IPS` won’t change behavior), and forwarded headers from real reverse proxies will not be trusted as i...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3944#discussion_r3179164612)

---
