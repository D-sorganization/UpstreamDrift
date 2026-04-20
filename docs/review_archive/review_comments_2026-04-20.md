# Review Comments Archive - 2026-04-20

Generated: 2026-04-20T03:35:51.847253

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #2847: Dockerfile:184

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Restrict trusted forwarded IPs in runtime CMD**

The runtime command now sets `--forwarded-allow-ips "*"`, and `uvicorn --help` explicitly defines `*` as trusting proxy headers from every source (instead of the safer localhost default). If this image is run without a strictly enforced trusted reverse proxy path, clients can spoof `X-Forwarded-For`/`X-Forwarded-Proto`; since API rate limiting is keyed from rem...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2847#discussion_r3110002655)

---

