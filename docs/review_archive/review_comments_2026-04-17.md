# Review Comments Archive - 2026-04-17

Generated: 2026-04-17T02:50:11.409359

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #2742: docker-compose.yml:58

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Restore dependency bootstrap for frontend startup**

The frontend service now runs `npm run dev` directly, but it no longer installs dependencies first; on a fresh clone (or after `docker compose down -v`) `/app/node_modules` is empty, so startup fails with missing tooling (for example `vite` not found). This is a functional regression from the previous behavior where `docker compose up` could bootstrap itsel...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2742#discussion_r3099206877)

---

