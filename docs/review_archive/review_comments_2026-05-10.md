# Review Comments Archive - 2026-05-10

Generated: 2026-05-10T12:17:00.539033

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #5112: tests/conftest.py:120

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Block all real HTTP entry points in unit network guard**

The autouse guard only monkeypatches module-level `get/post/put/delete/request`, which leaves common outbound paths like `urllib.request.urlopen`, `requests.Session.request`, and `httpx.Client.request` untouched. In unit-marked tests that use those APIs, real network calls will still execute, so the new policy is not reliably enforced and unit runs can...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/5112#discussion_r3215371991)

---

