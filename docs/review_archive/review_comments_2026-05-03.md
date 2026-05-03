# Review Comments Archive - 2026-05-03

Generated: 2026-05-03T07:15:43.282657

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #3931: scripts/ci/check_pip_audit_waivers.py:170

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Keep waiver diagnostics on stdout or check exit explicitly**

In the `Security Audit` workflow (`.github/workflows/ci-standard.yml`), waiver flags are read via `mapfile ... < <(check_pip_audit_waivers.py ...)`, and Bash does not propagate the subprocess exit code from process substitution. After this change moved stale/expired waiver messages to stderr, a failing waiver check can now produce no stdout flags, ...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3931#discussion_r3178230080)

---

