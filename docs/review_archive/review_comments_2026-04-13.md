# Review Comments Archive - 2026-04-13

Generated: 2026-04-13T11:19:59.374737

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #2656: .github/workflows/ci-standard.yml:84

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Restore dynamic runner selection for required CI jobs**

This job now hardcodes `runs-on: self-hosted`, but the same workflow still has a `pick-runner` dispatcher that explicitly falls back to `ubuntu-latest` when no self-hosted runner is online (`.github/workflows/ci-standard.yml` lines 61-81). That fallback is no longer used, so when the self-hosted fleet is offline or saturated, required CI checks can rema...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2656#discussion_r3074407069)

---

### PR #2656: .github/workflows/docker-security-scan.yml:24

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Preserve hosted-runner fallback for Docker security scan**

Switching this workflow from `ubuntu-latest` to `self-hosted` removes the only execution path for environments without an online self-hosted runner, because this file has no `pick-runner`/fallback job. In that scenario, scheduled and PR-triggered Trivy scans stop executing, leaving vulnerability reporting coverage unavailable until self-hosted capaci...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/2656#discussion_r3074407073)

---

