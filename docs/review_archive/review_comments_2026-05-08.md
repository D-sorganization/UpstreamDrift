# Review Comments Archive - 2026-05-08

Generated: 2026-05-08T13:07:49.135685

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #4611: .github/workflows/Jules-Redundant-Issue-Closer.yml:40

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Read dry-run from workflow inputs instead of action inputs**

The dry-run gate is effectively always enabled, so this workflow never closes duplicate issues. `core.getInput('dry_run')` reads inputs passed to `actions/github-script` itself, but this step does not pass a `dry_run` input; on issue events it resolves to an empty string, making `core.getInput(...) !== 'false'` true and forcing the `if (dryRun) ret...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4611#discussion_r3211008247)

---

### PR #4611: .github/workflows/Jules-Redundant-Issue-Closer.yml:48

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Implement scheduled backstop scan instead of returning early**

The scheduled/manual backstop path is a no-op because runs without an `issues` payload hit this early return before any repository scan occurs. That means duplicates are only evaluated when the duplicate issue itself fires an event, so stale duplicates can persist indefinitely (for example, when the older canonical issue is edited/reopened but ne...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4611#discussion_r3211008252)

---

