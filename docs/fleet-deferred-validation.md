# Fleet Deferred Validation

Use this procedure when an issue depends solely on an external provider, an
unavailable upstream service, or work outside this repository's authority.

1. Preserve the issue's original acceptance criteria and current claim state.
2. Verify that the required work belongs to the owning repository or external
   provider, and record the evidence and a durable reference in the issue.
3. Merge and validate any in-repository coordination or documentation change
   before resolving the issue.
4. Close an eligible external-only issue as `not_planned` with the `roadmap`
   label. Do not use this outcome for an issue that still has an actionable
   in-repository fix or protected ownership.

This procedure does not override a repository-specific `AGENTS.md`, an active
claim, or a `do-not-automate` label.
