# Security CI Branch Protection

Issue #3844's first wave makes security scanner failure handling explicit in
the core CI workflows. Branch protection for `main` should require these status
checks when they are present on a pull request:

- `CI Standard / Reject hosted runner routing`
- `CI Standard / Enforce action SHA pinning`
- `CI Standard / quality-gate`
- `CI Standard / tests`
- `CI Standard / shared-tools-consumer-contracts`
- `CI Standard / frontend-tests`
- `CI Standard / Rust Quickstart`
- `Docker Security Scan`
- `Spec Check`
- `Docs CI`

Security-specific expectations:

- The `quality-gate` job runs `scripts/check_workflows_no_silent_failures.py`.
- `pip-audit` waivers live in `scripts/config/pip_audit_waivers.json`, not
  inline workflow arguments.
- Waivers must include `id`, `package`, `reason`, and `expires_at`, and expired
  waivers fail before `pip-audit` runs.
- Security scanner setup and report generation must not use
  `continue-on-error: true`, `|| true`, or empty-report fallbacks that hide tool
  crashes.

CODEOWNERS currently names only `@dieterolson` for workflow and script review.
Do not add a second owner until the repository establishes a real GitHub team or
maintainer handle for security CI ownership.
