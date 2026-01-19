# Workflow and Claude Skills Report
Date: 2026-01-19
Scope: review of .github/workflows and recent git history (last 50 commits).

## 1. Current workflow automation snapshot
- CI Standard: quality gate (ruff, black, mypy, TODO/FIXME check, pip-audit, MATLAB quality report), tests with xvfb, cross-engine unit test, optional codecov.
- Nightly Cross-Engine Validation: scheduled, runs integration tests with deviation thresholds, opens issues on errors or warnings, uploads artifacts.
- Critical Files Guard: ensures root critical files exist and blocks deletions.
- Auto-Update PRs: auto-rebase PRs on main.
- Auto-Remediate Assessment Issues: scheduled or manual; can create PRs for ruff fixes, security headers, and API docs.
- Weekly CI Failure Digest: creates a weekly issue summarizing CI failures.
- PR Comment Responder: can use Jules or Claude API to address actionable PR comments.
- Jules automation suite: multiple specialized workflows (Auto-Repair, Test-Generator, Documentation Scribe, Conflict Fix, and more).

## 2. Git history themes (last 50 commit subjects)
Heuristic keyword counts based on recent commit subjects:
- fixes/bugs: 19
- tests: 9
- ci/workflows: 8
- refactor/perf: 5
- docs: 3
- security/config: 3

Observations:
- Heavy focus on fixes and CI automation.
- Frequent work on tests and quality tooling.
- Ongoing security and configuration adjustments.

## 3. Claude Skills recommendations
### CI/CD and quality
1. CI failure triage skill: parse CI logs, classify failure class, and suggest a fix or open an issue.
2. Lint and format assistant: run ruff and black on changed files and propose a patch.
3. Dependency and security updater: parse pip-audit output and open PRs with safe upgrades.
4. Test selection skill: map changed files to tests and run a fast subset automatically.
5. Coverage and regression summarizer: surface coverage drops and high-risk areas.

### Issue and PR operations
1. PR review summary skill: produce a risk analysis, test plan, and change summary.
2. Comment responder with patch application: extend the existing PR Comment Responder to auto-apply safe edits.
3. Release notes and changelog skill: generate CHANGELOG entries from merged PRs.

### Documentation and knowledge
1. Doc sync skill: update docs and API references when public APIs change (ties to the docs generator).
2. Runbook updater: keep AGENTS.md and copilot-instructions.md aligned with workflow changes.

### Domain-specific
1. Cross-engine validation summarizer: compare nightly results over time and flag drift trends.

## 4. CI/CD improvements worth considering
- CI Standard installs bandit and pydocstyle but does not run them; either add steps or remove the installs.
- Avoid duplicate test execution: the cross-engine unit test runs after full pytest. Exclude it from the general run or mark it to run only in the dedicated step.
- Enable pip caching in setup-python to reduce CI time.
- Add path filters to skip heavy tests on docs-only changes.
- Align Python version matrix with supported versions (3.10 and 3.11) if compatibility is a goal.

## 5. AGENTS.md improvement suggestions
- Sync workflow names and availability. AGENTS.md references jules-scientific-auditor.yml, while workflows include Jules-Physics-Auditor.yml; update names and remove stale roles if retired.
- Add a Claude Skills section: list available skills, triggers, and safety rules.
- Reduce duplication by linking to .github/copilot-instructions.md for generic guardrails; keep AGENTS.md focused on repo-specific rules.
- Provide a single "quality check" command (script or Makefile) to avoid drift in lint/test instructions.
- Document TODO/FIXME policy as "must map to a GitHub issue ID" to align with CI placeholder checks.

## 6. Suggested next steps
- Pilot the CI failure triage and PR review summary skills.
- Add caching and dedupe tests in CI Standard for faster feedback.
- Update AGENTS.md to reflect the current workflow suite.

## Appendix: sources reviewed
- .github/workflows/ci-standard.yml
- .github/workflows/critical-files-guard.yml
- .github/workflows/nightly-cross-engine.yml
- .github/workflows/auto-remediate-issues.yml
- .github/workflows/auto-update-prs.yml
- .github/workflows/ci-failure-digest.yml
- .github/workflows/PR-Comment-Responder.yml
- .github/copilot-instructions.md
- git log (last 50 commit subjects)
