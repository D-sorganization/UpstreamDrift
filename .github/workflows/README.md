# Jules Workflow Inventory

Auto-generated table of Jules automation workflows. See [issue #3065](https://github.com/D-Sorganization/UpstreamDrift/issues/3065) for cleanup context.

## Kill Switch

To pause **all** Jules automation immediately, either:

- Create the file `.github/WORKFLOWS_PAUSED` in the repo (any content), **or**
- Set the repository variable `WORKFLOWS_PAUSED` to `"true"` via the Actions UI.

The composite action `.github/actions/check-kill-switch` is called at the top of each
Jules worker workflow and will fail-fast if either condition is met.

Jules Control Tower also honours the `JULES_ENABLED` repository variable (legacy, managed
by `jules-kill-switch.yml`). See that workflow for enable/disable instructions.

## Jules Workflows

| File | Name | Triggers |
|------|------|----------|
| `Jules-Archivist.yml` | Jules Archivist | `workflow_call` |
| `Jules-Assessment-AutoFix.yml` | Jules Assessment Auto-Fix | `workflow_dispatch`, `schedule` |
| `Jules-Assessment-Generator.yml` | Jules Assessment Generator (Worker) | `workflow_call`, `workflow_dispatch` |
| `Jules-Assessment-Remediator.yml` | Jules Assessment Remediator | `workflow_dispatch` |
| `Jules-Auto-Assign-Issues.yml` | Jules Auto-Assign Issues | `issues` |
| `Jules-Auto-Repair.yml` | Jules Auto-Repair (Worker) | `workflow_call`, `workflow_dispatch` |
| `Jules-Code-Quality-Fixer.yml` | Jules Code Quality Fixer (Worker) | `workflow_call`, `workflow_dispatch` |
| `Jules-Code-Quality-Reviewer.yml` | Jules Code Quality Reviewer (Worker) | `workflow_call`, `workflow_dispatch` |
| `Jules-Comment-Processor.yml` | Jules Comment Processor | `workflow_call`, `workflow_dispatch`, `schedule` |
| `Jules-Completist.yml` | Jules Completist (Incomplete Implementation Auditor) | `workflow_call`, `workflow_dispatch` |
| `Jules-Comprehensive-Assessment.yml` | Jules Comprehensive Assessment | `workflow_call`, `workflow_dispatch` |
| `Jules-Conflict-Fix.yml` | Jules Conflict Resolver | `workflow_call` |
| `Jules-Consolidator.yml` | Jules Consolidator (Daily PR Merge) | `workflow_dispatch` |
| `Jules-Control-Tower.yml` | Jules Control Tower | `push`, `pull_request`, `workflow_run`, `schedule`, `workflow_dispatch` |
| `Jules-Critics-Comments.yml` | Jules Critics Comments Writer | `workflow_call`, `workflow_dispatch` |
| `Jules-Documentation-Auditor.yml` | Jules Documentation Auditor (Worker) | `workflow_call`, `workflow_dispatch` |
| `Jules-Documentation-Scribe.yml` | Jules Documentation Scribe (Worker) | `workflow_call` |
| `Jules-Hotfix-Creator.yml` | Jules Hotfix-Creator (Worker) | `workflow_call` |
| `Jules-Issue-Mention-Handler.yml` | Jules Issue Mention Handler | `issue_comment` |
| `Jules-Issue-Resolver.yml` | Jules Issue Resolver (Daily Priority Fixer) | `workflow_call`, `workflow_dispatch` |
| `Jules-Laymans-Terms-Writer.yml` | Jules Layman's Terms Writer | `workflow_call`, `workflow_dispatch` |
| `Jules-PR-AutoFix.yml` | Jules PR AutoFix (Direct Push with CI Verification) | `workflow_run`, `workflow_dispatch` |
| `Jules-PR-Cleanup.yml` | Jules PR Cleanup | `schedule`, `workflow_dispatch` |
| `Jules-PR-Compiler.yml` | Jules PR Compiler (Consolidate Open PRs) | `workflow_call`, `workflow_dispatch`, `schedule` |
| `Jules-Physics-Auditor.yml` | Jules Physics Auditor (Technical/Scientific Review) | `workflow_call`, `workflow_dispatch` |
| `Jules-Review-Fix.yml` | Jules Review Responder | `pull_request_review` |
| `Jules-Sentinel.yml` | Jules Sentinel (Security Audit) | `workflow_call`, `workflow_dispatch` |
| `Jules-Supersede-Check.yml` | Jules Supersede Check | `push`, `workflow_dispatch` |
| `Jules-Tech-Custodian.yml` | Jules Tech Debt Custodian | `workflow_call` |
| `Jules-Tech-Debt-Assessor.yml` | Jules Tech Debt Assessor (Worker) | `workflow_call`, `workflow_dispatch` |
| `Jules-Test-Generator.yml` | Jules Test Generator (Worker) | `workflow_call` |

### Archived (`.github/workflows/archived/`)

These workflows have been moved to the archive and are no longer active:

| File | Notes |
|------|-------|
| `Jules-Auto-Rebase.yml` | Archived |
| `Jules-Auto-Refactor.yml` | Archived |
| `Jules-Cleaner.yml` | Archived |
| `Jules-Competitor-Analyst.yml` | Archived |
| `Jules-Curie.yml` | Archived |
| `Jules-DRY-Orthogonality.yml` | Archived |
| `Jules-Hypatia.yml` | Archived |
| `Jules-Ideas-Generator.yml` | Archived |
| `Jules-Patent-Reviewer.yml` | Archived |
| `Jules-Render-Healer.yml` | Archived |

## Known Overlaps (from `ci-standard.yml` inventory comment)

| Duplicate pair | Canonical version | Status |
|----------------|-------------------|--------|
| `Jules-Code-Quality-Fixer` vs `Jules-Assessment-AutoFix` | `Jules-Assessment-AutoFix` (generates + fixes in one pass) | Review for merge pending |
| `Jules-Auto-Repair` vs `Jules-Hotfix-Creator` | Both kept — different strategies (direct push vs new branch) | Not redundant |
