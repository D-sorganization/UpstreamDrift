---
name: issues-10-sequential
description: Address 10 GitHub issues sequentially, creating and merging PRs one by one with CI/CD verification
---

# Fix 10 GitHub Issues Sequentially

Address up to 10 GitHub issues from this repository sequentially. Each issue gets its own PR that must pass CI/CD before merging. Continue autonomously until all 10 issues are resolved or no more issues remain.

## Instructions

**IMPORTANT**: Proceed continuously without user intervention until complete.

### 1. Get Open Issues

```bash
gh issue list --state open --limit 10 --json number,title,labels,body
```

### 2. For Each Issue (Sequential Loop)

For issues 1 through 10 (or until no more issues):

#### a. Select the Next Issue
- Pick the highest priority issue (critical > high > medium > low)
- If no priority labels, pick the oldest issue

#### b. Create a Feature Branch
```bash
git checkout main && git pull
git checkout -b fix/issue-<NUMBER>-<short-description>
```

#### c. Implement the Fix
- Read the issue description carefully
- Make the necessary code changes
- Run local linting: `ruff check . --fix && black .`
- Run local tests if applicable

#### d. Commit and Push
```bash
git add -A
git commit -m "fix: <description>

Closes #<NUMBER>

Co-Authored-By: Claude <noreply@anthropic.com>"
git push -u origin fix/issue-<NUMBER>-<short-description>
```

#### e. Create Pull Request
```bash
gh pr create --title "fix: <description>" --body "## Summary
<brief description of changes>

## Test plan
- [x] Local linting passes
- [x] Changes address issue requirements

Closes #<NUMBER>"
```

#### f. Wait for CI/CD
```bash
# Poll CI status until complete
gh pr checks <PR_NUMBER>
```

- If CI fails: fix issues, commit, push, repeat until green
- Maximum 3 fix iterations per PR

#### g. Merge When Green
```bash
gh pr merge <PR_NUMBER> --squash
```

#### h. Clean Up
```bash
git checkout main && git pull
git branch -d fix/issue-<NUMBER>-<short-description>
```

#### i. Continue to Next Issue
- Repeat steps a-h for the next issue
- Track progress: "Completed X of 10 issues"

### 3. Final Summary

After completing all issues (or exhausting available issues), provide:

```markdown
## Issues Resolution Summary

| # | Issue | PR | Status |
|---|-------|-----|--------|
| 1 | #XXX - Title | #YYY | Merged |
| 2 | #XXX - Title | #YYY | Merged |
...

**Completed**: X issues
**Remaining open issues in repo**: Y
```

## Success Criteria

- Each PR passes all CI/CD checks before merging
- Issues are properly linked and closed
- Code follows repository standards
- No user intervention required during execution

## Notes

- If an issue cannot be resolved after 3 CI fix attempts, skip it and note in summary
- If fewer than 10 issues exist, complete all available issues
- Always verify main branch is up-to-date before starting each issue
