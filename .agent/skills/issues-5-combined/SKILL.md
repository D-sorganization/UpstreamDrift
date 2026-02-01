---
name: issues-5-combined
description: Fix 5 GitHub issues in a single combined PR, iterating until CI/CD passes
---

# Fix 5 GitHub Issues in Combined PR

Address up to 5 GitHub issues from this repository in a single combined PR. Iterate on fixes until all CI/CD checks pass, then merge. Continue autonomously until complete.

## Instructions

**IMPORTANT**: Proceed continuously without user intervention until complete.

### 1. Get Open Issues

```bash
gh issue list --state open --limit 5 --json number,title,labels,body
```

Select up to 5 issues to address. Prioritize by:
1. Critical/high priority labels
2. Related issues that can be fixed together
3. Oldest issues

### 2. Create Feature Branch

```bash
git checkout main && git pull
git checkout -b fix/5-issues-batch-$(date +%Y%m%d)
```

### 3. Implement All Fixes

For each of the 5 issues:

#### a. Read and Understand
- Review issue description and requirements
- Identify affected files

#### b. Make Changes
- Implement the fix for each issue
- Keep changes organized by issue

#### c. Track Progress
- Note which issues have been addressed
- Document any issues that cannot be resolved

### 4. Run Local Quality Checks

```bash
ruff check . --fix
black .
# Run tests if applicable
```

### 5. Commit Changes

```bash
git add -A
git commit -m "fix: Address 5 GitHub issues

Issues addressed:
- #XXX: <brief description>
- #XXX: <brief description>
- #XXX: <brief description>
- #XXX: <brief description>
- #XXX: <brief description>

Closes #XXX, closes #XXX, closes #XXX, closes #XXX, closes #XXX

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 6. Push and Create PR

```bash
git push -u origin fix/5-issues-batch-$(date +%Y%m%d)

gh pr create --title "fix: Address 5 GitHub issues" --body "## Summary
This PR addresses 5 open issues:

| Issue | Title | Fix Description |
|-------|-------|-----------------|
| #XXX | Title | Brief fix description |
| #XXX | Title | Brief fix description |
| #XXX | Title | Brief fix description |
| #XXX | Title | Brief fix description |
| #XXX | Title | Brief fix description |

## Test plan
- [x] Local linting passes (ruff, black)
- [x] All issue requirements addressed
- [ ] CI/CD checks pass

Closes #XXX, closes #XXX, closes #XXX, closes #XXX, closes #XXX"
```

### 7. CI/CD Iteration Loop

```bash
gh pr checks <PR_NUMBER>
```

**If CI fails**:
1. Review failure logs: `gh run view --job <JOB_ID> --log-failed`
2. Fix the issues locally
3. Commit with message: `fix: Address CI feedback - <description>`
4. Push changes
5. Repeat until all checks pass

**Maximum iterations**: 5 fix cycles

### 8. Merge When Green

Once all CI/CD checks pass:

```bash
gh pr merge <PR_NUMBER> --squash
```

### 9. Clean Up

```bash
git checkout main && git pull
git branch -d fix/5-issues-batch-*
```

### 10. Final Summary

```markdown
## Combined Issues Resolution Summary

**PR**: #<PR_NUMBER>
**Status**: Merged

### Issues Resolved

| Issue | Title | Status |
|-------|-------|--------|
| #XXX | Title | Fixed |
| #XXX | Title | Fixed |
| #XXX | Title | Fixed |
| #XXX | Title | Fixed |
| #XXX | Title | Fixed |

**CI/CD Iterations**: X attempts until green
**Remaining open issues in repo**: Y
```

## Success Criteria

- Single PR addresses all 5 issues (or all available if fewer)
- All CI/CD checks pass before merge
- Issues are properly linked and closed
- No user intervention required

## Notes

- If an issue cannot be resolved, document it and exclude from PR
- If fewer than 5 issues exist, address all available
- Group related issues for cleaner commits
- Always rebase on main if it has been updated during iteration
