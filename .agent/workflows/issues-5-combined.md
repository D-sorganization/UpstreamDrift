---
description: Fix 5 GitHub issues in a single combined PR, iterating until CI/CD passes
---

# Fix 5 Issues Combined Workflow

Address up to 5 GitHub issues in a single PR with CI/CD iteration.

## Steps

// turbo
1. **Get open issues**:

   ```bash
   gh issue list --state open --limit 5 --json number,title,labels,body
   ```

// turbo
2. **Create feature branch**:

   ```bash
   git checkout main && git pull
   git checkout -b fix/5-issues-batch-$(date +%Y%m%d)
   ```

3. **Implement all fixes**:
   - Address each of the 5 issues
   - Run local linting

// turbo
4. **Run quality checks**:

   ```bash
   ruff check . --fix
   black .
   ```

// turbo
5. **Commit changes**:

   ```bash
   git add -A && git commit -m "fix: Address 5 GitHub issues

   Closes #XXX, closes #XXX, closes #XXX, closes #XXX, closes #XXX

   Co-Authored-By: Claude <noreply@anthropic.com>"
   ```

// turbo
6. **Push and create PR**:

   ```bash
   git push -u origin fix/5-issues-batch-*
   gh pr create --title "fix: Address 5 GitHub issues" --body "Combined fix PR"
   ```

// turbo
7. **Iterate until CI passes**:

   ```bash
   gh pr checks <PR_NUMBER>
   # Fix any failures, commit, push, repeat
   ```

// turbo
8. **Merge when green**:

   ```bash
   gh pr merge <PR_NUMBER> --squash
   ```

// turbo
9. **Clean up**:

   ```bash
   git checkout main && git pull
   git branch -d fix/5-issues-batch-*
   ```

## Output

Summary of all issues resolved, CI iterations required, and remaining open issues count.
