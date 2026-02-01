---
description: Address 10 GitHub issues sequentially, creating and merging PRs one by one with CI/CD verification
---

# Fix 10 Issues Sequentially Workflow

Address up to 10 GitHub issues sequentially with autonomous PR creation and merging.

## Steps

// turbo
1. **Get open issues**:

   ```bash
   gh issue list --state open --limit 10 --json number,title,labels,body
   ```

// turbo
2. **For each issue, create branch**:

   ```bash
   git checkout main && git pull
   git checkout -b fix/issue-<NUMBER>-<description>
   ```

3. **Implement the fix**:
   - Read issue requirements
   - Make code changes
   - Run linting

// turbo
4. **Commit and push**:

   ```bash
   git add -A && git commit -m "fix: <description>

   Closes #<NUMBER>

   Co-Authored-By: Claude <noreply@anthropic.com>"
   git push -u origin fix/issue-<NUMBER>-<description>
   ```

// turbo
5. **Create PR**:

   ```bash
   gh pr create --title "fix: <description>" --body "Closes #<NUMBER>"
   ```

// turbo
6. **Wait for CI and merge**:

   ```bash
   gh pr checks <PR_NUMBER>
   gh pr merge <PR_NUMBER> --squash
   ```

// turbo
7. **Clean up and continue**:

   ```bash
   git checkout main && git pull
   git branch -d fix/issue-<NUMBER>-<description>
   ```

8. **Repeat** steps 2-7 for remaining issues.

## Output

Summary table of all issues resolved, PRs merged, and remaining open issues count.
