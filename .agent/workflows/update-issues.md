---
description: Review recent assessments and sync GitHub issues - create missing issues and close resolved ones
---

# Update Issues Workflow

Sync GitHub issues with recent code assessments.

## Steps

// turbo
1. **Find recent assessments**:

   ```bash
   find . -type f \( -name "*assessment*" -o -name "*audit*" -o -name "*review*" \) -mtime -7
   ```

// turbo
2. **Get current open issues**:

   ```bash
   gh issue list --state open --limit 100 --json number,title,body,labels
   ```

3. **Parse assessment findings**:
   - Extract issues/findings from each assessment
   - Categorize by priority and type

4. **Cross-reference with existing issues**:
   - Check if finding already tracked
   - Determine if issues have been resolved

// turbo
5. **Create missing issues**:

   ```bash
   gh issue create --title "<type>: <description>" \
     --body "From assessment review" \
     --label "<priority>"
   ```

// turbo
6. **Close resolved issues**:

   ```bash
   gh issue close <NUMBER> --comment "Resolved - auto-closed by assessment review"
   ```

## Output

Summary table of assessments reviewed, new issues created, issues closed, and current open count.
