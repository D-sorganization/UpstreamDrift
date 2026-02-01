---
name: update-issues
description: Review recent assessments and sync GitHub issues - create missing issues and close resolved ones
---

# Update GitHub Issues from Assessments

Review code quality assessments from the last week and synchronize GitHub issues. Create issues for problems not yet tracked, and close issues that have been resolved.

## Instructions

**IMPORTANT**: Proceed continuously without user intervention until complete.

### 1. Find Recent Assessments

Search for assessment files from the last 7 days:

```bash
find . -type f \( -name "*assessment*" -o -name "*audit*" -o -name "*review*" \) \
  -mtime -7 -exec ls -la {} \;
```

Common locations:
- `.assessments/`
- `docs/assessments/`
- `reports/`
- Root directory markdown files

### 2. Get Current Open Issues

```bash
gh issue list --state open --limit 100 --json number,title,body,labels
```

### 3. Parse Assessment Findings

For each assessment file found:

#### a. Extract Issues/Findings
Look for sections like:
- "Issues Found"
- "Critical", "High", "Medium", "Low" priority items
- "Recommendations"
- "Action Items"
- Bullet points with problem descriptions

#### b. Categorize Each Finding
- **File/Location**: Where the issue exists
- **Description**: What the problem is
- **Priority**: Critical/High/Medium/Low
- **Type**: Bug, Security, Performance, Documentation, etc.

### 4. Cross-Reference with Existing Issues

For each finding from assessments:

#### a. Check if Issue Already Exists
Search existing issues for matching:
- File paths mentioned
- Similar keywords/descriptions
- Related labels

```bash
gh issue list --search "<keywords>" --json number,title
```

#### b. Determine Action
- **Not tracked**: Create new issue
- **Already exists**: Skip (note as already tracked)
- **Resolved but open**: Close the issue

### 5. Create Missing Issues

For findings not yet tracked:

```bash
gh issue create --title "<type>: <brief description>" \
  --body "## Summary
<description from assessment>

## Location
- **File**: \`<filepath>\`
- **Source**: <assessment filename>

## Priority
<Critical/High/Medium/Low>

## Suggested Fix
<if mentioned in assessment>

---
*Auto-generated from assessment review*" \
  --label "<priority>,<type>"
```

### 6. Check for Resolved Issues

For each open issue:

#### a. Check if Fixed
- Search codebase for the issue's file/location
- Check recent commits that might address it
- Review if the described problem still exists

```bash
git log --oneline --since="7 days ago" -- <filepath>
```

#### b. Close if Resolved
```bash
gh issue close <NUMBER> --comment "This issue has been resolved.

**Evidence**: <brief explanation of fix>
**Commit/PR**: <reference if found>

*Auto-closed by assessment review*"
```

### 7. Final Summary

```markdown
## GitHub Issues Sync Summary

### Assessment Files Reviewed
| File | Date | Findings |
|------|------|----------|
| <filename> | <date> | X items |

### New Issues Created
| Issue | Title | Priority | Source |
|-------|-------|----------|--------|
| #XXX | Title | High | assessment.md |

### Issues Closed (Resolved)
| Issue | Title | Resolution |
|-------|-------|------------|
| #XXX | Title | Fixed in commit abc123 |

### Issues Already Tracked
| Finding | Existing Issue |
|---------|----------------|
| Description | #XXX |

### Statistics
- **Assessments reviewed**: X
- **Total findings**: X
- **New issues created**: X
- **Issues closed**: X
- **Already tracked**: X
- **Current open issues**: X
```

## Success Criteria

- All assessment files from last 7 days reviewed
- New issues created for untracked findings
- Resolved issues properly closed
- No duplicate issues created
- Clear audit trail in issue comments

## Notes

- Use consistent labeling: `critical`, `high`, `medium`, `low`, `security`, `bug`, `enhancement`
- Always link back to source assessment in new issues
- Add comment explaining closure reason for resolved issues
- Skip findings that are too vague to actionize
