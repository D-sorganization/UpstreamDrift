# PR Cleanup Strategy

## Problem Analysis

**Current State:**
- 116 total open PRs across 9 repos
- Golf Suite: 95 PRs (82% of total!)
- Gasification: 8 PRs  
- All others: 1-2 PRs each

**Root Cause:**
Jules automation creates individual PRs for each fix attempt but doesn't close superseded PRs when creating new ones. This results in PR explosion:
- PR #668: Fix cv2 import
- PR #669: Fix mypy errors
- PR #670: Fix mypy errors (again)
- PR #671: Remove TODO placeholders
- etc.

Each PR attempts to fix the CI failure from the previous PR, but the old PRs are never closed.

## Solution Strategy

### Immediate Actions (Round 5)

1. **Identify Current "Good" PRs** (our consolidated PRs):
   - MLProjects #57 - chore(ci): standardize linting tool versions
   - Golf Suite #624 - chore(ci): workflow standardization (MISSING - need to find)
   - AffineDrift #604 - chore(ci): workflow standardization
   - Gasification #777 - fix(lint): workflow standardization
   - Playground #68 - chore(ci): standardize linting tool versions
   - Games #199 - chore(ci): standardize linting tool versions
   - Repository_Management #59 - chore(ci): standardize linting tool versions
   - MEB_Conversion #16 - feat(workflows): add missing workflows
   - Tools #302 - docs: Update test coverage

2. **Close All Jules-Generated PRs** in Golf Suite and Gasification:
   - These are all superseded by our consolidated PRs
   - Criteria: author = google-labs-jules, created in last 24 hours
   - Action: Close with comment "Superseded by comprehensive PR #XXX"

3. **Verify Our Consolidated PRs Exist**:
   - Golf Suite #624 might have been lost
   - Need to verify all repos have exactly ONE open PR (our consolidated one)

### Long-term Prevention

1. **Update Jules Workflow**:
   - Add step to close previous PR when creating new fix PR
   - Use branch strategy: update existing branch instead of creating new ones
   - Add PR consolidation step after multiple fixes

2. **Add PR Cleanup Automation**:
   - Daily job to close stale auto-generated PRs
   - Close PRs superseded by newer PRs on same files

3. **Change Fix Strategy**:
   - Instead of: Create PR → CI fails → Create new PR → repeat
   - Use: Create PR → CI fails → Push fix to same PR → repeat

## Execution Plan

### Step 1: Find Golf Suite consolidated PR
```bash
cd Golf_Modeling_Suite
gh pr list --author dieterolson --search "workflow standardization" --limit 10
```

### Step 2: Close all Jules PRs
```bash
# For each Jules PR, close with message
for pr_num in $(gh pr list --author google-labs-jules --json number --jq '.[].number'); do
    gh pr close $pr_num --comment "Superseded by consolidated workflow standardization PR #624"
done
```

### Step 3: Verify final state
- Each repo should have exactly 1 PR (our consolidated one)
- Total PRs should be ~9 (one per repo)

