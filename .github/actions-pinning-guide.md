# GitHub Actions Pinning Guide (Issue #3066)

This document tracks the pinning of GitHub Actions to specific commit SHAs for supply chain security.

## Pinning Strategy

All GitHub Actions should be pinned to specific commit SHAs instead of version tags (v1, v2, v3, etc.).

**Format:**

```yaml
- uses: actions/checkout@eef61447b9ff4aafe5dcd4e0bbf5d482be7e7871 # v4.2.1
```

## Critical Actions to Pin

### 1. actions/checkout

- Current: v6
- Pin to: eef61447b9ff4aafe5dcd4e0bbf5d482be7e7871 (v4.2.1)

### 2. actions/setup-python

- Current: v6
- Pin to: 0b93645274ce2973f2fb2b04e521ba9a448e4229 (v5.0.1)

### 3. actions/upload-artifact

- Current: v7
- Pin to: 6f51ac03b9356f520e9adb1b312f6c7d01159334 (v4.5.0)

### 4. actions/download-artifact

- Current: v5
- Pin to: 9c5a7168910dcb17d191cd8f8138177c8eccaa3ee (v4.1.7)

### 5. actions/setup-node

- Current: v4
- Pin to: 60edb3dd545a775178fba7601554c5ec61b3b331 (v4.0.3)

### 6. actions/cache

- Current: v5
- Pin to: 0865c47f36b7a7154ec3007eea82a0597437d674 (v4.0.0)

### 7. codecov/codecov-action

- Current: v5
- Note: Third-party action, requires separate research

## Workflow Updates Required

1. **ci-standard.yml** - Primary test workflow

   - checkout: 2 references (lines 37, 231, 326, 328, 374, 442, 460, 478, 523)
   - setup-python: 3 references (lines 40, 247, 349, 461)
   - upload-artifact: 1 reference (line 217)
   - setup-node: 1 reference (line 401)
   - cache: 1 reference (line 538)
   - codecov/codecov-action: 1 reference (line 312)

2. **release.yml** - Release workflow

   - checkout: 2 references
   - setup-python: 1 reference
   - download-artifact: 2 references
   - pypa/gh-action-pypi-publish: requires research

3. **All other workflows** - 54 additional files requiring systematic updates

## Next Steps

1. Verify SHAs against GitHub Actions repository releases
2. Update ci-standard.yml and release.yml as priority
3. Create automated tooling to mass-update remaining workflows
4. Add pre-commit hook to enforce SHA pinning on new action references
