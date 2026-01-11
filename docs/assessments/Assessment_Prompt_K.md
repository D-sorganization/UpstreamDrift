# Assessment K: CI/CD Pipeline Health

**Assessment Type**: DevOps Infrastructure Audit
**Rotation Day**: Day 11 (Monthly)
**Focus**: Build times, reliability, coverage gates, automation maturity

---

## Objective

Conduct a CI/CD pipeline audit identifying:

1. Build time optimization opportunities
2. Pipeline reliability issues
3. Coverage and quality gates
4. Automation gaps
5. Developer experience friction

---

## Mandatory Deliverables

### 1. Pipeline Health Summary

- Average build time: X minutes
- Pipeline reliability: X%
- Quality gates: X enabled
- Automation coverage: X%

### 2. CI/CD Scorecard

| Category       | Score (0-10) | Weight | Evidence Required  |
| -------------- | ------------ | ------ | ------------------ |
| Build Speed    |              | 2x     | Average build time |
| Reliability    |              | 2x     | Success rate       |
| Quality Gates  |              | 2x     | Gates enabled      |
| Automation     |              | 1.5x   | Manual steps       |
| Caching        |              | 1.5x   | Cache hit rate     |
| Feedback Speed |              | 2x     | Time to feedback   |

### 3. Pipeline Findings

| ID  | Stage | Issue | Impact | Fix | Effort | Priority |
| --- | ----- | ----- | ------ | --- | ------ | -------- |
|     |       |       |        |     |        |          |

---

## Categories to Evaluate

### 1. Build Performance

- [ ] Build time < 10 minutes
- [ ] Parallelization used
- [ ] Caching implemented
- [ ] Incremental builds where possible

### 2. Reliability

- [ ] Pipeline success rate > 95%
- [ ] No flaky tests
- [ ] Deterministic builds
- [ ] Retry logic for transient failures

### 3. Quality Gates

- [ ] Linting enforced (Ruff)
- [ ] Type checking enforced (Mypy)
- [ ] Formatting enforced (Black)
- [ ] Test coverage threshold
- [ ] Security scanning (pip-audit)

### 4. Automation

- [ ] All checks automated
- [ ] No manual approval for standard PRs
- [ ] Auto-merge for dependencies
- [ ] Release automation

### 5. Developer Experience

- [ ] Clear failure messages
- [ ] Fast feedback (< 5 min for lint)
- [ ] Local reproduction possible
- [ ] Skip mechanisms for drafts

### 6. Security

- [ ] Secrets management
- [ ] Dependency scanning
- [ ] Container scanning (if applicable)
- [ ] SAST integration

---

## Workflow Analysis

### GitHub Actions Checklist

- [ ] Workflow triggers appropriate
- [ ] Job matrix efficient
- [ ] Actions pinned to SHA
- [ ] Permissions minimized
- [ ] Artifacts managed

### Metrics to Track

| Metric           | Target   | Current |
| ---------------- | -------- | ------- |
| Build time       | < 10 min |         |
| Success rate     | > 95%    |         |
| Time to feedback | < 5 min  |         |
| Cache hit rate   | > 80%    |         |

---

## Analysis Commands

```bash
# GitHub Actions workflow analysis
gh run list --limit 20 --json conclusion,createdAt,updatedAt

# Check workflow completion times
gh run list --json name,conclusion,createdAt,updatedAt \
  --jq '.[] | "\(.name): \(.conclusion)"'

# Local CI reproduction
act -l  # List workflows
act push  # Run push workflows locally
```

---

## Best Practices

### Pipeline Design

- [ ] Fail fast (lint before test)
- [ ] Parallel where possible
- [ ] Cache aggressively
- [ ] Use matrix builds wisely

### Gates

- [ ] Required checks for main
- [ ] Branch protection enabled
- [ ] Status checks required
- [ ] Review requirements

---

_Assessment K focuses on CI/CD. See Assessment A-J for other dimensions._
