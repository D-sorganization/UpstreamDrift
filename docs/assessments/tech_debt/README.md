# Technical Debt Reports

This directory contains automated technical debt assessments generated weekly by the Jules Tech Debt Assessor.

## Overview

Technical debt represents the implied cost of future rework caused by choosing an expedient solution now instead of a better approach that would take longer. This automated assessment helps track and prioritize debt reduction efforts.

## Report Schedule

- **Frequency:** Weekly (Sunday 5 AM PST / 1 PM UTC)
- **Trigger:** Automated via Jules Control Tower or manual dispatch
- **Workflow:** `.github/workflows/Jules-Tech-Debt-Assessor.yml`

## What Gets Assessed

### 1. Code Complexity
- Cyclomatic complexity analysis using Radon
- Maintainability index scoring
- Functions with complexity > 10 are flagged

### 2. Incomplete Implementations
- `raise NotImplementedError` placeholders
- Empty function bodies (`pass` only)
- Stub/placeholder comments

### 3. Type Annotation Coverage
- MyPy type checking results
- Bare `# type: ignore` comments (should use specific error codes)
- Missing type annotations

### 4. Dead Code
- Vulture analysis for unused code
- Unreachable code paths
- Unused imports and variables

### 5. Security Debt
- Bandit security scan (medium+ severity)
- pip-audit dependency vulnerabilities
- Known CVE exposures

### 6. Linting Debt
- Ruff violations by category
- Files with most violations
- Style inconsistencies

### 7. Test Coverage Debt
- Modules without test files
- Coverage gap analysis (current 10%, target 60%)

## Understanding the Debt Score

The debt score is calculated using weighted factors:

| Factor | Weight | Rationale |
|--------|--------|-----------|
| Security Issues | x10 | Highest priority - potential vulnerabilities |
| NotImplementedError | x5 | Incomplete features affect stability |
| High Complexity | x3 | Makes code hard to maintain and test |
| Untested Modules | x3 | Risk of undetected regressions |
| Bare Type Ignores | x2 | May mask real type errors |
| Stub Comments | x2 | Indicates incomplete work |
| Type Errors | /10 | Divided to normalize large counts |
| Dead Code | /5 | Lower priority, cosmetic cleanup |
| Lint Violations | /20 | Style issues, lowest severity |

## Severity Levels

- **LOW** (0-20): Healthy codebase - routine maintenance only
- **MODERATE** (21-50): Manageable debt - schedule regular cleanup
- **HIGH** (51-100): Significant debt - prioritize reduction
- **CRITICAL** (>100): Urgent attention required

## Taking Action

### Immediate (Security/Critical)
1. Address security vulnerabilities immediately
2. Complete or remove NotImplementedError stubs in critical paths
3. Refactor functions with cyclomatic complexity > 20

### Short-term (1-2 Sprints)
4. Add specific error codes to bare `# type: ignore` comments
5. Add tests for critical untested modules
6. Remove dead code identified by Vulture

### Long-term (Quarterly)
7. Increase test coverage from 10% to 60%
8. Reduce average cyclomatic complexity below 5
9. Ensure all public APIs have docstrings

## Manual Trigger

To run the assessment manually:

```bash
gh workflow run Jules-Tech-Debt-Assessor.yml
```

Or via the GitHub Actions UI with optional inputs:
- `create_issues`: Create/update GitHub issues for tracking (default: true)
- `threshold_critical`: Threshold for critical debt items (default: 10)

## Related Documentation

- [Code Quality Review](../Code_Quality_Review_Latest.md)
- [Comprehensive Assessment](../Comprehensive_Assessment.md)
- [AGENTS.md](../../../AGENTS.md) - Coding standards and guidelines
