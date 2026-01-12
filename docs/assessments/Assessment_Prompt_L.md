# Assessment L: Long-Term Maintainability

## Assessment Overview

You are a **technical lead** evaluating the codebase for **long-term sustainability, technical debt, and succession planning**.

---

## Key Metrics

| Metric            | Target     | Critical Threshold |
| ----------------- | ---------- | ------------------ |
| Deprecated Deps   | 0          | >3 = MAJOR         |
| Unmaintained Code | <10%       | >30% = CRITICAL    |
| Bus Factor        | >2         | 1 = CRITICAL       |
| Upgrade Path      | Documented | Missing = MAJOR    |

---

## Review Categories

### A. Dependency Health

- Last update dates for dependencies
- Dependencies nearing EOL
- Python version compatibility
- Upgrade path planning (Python 3.12+, NumPy 2.x)

### B. Code Aging

- Files not modified in >1 year
- Modules without tests
- Orphaned code (unused but present)
- Technical debt inventory

### C. Knowledge Distribution

- Modules with single author
- Documentation for complex areas
- Onboarding for new maintainers
- Code review practices

### D. Sustainability

- Automated update tools (Dependabot, Renovate)
- Maintenance schedule
- Deprecation tracking
- Migration guides

---

## Output Format

### 1. Maintainability Assessment

| Area           | Status   | Risk            | Action |
| -------------- | -------- | --------------- | ------ |
| Dependency age | ✅/⚠️/❌ | Low/Medium/High | ...    |
| Code coverage  | ✅/⚠️/❌ | Low/Medium/High | ...    |
| Bus factor     | ✅/⚠️/❌ | Low/Medium/High | ...    |

### 2. Remediation Roadmap

**48 hours:** Identify critical single-author modules
**2 weeks:** Update deprecated dependencies
**6 weeks:** Full technical debt reduction plan

---

_Assessment L focuses on maintainability. See Assessment B for code quality and Assessment O for CI/CD._
