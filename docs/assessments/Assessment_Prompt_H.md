# Assessment H: Error Handling & Debugging

## Assessment Overview

You are a **developer experience engineer** conducting an **adversarial** error handling review. Your job is to identify **cryptic errors, missing context, and debugging friction**.

---

## Key Metrics

| Metric                   | Target | Critical Threshold |
| ------------------------ | ------ | ------------------ |
| Actionable Error Rate    | >80%   | <50% = CRITICAL    |
| Time to Understand Error | <2 min | >10 min = MAJOR    |
| Recovery Path Documented | 100%   | Missing = MAJOR    |
| Verbose Mode Available   | Yes    | No = MINOR         |

---

## Review Categories

### A. Error Message Quality

**Bad vs Good Examples:**

```python
# BAD
ValueError: 42

# GOOD
ValueError: Invalid data format in 'input.csv' at row 42.
  Expected: numeric value in column 'price'
  Found: 'N/A'
  Fix: Replace 'N/A' with numeric value or use --skip-invalid flag
```

### B. Exception Hierarchy

- Consistent exception types
- Custom exceptions for domain errors
- Exception chaining (from e)
- Appropriate exception granularity

### C. Debugging Support

- Verbose/debug mode available
- Intermediate state inspection
- Structured logging
- Stack trace clarity

### D. Recovery Strategies

- Automatic retry for transient failures
- Graceful degradation
- Partial result handling
- State recovery after crash

### E. Error Documentation

- Error codes documented
- Troubleshooting guide exists
- FAQ for common errors
- Links from error messages to docs

---

## Error Scenario Testing

| Scenario           | Current Message | Actionable? | Fix |
| ------------------ | --------------- | ----------- | --- |
| Invalid input file | ...             | ✅/❌       | ... |
| Missing config     | ...             | ✅/❌       | ... |
| Network failure    | ...             | ✅/❌       | ... |
| Permission denied  | ...             | ✅/❌       | ... |
| Out of memory      | ...             | ✅/❌       | ... |

---

## Output Format

### 1. Error Quality Audit

| Error Type     | Current Quality | Fix Priority    |
| -------------- | --------------- | --------------- |
| File not found | GOOD/POOR       | High/Medium/Low |
| Invalid format | GOOD/POOR       | High/Medium/Low |
| Config error   | GOOD/POOR       | High/Medium/Low |

### 2. Remediation Roadmap

**48 hours:** Top 5 worst error messages
**2 weeks:** All user-facing errors actionable
**6 weeks:** Full troubleshooting guide, verbose mode

---

_Assessment H focuses on error handling. See Assessment D for user experience and Assessment G for testing._
