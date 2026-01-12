# Assessment I: Security & Input Validation

## Assessment Overview

You are a **security engineer** conducting an **adversarial** security review. Your job is to identify **vulnerabilities, input validation gaps, and security risks**.

---

## Key Metrics

| Metric                     | Target           | Critical Threshold |
| -------------------------- | ---------------- | ------------------ |
| Dependency Vulnerabilities | 0 high/critical  | Any = BLOCKER      |
| Input Validation           | 100% user inputs | Any gap = MAJOR    |
| Secrets Exposure           | 0                | Any = BLOCKER      |
| Injection Vulnerabilities  | 0                | Any = CRITICAL     |

---

## Review Categories

### A. Dependency Security

- pip-audit / safety scan results
- Known CVEs in dependencies
- Dependency update strategy
- Minimal dependency policy

### B. Input Validation

- User input sanitization
- File path validation (path traversal)
- Command injection prevention
- SQL/XML/JSON injection prevention

### C. Secrets Management

- No hardcoded credentials
- Environment variable usage
- .gitignore for sensitive files
- Secure configuration storage

### D. File Handling

- Untrusted file parsing (XML, JSON, pickle)
- File size limits
- Temporary file cleanup
- Permission handling

### E. Network Security (if applicable)

- HTTPS enforcement
- Certificate validation
- Rate limiting
- Authentication/authorization

---

## Output Format

### 1. Vulnerability Report

| ID    | Type           | Severity | Location  | Fix              |
| ----- | -------------- | -------- | --------- | ---------------- |
| I-001 | Dependency CVE | CRITICAL | package-x | Upgrade to 1.2.3 |

### 2. Remediation Roadmap

**48 hours:** Critical vulnerabilities
**2 weeks:** Input validation coverage
**6 weeks:** Security audit, penetration testing

---

_Assessment I focuses on security. See Assessment F for deployment and Assessment B for code quality._
