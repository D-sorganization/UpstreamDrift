# Assessment E: Security Deep Dive

**Assessment Type**: Security Audit
**Rotation Day**: Day 5 (Friday)
**Focus**: Vulnerability detection, secure coding, attack surface analysis

---

## Objective

Conduct an adversarial security audit identifying:

1. Injection vulnerabilities (SQL, command, path)
2. Authentication and authorization flaws
3. Data exposure risks
4. Dependency vulnerabilities (CVEs)
5. Cryptographic weaknesses

---

## Mandatory Deliverables

### 1. Security Posture Statement

- Overall security rating (Critical/Major/Minor/Good)
- Number of vulnerabilities by severity
- Whether production-safe for security

### 2. Security Scorecard

| Category            | Score (0-10) | Weight | Evidence Required        |
| ------------------- | ------------ | ------ | ------------------------ |
| Input Validation    |              | 2x     | Injection points audited |
| Authentication      |              | 2x     | Auth mechanisms reviewed |
| Data Protection     |              | 2x     | Encryption, PII handling |
| Dependency Security |              | 2x     | CVE scan results         |
| Secure Coding       |              | 1.5x   | Anti-patterns identified |
| Attack Surface      |              | 1.5x   | Entry points enumerated  |

### 3. Vulnerability Findings Table

| ID    | CVSS | Category | Location | Vulnerability | Exploit Scenario | Fix | Priority |
| ----- | ---- | -------- | -------- | ------------- | ---------------- | --- | -------- |
| E-001 |      |          |          |               |                  |     |          |

### 4. Attack Surface Map

List all entry points where untrusted data enters the system.

---

## Categories to Evaluate

### 1. Injection Prevention

- [ ] No `eval()` or `exec()` with user input
- [ ] No shell command injection (`subprocess` with shell=True)
- [ ] No SQL injection (parameterized queries only)
- [ ] No path traversal (validated file paths)
- [ ] No template injection

### 2. Authentication & Authorization

- [ ] Credentials not hardcoded
- [ ] API keys in environment variables
- [ ] No default passwords
- [ ] Session management secure (if applicable)
- [ ] Role-based access implemented (if applicable)

### 3. Data Protection

- [ ] Sensitive data encrypted at rest
- [ ] TLS for network communications
- [ ] No PII in logs
- [ ] Secure deletion of temporary files
- [ ] No secrets in version control

### 4. Dependency Security

- [ ] pip-audit or safety scan run
- [ ] No known CVEs in dependencies
- [ ] Dependencies pinned with hashes
- [ ] Regular dependency updates

### 5. Secure Coding Practices

- [ ] Input validation on all entry points
- [ ] Output encoding for display
- [ ] Error messages don't leak information
- [ ] Safe deserialization (no pickle from untrusted)
- [ ] Safe file operations (atomic writes, proper permissions)

### 6. Cryptography

- [ ] Modern algorithms (AES-256, SHA-256+)
- [ ] No MD5 or SHA1 for security purposes
- [ ] Proper key management
- [ ] Secure random number generation

---

## Security Anti-Patterns to Flag

### Critical (CVSS 9.0+)

```python
# NEVER DO THIS
eval(user_input)
exec(user_input)
subprocess.run(user_input, shell=True)
pickle.loads(untrusted_data)
os.system(f"command {user_input}")
```

### High (CVSS 7.0-8.9)

```python
# DANGEROUS
open(user_provided_path)  # Path traversal
cursor.execute(f"SELECT * WHERE id={user_id}")  # SQL injection
password = "hardcoded123"  # Hardcoded secrets
logging.info(f"User password: {password}")  # PII in logs
```

### Medium (CVSS 4.0-6.9)

```python
# RISKY
import pickle  # Insecure serialization
hashlib.md5(data)  # Weak hash for security
random.random()  # Not cryptographically secure
```

---

## Scan Commands

```bash
# Dependency vulnerability scan
pip install pip-audit
pip-audit --strict

# Alternative with safety
pip install safety
safety check

# Secret scanning
pip install detect-secrets
detect-secrets scan

# SAST scanning with bandit
pip install bandit
bandit -r . -f json -o security_report.json

# Check for known patterns
grep -rn "eval\|exec\|subprocess.*shell=True" --include="*.py"
grep -rn "password\|secret\|api_key\|token" --include="*.py"
```

---

## Output Format

### Security Grade

- **A (9-10)**: Secure by design, no known vulnerabilities
- **B (7-8)**: Minor issues, safe with precautions
- **C (5-6)**: Moderate risks, address before production
- **D (3-4)**: Significant vulnerabilities, not production-safe
- **F (0-2)**: Critical vulnerabilities, immediate remediation

---

## Repository-Specific Focus

### For Tools Repository

- File path handling in tools
- User input in calculators
- Script execution safety

### For Scientific Repositories

- Data file parsing security
- External data ingestion
- Result export safety

### For Web Repositories

- XSS prevention
- CSRF protection
- Cookie security

---

_Assessment E focuses on security. See Assessment A-D for other quality dimensions._
