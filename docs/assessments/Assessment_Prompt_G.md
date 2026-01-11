# Assessment G: Dependency Health & Supply Chain

**Assessment Type**: Dependency Security Audit
**Rotation Day**: Day 7 (Weekly)
**Focus**: CVE detection, outdated packages, license compliance, supply chain risk

---

## Objective

Conduct a comprehensive dependency audit identifying:

1. Known vulnerabilities (CVEs) in dependencies
2. Outdated packages requiring updates
3. License compatibility issues
4. Transitive dependency risks
5. Supply chain attack vectors

---

## Mandatory Deliverables

### 1. Dependency Health Summary

- Total dependencies: X (direct) + Y (transitive)
- CVEs found: X (Critical/High/Medium/Low)
- Outdated packages: X of Y
- License conflicts: X

### 2. Dependency Scorecard

| Category           | Score (0-10) | Weight | Evidence Required      |
| ------------------ | ------------ | ------ | ---------------------- |
| CVE Status         |              | 3x     | pip-audit results      |
| Freshness          |              | 2x     | Outdated package count |
| License Compliance |              | 2x     | License audit          |
| Pin Strategy       |              | 1.5x   | Requirements analysis  |
| Supply Chain       |              | 2x     | Source verification    |
| Transitive Risk    |              | 1.5x   | Dependency tree depth  |

### 3. Vulnerability Findings

| Package | Version | CVE | CVSS | Status | Fix Version | Priority |
| ------- | ------- | --- | ---- | ------ | ----------- | -------- |
|         |         |     |      |        |             |          |

---

## Categories to Evaluate

### 1. Vulnerability Scanning

- [ ] pip-audit run with no critical findings
- [ ] safety check run
- [ ] No known CVEs in production dependencies
- [ ] CVE remediation plan for any findings

### 2. Package Freshness

- [ ] Dependencies updated within 6 months
- [ ] Major version updates evaluated
- [ ] Security patches applied promptly
- [ ] Changelog reviewed for updates

### 3. License Compliance

- [ ] All licenses compatible with project license
- [ ] No copyleft licenses in proprietary code
- [ ] License attribution documented
- [ ] SBOM (Software Bill of Materials) available

### 4. Pinning Strategy

- [ ] Production dependencies pinned
- [ ] Hashes used for security
- [ ] Range constraints appropriate
- [ ] Lock file maintained

### 5. Supply Chain Security

- [ ] Packages from trusted sources (PyPI)
- [ ] No typosquatting risks
- [ ] Maintainer activity verified
- [ ] No abandoned packages

### 6. Transitive Dependencies

- [ ] Dependency tree reviewed
- [ ] Deep transitive chains identified
- [ ] Vulnerable transitive deps detected
- [ ] Minimal dependency principle applied

---

## Scan Commands

```bash
# CVE scanning with pip-audit
pip install pip-audit
pip-audit --strict --desc on

# Alternative with safety
pip install safety
safety check --full-report

# Check outdated packages
pip list --outdated

# Generate dependency tree
pip install pipdeptree
pipdeptree --warn silence

# License check
pip install pip-licenses
pip-licenses --format=markdown

# Generate SBOM
pip install cyclonedx-bom
cyclonedx-py requirements requirements.txt -o sbom.json
```

---

## Output Format

### Dependency Health Grade

- **A (9-10)**: No CVEs, all current, compliant licenses
- **B (7-8)**: No critical CVEs, mostly current
- **C (5-6)**: Some CVEs, several outdated
- **D (3-4)**: Critical CVEs present
- **F (0-2)**: Multiple critical CVEs, abandoned packages

---

_Assessment G focuses on dependency health. See Assessment A-F for other dimensions._
