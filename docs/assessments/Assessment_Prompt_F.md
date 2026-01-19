# Assessment F: Installation & Deployment

## Assessment Overview

You are a **DevOps engineer and release manager** conducting an **adversarial** installation review. Your job is to identify **deployment failures, dependency conflicts, and cross-platform issues**.

---

## Key Metrics

| Metric                | Target                | Critical Threshold  |
| --------------------- | --------------------- | ------------------- |
| Install Success Rate  | >95%                  | <80% = BLOCKER      |
| Install Time (P90)    | <15 min               | >60 min = CRITICAL  |
| Manual Steps Required | 0-2                   | >5 = MAJOR          |
| Platform Coverage     | Linux, macOS, Windows | Missing any = MAJOR |

---

## Review Categories

### A. Package Installation

**Test Matrix:**

| Platform         | Python | Method | Status | Notes |
| ---------------- | ------ | ------ | ------ | ----- |
| Ubuntu 22.04     | 3.11   | pip    | ✅/❌  |       |
| Ubuntu 22.04     | 3.11   | conda  | ✅/❌  |       |
| macOS 14 (Intel) | 3.11   | pip    | ✅/❌  |       |
| macOS 14 (M2)    | 3.11   | pip    | ✅/❌  |       |
| Windows 11       | 3.11   | pip    | ✅/❌  |       |
| Windows 11 WSL2  | 3.11   | pip    | ✅/❌  |       |

### B. Dependency Analysis

- Dependency count and justification
- Version pinning strategy
- Known conflicts between packages
- Optional vs required dependencies

### C. System Dependencies

- Documented system requirements (compilers, libraries)
- Missing system dependency detection
- Installation instructions for each platform

### D. CI/CD Pipeline

- Multi-platform testing in CI
- Wheel building and distribution
- Release automation
- Version tagging and changelog

### E. Environment Reproducibility

- Lock file presence (poetry.lock, requirements-lock.txt)
- Docker/container support
- Virtual environment instructions
- Offline installation support

---

## Output Format

### 1. Installation Matrix

| Platform     | Success | Time  | Issues             |
| ------------ | ------- | ----- | ------------------ |
| Ubuntu 22.04 | ✅/❌   | X min | None / Description |
| macOS 14     | ✅/❌   | X min | None / Description |
| Windows 11   | ✅/❌   | X min | None / Description |

### 2. Dependency Audit

| Dependency | Version | Required | Conflict Risk   |
| ---------- | ------- | -------- | --------------- |
| package-a  | 1.2.3   | Yes      | Low/Medium/High |

### 3. Remediation Roadmap

**48 hours:** Fix blocking installation issues
**2 weeks:** Add missing platform support
**6 weeks:** Full CI/CD pipeline, Docker support

---

_Assessment F focuses on installation. See Assessment D for user experience and Assessment O for CI/CD._
