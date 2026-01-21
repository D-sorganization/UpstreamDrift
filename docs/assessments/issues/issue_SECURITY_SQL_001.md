---
title: "Security: Potential SQL Injection in Recording Library"
labels: ["security", "high-priority", "jules:sentinel"]
---

## Description
Bandit scan identified a possible SQL injection vector through string-based query construction.

**Findings:**
- `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/recording_library.py:628` (B608)

## Remediation
Ensure all SQL queries use parameterized queries (bind variables) instead of string formatting or concatenation.

```python
# Vulnerable
cursor.execute("SELECT * FROM users WHERE name = '%s'" % name)

# Secure
cursor.execute("SELECT * FROM users WHERE name = ?", (name,))
```
