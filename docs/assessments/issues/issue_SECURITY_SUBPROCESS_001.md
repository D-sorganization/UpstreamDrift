---
title: "Security: Insecure Subprocess Execution in Tools"
labels: ["security", "jules:sentinel"]
---

## Description
Tools use `subprocess` with potentially insecure configurations, such as `shell=True` or passing untrusted input without validation.

**Findings:**
- `tests/integration/test_phase1_security_integration.py:124` (B604)
- `tests/unit/test_secure_subprocess.py:117` (B604)
- `tests/verify_changes.py:91` (B603)
- `tools/matlab_utilities/scripts/matlab_quality_check.py:141` (B603)
- `tools/urdf_generator/mujoco_viewer.py:566` (B603)

## Remediation
- Avoid `shell=True`.
- Use a list of arguments instead of a command string.
- Validate and sanitize all inputs passed to subprocesses.
