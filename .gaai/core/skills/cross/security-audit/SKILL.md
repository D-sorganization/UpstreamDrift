---
name: security-audit
description: Detect security vulnerabilities and governance violations across delivered code, configurations, and deployed environments. Activate after implementation or periodically as a governance check.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-SECURITY-AUDIT-001
  updated_at: 2026-02-26
  status: experimental
inputs:
  - codebase
  - configuration_files
  - deployed_environment_metadata  (optional)
  - contexts/rules/**  (security rules)
outputs:
  - vulnerability_report
  - severity_scores
  - compliance_status
  - remediation_actions
---

# Security Audit

## Purpose / When to Activate

Activate:
