---
title: Audit URL Open for Permitted Schemes
labels: security, jules:sentinel
severity: MEDIUM
status: OPEN
---

**Description:**
`tools/urdf_generator/model_library.py` uses `urllib.request.urlopen` which allows `file:/` or custom schemes. This can be unexpected and potentially dangerous.

**File:** `./tools/urdf_generator/model_library.py:233`
**Finding ID:** B310

**Remediation:**
Audit the URL schemes allowed or use a safer library/method that restricts schemes to http/https.
