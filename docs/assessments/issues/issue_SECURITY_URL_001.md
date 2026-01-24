---
title: "Security: Arbitrary URL Access"
labels: ["security", "jules:sentinel"]
---

## Description
Code performs `urllib.request.urlopen` on URLs that may be user-controlled or variable, potentially allowing access to `file://` or internal network resources (SSRF).

**Findings:**
- `shared/python/standard_models.py:151` (B310)
- `tools/urdf_generator/model_library.py:233` (B310)

## Remediation
- Validate the URL scheme is `http` or `https`.
- Allowlist permitted domains if possible.
- Disallow `file://` scheme.
