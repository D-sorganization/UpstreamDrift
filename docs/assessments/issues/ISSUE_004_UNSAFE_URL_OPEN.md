# Security: Unsafe URL open in URDF Model Library

**Status:** Open
**Priority:** Medium
**Labels:** security, jules:sentinel

## Description
Bandit static analysis identified use of `urllib.request.urlopen` which may permit usage of unexpected schemes (e.g., `file:///`) if not properly validated.

## Findings
- **File:** `tools/urdf_generator/model_library.py`
  - **Lines:** 233
  - **Issue:** `B310: Audit url open for permitted schemes`
  - **Code:** `with urllib.request.urlopen(model_info["urdf_url"]) as response:`

## Remediation Steps
Validate the URL scheme before opening, or switch to the `requests` library which has safer defaults.

```python
# Remediation Example
from urllib.parse import urlparse

parsed = urlparse(url)
if parsed.scheme not in ('http', 'https'):
    raise ValueError("Invalid scheme")
```
