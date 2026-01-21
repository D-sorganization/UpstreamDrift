# Security: XML Injection vulnerabilities in URDF tools

**Status:** Open
**Priority:** Medium
**Labels:** security, jules:sentinel

## Description
Bandit static analysis identified potential XML injection vulnerabilities due to the use of `xml.etree.ElementTree`. This library is not secure against maliciously constructed XML data.

## Findings
- **File:** `tools/urdf_generator/mujoco_viewer.py`
  - **Lines:** 69, 431
  - **Issue:** `B314: Using xml.etree.ElementTree.fromstring`
- **File:** `tests/unit/test_urdf_io.py`
  - **Lines:** 115
  - **Issue:** `B314: Using xml.etree.ElementTree.fromstring`

## Remediation Steps
Replace `xml.etree.ElementTree` with `defusedxml.ElementTree` or ensure `defusedxml.defuse_stdlib()` is called to prevent XML Entity Expansion (XEE) attacks.

```python
# Vulnerable
import xml.etree.ElementTree as ET
root = ET.fromstring(urdf_content)

# Secure
import defusedxml.ElementTree as ET
root = ET.fromstring(urdf_content)
```
