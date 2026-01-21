---
title: "Security: XML Injection Vulnerabilities in Core and Tools"
labels: ["security", "high-priority", "jules:sentinel"]
---

## Description
Bandit scan identified usage of `xml.etree.ElementTree` to parse potentially untrusted XML data. This is vulnerable to XML External Entity (XXE) attacks and other XML-related exploits.

**Findings:**
- `engines/physics_engines/mujoco/python/tests/test_urdf_io.py:186` (B314)
- `engines/physics_engines/mujoco/python/tests/test_urdf_io.py:204` (B314)
- `shared/python/myoconverter_integration.py:190` (B314)
- `tests/test_urdf_generator.py:131` (B314)
- `tests/test_urdf_generator.py:196` (B314)
- `tests/test_urdf_tools.py:51` (B314)
- `tests/test_urdf_tools.py:78` (B314)
- `tests/unit/test_physical_constants_xml.py` (Multiple instances)
- `tests/unit/test_urdf_io.py` (Multiple instances)
- `tools/urdf_generator/mujoco_viewer.py:69` (B314)
- `tools/urdf_generator/mujoco_viewer.py:431` (B314)

## Remediation
Replace `xml.etree.ElementTree` with `defusedxml.ElementTree` or ensure `defusedxml.defuse_stdlib()` is called before parsing.

```python
# Vulnerable
import xml.etree.ElementTree as ET
tree = ET.parse('data.xml')

# Secure
import defusedxml.ElementTree as ET
tree = ET.parse('data.xml')
```
