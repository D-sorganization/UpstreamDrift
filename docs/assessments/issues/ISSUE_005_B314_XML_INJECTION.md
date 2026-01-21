# Security: XML Injection Risk (B314)

**Labels:** security, jules:sentinel

## Description
Use of `xml.etree.ElementTree.fromstring` to parse untrusted XML data is known to be vulnerable to XML attacks (e.g., Billion Laughs).

## Locations
- `shared/python/myoconverter_integration.py:190`
- `tools/urdf_generator/mujoco_viewer.py:69`
- `tools/urdf_generator/mujoco_viewer.py:431`
- Multiple instances in test files (e.g., `tests/unit/test_urdf_io.py`)

## Remediation
Use `defusedxml.ElementTree` instead of the standard library version to parse XML safely.
