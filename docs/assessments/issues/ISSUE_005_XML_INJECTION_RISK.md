---
title: XML Injection Risk (ElementTree)
labels: security, jules:sentinel
severity: MEDIUM
status: OPEN
---

**Description:**
`tools/urdf_generator/mujoco_viewer.py` uses `xml.etree.ElementTree.fromstring` to parse untrusted XML data. This is vulnerable to XML attacks (e.g., entity expansion).

**File:** `./tools/urdf_generator/mujoco_viewer.py:69` (and others)
**Finding ID:** B314

**Remediation:**
Replace `xml.etree.ElementTree` with `defusedxml` equivalent functions (e.g., `defusedxml.ElementTree.fromstring`).
