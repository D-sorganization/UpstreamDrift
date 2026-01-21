---
title: Insecure Bind to All Interfaces
labels: security, jules:sentinel
severity: MEDIUM
status: OPEN
---

**Description:**
`start_api_server.py` binds to `0.0.0.0` by default. This configures the service to listen on all interfaces, which may expose the service to unintended networks.

**File:** `./start_api_server.py:94`
**Finding ID:** B104

**Remediation:**
Configure the service to listen only on localhost (`127.0.0.1`) unless explicitly required otherwise.
