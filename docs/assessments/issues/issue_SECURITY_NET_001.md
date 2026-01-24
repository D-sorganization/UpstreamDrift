---
title: "Security: Insecure Network Interface Binding"
labels: ["security", "jules:sentinel"]
---

## Description
The application binds to all network interfaces (`0.0.0.0`), which may expose the service to unintended networks if not properly firewalled.

**Findings:**
- `api/server.py:763` (B104)
- `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/meshcat_adapter.py:47` (B104)
- `start_api_server.py:94` (B104)

## Remediation
Bind to `127.0.0.1` (localhost) unless external access is explicitly required and secured. If running in a container, ensure the container port mapping is secure.
