# Security: Possible binding to all interfaces (B104)

**Labels:** security, jules:sentinel

## Description
Bandit identified potential security risks where services bind to all network interfaces (0.0.0.0). This can expose the service to unintended networks.

## Locations
- `api/server.py:763`
- `start_api_server.py:94`
- `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/meshcat_adapter.py:47`

## Remediation
Bind to localhost (127.0.0.1) unless external access is explicitly required and secured.
