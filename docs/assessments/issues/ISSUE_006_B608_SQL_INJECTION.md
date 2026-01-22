# Security: SQL Injection Risk (B608)

**Labels:** security, high-priority, jules:sentinel

## Description
Bandit flagged a possible SQL injection vector through string-based query construction. This occurs when SQL queries are built using string formatting or concatenation instead of parameterized queries.

## Locations
- `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/recording_library.py:628`

## Remediation
Use parameterized queries (e.g., `?` or `%s` placeholders) provided by the database driver instead of constructing queries with string operations.
