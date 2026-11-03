# Config and Secret Separation With Upgrade and Rollback Guarantees (#14274)

## Overview

As part of the professional deployment epic ([#14276](https://github.com/D-sorganization/UpstreamDrift/issues/14276)),
UpstreamDrift enforces strict separation between application configuration and sensitive
secrets, alongside fail-closed guarantees for zero-downtime upgrades and reversible rollbacks.

## Configuration Versus Secrets Separation

1. **Non-Sensitive Configuration**:

   - Host bindings, ports, CORS origins, log levels, algorithm options, and model weights
     paths are managed via typed settings (`src/shared/python/config/typed_settings.py`)
     and canonical accessors (`src/shared/python/config/environment.py`).
   - Default values are permitted for non-sensitive operational parameters to ensure
     backward compatibility when new options are introduced.

2. **Sensitive Secrets**:
   - Authentication secrets (`SECRET_KEY`), database passwords (`DATABASE_URL`), admin
     credentials (`ADMIN_PASSWORD`), and API tokens must never have default values in
     source code or settings models.
   - Secrets are injected strictly through runtime environment variables or secure secret
     mounts.
   - Repository root `.gitignore` and `.dockerignore` strictly exclude `.env` and `.env.*`
     to prevent accidental credential leakage in Git commits and container build contexts.
   - The `.env.example` file contains only documented, non-functional placeholder values
     (e.g., `change_me_in_production`). Startup validation via
     `src/shared/python/security/env_validator.py` hard-fails in production if placeholder
     values are detected.

## Upgrade and Rollback Guarantees

1. **Reversible Database Migrations**:

   - Every Alembic migration in `src/api/migrations/versions` must provide both an
     `upgrade()` function and a non-trivial, reversible `downgrade()` function.
   - CI executes the `alembic-postgres-round-trip` verification lane to test that migrations
     can be applied and rolled back cleanly without schema inconsistency or data corruption.

2. **Configuration Backward Compatibility**:
   - Rolling back application binaries to an earlier revision must not cause startup crashes
     due to unrecognized or omitted newly introduced configuration variables.
   - Deprecated configuration variables must be preserved with fallback accessors prior to
     eventual retirement.

## Automated Verification

Automated compliance with config/secret separation and rollback guarantees is verified via
`scripts/ci/check_config_secret_separation.py`:

```bash
# Run verification across the repository
python scripts/ci/check_config_secret_separation.py

# Emit machine-readable report
python scripts/ci/check_config_secret_separation.py --json
```
