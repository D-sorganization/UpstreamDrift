# Database Tier

## Schema Management

Production schema changes are owned by Alembic. Deployments must run:

```bash
python3 scripts/db_migrate.py upgrade head
```

before starting the new API version. On startup with
`ENVIRONMENT=production`, `src/api/database.py` verifies that the
database `alembic_version` revision exactly matches the migration head in the
checked-out codebase. If the version table is missing, unreadable, empty, or
behind/ahead of the codebase, the server refuses to start.

`ENVIRONMENT` is the canonical variable (the same one used by `SECURITY.md`'s
production checklist and `src/api/auth/security.py`). The legacy
`UPSTREAM_DRIFT_ENV` spelling is still honoured as a fallback, and the gate
fails closed: if _either_ variable names `production`, the production path is
taken. Previously this gate read only `UPSTREAM_DRIFT_ENV`, so a deployment
that followed `SECURITY.md` silently ran `create_all()` and seeded a default
admin account (issue #7994).

Development and test environments may still use SQLAlchemy `create_all()` for
local convenience. That path is not the production schema-management path.

## Drift Detection

Run the drift check whenever SQLAlchemy models or migration files change:

```bash
python3 scripts/check_alembic_drift.py
```

The helper delegates to Alembic autogenerate checking and exits non-zero when
model metadata does not match committed migrations.

## Backup, RPO, and RTO

Production databases must use managed PostgreSQL backups with point-in-time
restore enabled before production traffic is accepted.

- Backup strategy: automated daily full backups plus point-in-time restore.
- RPO target: 15 minutes, bounded by WAL/archive frequency.
- RTO target: 60 minutes for restore, migration verification, and API restart.
- Restore validation: rehearse restore into a non-production database before
  each major release and after backup-provider changes.

SQLite databases are for local development and tests only. They are not a
supported production backup target.

## Connection Pooling

For non-SQLite URLs, the application creates a SQLAlchemy engine with
`pool_pre_ping=True` and `pool_recycle=300`. The current code does not override
SQLAlchemy's `QueuePool` defaults, so the effective defaults are:

- Pool size: 5 checked-in connections per process.
- Max overflow: 10 transient connections per process.
- Recycle: 300 seconds.

Capacity planning must multiply these values by the number of API worker
processes and replicas. If the database cannot support that total, set explicit
pool limits before increasing replica count.

## Failure Modes

- DB unreachable at startup: production startup fails while verifying
  `alembic_version`; readiness must remain unavailable until the dependency is
  restored.
- DB version drift: production startup fails and reports the database revision
  set and codebase head set.
- Missing migration for a model change: `scripts/check_alembic_drift.py` exits
  non-zero and the PR must add an Alembic revision.
- Connection pool exhaustion: requests can block or fail while waiting for a
  connection. Reduce worker/replica count or configure smaller per-process pool
  limits before retrying deployment.

## Seed Data

Schema migrations live in `src/api/migrations/versions/`. Environment-specific
or demo seed data must not be hidden inside schema migrations. If seed data is
needed, place it under `src/api/migrations/data/` with an idempotent loader and
document which environments run it.
