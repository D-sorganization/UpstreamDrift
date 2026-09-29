# Migration Data

Use this directory for explicit, idempotent seed-data loaders when an
environment requires data beyond schema migrations.

Schema migrations belong in `src/api/migrations/versions/`. Seed data must be
documented with its target environment, rollback behavior, and ownership before
it is run in production.
