# Data Handling

## Scope

The current API database schema stores authentication and account data in three
tables: `users`, `api_keys`, and `sessions`.

## users

- Classification: PII and account metadata. `email`, `full_name`, and
  `organization` identify or describe a person or organization.
- Secrets: `hashed_password` is an authentication secret derivative and must
  never be logged or exported in plaintext.
- Retention: retain active accounts while the account exists. After account
  deletion, remove direct identifiers within 30 days unless legal or billing
  obligations require a longer hold.
- Deletion path: delete the user row through an account-deletion workflow.
  Database foreign keys cascade related API keys and sessions.
- Encryption at rest: production PostgreSQL storage must use provider-managed
  encryption at rest. Backups and point-in-time restore archives must inherit
  the same or stronger encryption controls.

## api_keys

- Classification: authentication secret metadata. `key_hash` and `key_prefix`
  are not plaintext API keys, but they are still security-sensitive.
- Secrets: plaintext API keys must only be shown at creation time and must not
  be stored. Hash verification must remain one-way.
- Retention: retain active API keys until revoked, expired, or the owning user
  is deleted. Retain revoked-key audit metadata only if an audit requirement is
  defined for the deployment.
- Deletion path: revoke/delete keys directly for key-rotation requests; user
  deletion cascades keys through the `api_keys.user_id` foreign key.
- Encryption at rest: same production database and backup encryption baseline
  as `users`.

## sessions

- Classification: authentication session metadata. `session_token`,
  `refresh_token`, `ip_address`, and `user_agent` are security-sensitive and
  may be personal data.
- Secrets: session and refresh tokens must not be logged, exported, or exposed
  through diagnostics.
- Retention: delete expired sessions during routine cleanup. Delete all sessions
  immediately on user deletion, credential compromise, or explicit logout-all
  requests.
- Deletion path: direct session revocation for logout/security events; user
  deletion cascades sessions through the `sessions.user_id` foreign key.
- Encryption at rest: same production database and backup encryption baseline
  as `users`.

## Operational Requirements

- Production logs must not include passwords, API keys, session tokens, refresh
  tokens, or raw database rows from these tables.
- Exports containing these tables require security review and a defined
  retention period.
- Backup restore tests must treat restored datasets as production-sensitive
  unless data is anonymized before use.
