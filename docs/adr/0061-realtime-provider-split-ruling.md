# ADR-0061: Realtime Provider Split Ruling Between UpstreamDrift and Tools

- Status: Accepted
- Date: 2026-11-13
- Decision Makers: Architecture and Seam Governance (#12046)
- Related Issues/PRs: #12046, #8942, #9912, #9916, #14928, Tools #5090, ADR-0007, ADR-0012, ADR-0013

## Context

`src/shared/python/realtime/` in UpstreamDrift predates the two-file `realtime`
package added to `D-sorganization/Tools` (`vendor/ud-tools/src/shared/python/realtime/`)
for persistent `FileTransport` support (#5090, #8942). Because UpstreamDrift's
cluster fallback finder (`src/__init__.py`) resolves a `src.shared.python.<cluster>`
package from `vendor/ud-tools` only when `src/shared/python/<cluster>` is absent,
retaining `src/shared/python/realtime/` registers in
`tests/unit/repo_hygiene/test_no_shadow_of_tools_shared.py`.

Removing UpstreamDrift's package without replacing its public API would break
`src/launchers/embedded_host.py`, pose-studio WebSocket integrations,
`tests/unit/repo_hygiene/test_cross_engine_api_versions.py` (`__version__`,
`SCHEMA_VERSION`, `__all__`), and `tests/unit/repo_hygiene/test_import_boundaries.py`.

## Evidence

| Package Root                                  | Modules                                                                                    | Exported Public Surface & Semantics                                                                            |
| --------------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `src/shared/python/realtime/` (UpstreamDrift) | `__init__.py`, `api.py`, `channels.py`, `protocol.py`, `transport_file.py`, `ws_pubsub.py` | Cross-engine `publish`, `subscribe`, `register_channel`, `CHANNEL_REGISTRY`, `Subscription`, semver metadata   |
| `vendor/ud-tools/src/shared/python/realtime/` | `__init__.py`, `transport_file.py`                                                         | Standalone `FileTransport` and `default_channel_path` with divergent handle lifetime, polling, and read cursor |

Issue #8942 closed after Tools received persistent file transport, leaving
`scripts/config/shadow_modules.yaml` with an expired temporary `sunset_date`
that required repeated date-extension churn (#12046, #14928).

## Decision

Record an explicit **split** ownership ruling for `realtime` under
`docs/shared_tools/seam_rulings.v1.json` and `scripts/config/shadow_modules.yaml`:

1. UpstreamDrift retains canonical ownership of the `src.shared.python.realtime`
   publish/subscribe facade, channel registry, protocol schema, and WebSocket
   integration.
2. Tools retains ownership of its standalone `FileTransport` utility under
   `vendor/ud-tools/src/shared/python/realtime/` without UpstreamDrift claiming
   runtime API equivalence.
3. Replace the temporary `sunset_date` on `realtime` in
   `scripts/config/shadow_modules.yaml` with
   `adr_ruling: "docs/adr/0061-realtime-provider-split-ruling.md"`, enforced by
   `tests/unit/repo_hygiene/test_no_shadow_of_tools_shared.py` without extending
   sunset deadlines.

## Alternatives Considered

1. **Port UpstreamDrift's publish/subscribe facade, registry, and WebSocket
   layers into Tools and delete UD's copy:** Rejected because those cross-engine
   channel contracts and `src.api.routes.realtime` bindings are UpstreamDrift-specific
   and `FileTransport` semantics differ between the two trees.
2. **Repeatedly extend `sunset_date` in `shadow_modules.yaml`:** Rejected under
   #12046; sunset extensions mask permanent ownership boundaries with recurring
   queue-blocking expiration dates.

## Consequences

- Positive: `test_no_shadow_of_tools_shared.py` passes cleanly without date
  extensions while preserving `test_cross_engine_api_versions.py` and existing
  `src.shared.python.realtime` callers.
- Negative: `src/shared/python/realtime/` does not auto-inherit future edits to
  `vendor/ud-tools/src/shared/python/realtime/transport_file.py`.
- Follow-ups: Future unification may retire this ruling only if Tools adopts
  a compatible pub/sub facade and a migration ADR verifies `FileTransport`
  handle-lifetime and polling equivalence across both consumers.

## Validation

- `tests/unit/repo_hygiene/test_no_shadow_of_tools_shared.py` verifies that
  `realtime` in `scripts/config/shadow_modules.yaml` carries `tracking_issue: 12046`,
  no `sunset_date`, and a valid `adr_ruling` pointing to this existing file.
- `python scripts/shared_tools/check_seam_drift.py` verifies `seam_rulings.v1.json`
  keeps `realtime` recorded as `split` with valid metadata.
- `tests/unit/repo_hygiene/test_cross_engine_api_versions.py` verifies
  `src.shared.python.realtime` continues exposing valid semver `__version__`,
  `SCHEMA_VERSION`, and `__all__`.
