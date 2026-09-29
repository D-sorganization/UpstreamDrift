# ADR: API Versioning Policy

- Status: Accepted
- Date: 2026-04-30
- Decision Makers: Platform/API team
- Related Issues/PRs: #3508 (deferred from #3520)

## Context

The HTTP API is consumed by internal services, the desktop UI, and external
research tooling. As physics engines, schemas, and analysis surfaces evolve,
clients must be able to upgrade independently of the server. Without an
explicit versioning contract, additive and breaking changes are easy to
conflate, sunset windows are ad hoc, and clients cannot reliably detect
deprecated endpoints.

This ADR establishes the policy for evolving the public HTTP surface. It is
intentionally narrow in scope: it codifies _how_ versions are expressed and
deprecated, not _what_ belongs in any specific version.

## Decision

Adopt **URI-path versioning** for the HTTP API:

- All versioned endpoints are mounted under a major-version segment:
  `/v1/...`, `/v2/...`. Minor and patch revisions are non-breaking and do
  _not_ change the URI.
- The current major version is additionally exposed via a
  **default-unprefixed alias** (e.g. `/simulations` resolves to
  `/v1/simulations` while `v1` is current). This preserves backwards
  compatibility for clients that have not yet adopted the prefix.
- A new helper, `src/api/versioning.py::make_versioned_router`, is the single
  supported way to construct a versioned `APIRouter`. It enforces the prefix
  format and wires deprecation/sunset response headers when applicable.

### Lifecycle

- **Announcement**: a new major version is announced in release notes and via
  `Deprecation: true` headers on the previous major's responses.
- **Deprecation window**: **12 months** between announcement and removal of a
  deprecated major version. Until removal, deprecated endpoints continue to
  function unchanged.
- **Sunset signaling**: deprecated endpoints set
  `Sunset: <RFC 1123 date>` (per RFC 8594) on every response, indicating the
  earliest date the endpoint may be removed.
- **Deprecation signaling**: deprecated endpoints set `Deprecation: true`
  (per RFC 9745) on every response. Where available, a `Link: <...>;
rel="successor-version"` header points clients at the replacement.
- **Removal**: only after the sunset date has passed and at least one
  release has shipped containing the new major. The default-unprefixed
  alias is reassigned to the new current major at removal time.

### Compatibility rules

- **Non-breaking (allowed within a major version):**
  - Adding new endpoints.
  - Adding new optional request fields with safe defaults.
  - Adding new response fields (clients must ignore unknown fields).
  - Loosening validation (accepting more inputs).
  - Adding new enum members where the field is documented as open.
- **Breaking (requires a new major version):**
  - Removing or renaming endpoints, fields, or query parameters.
  - Changing field types or semantics.
  - Tightening validation (rejecting previously accepted inputs).
  - Changing default values that alter observable behavior.
  - Changing authentication or authorization requirements.

### Client guidance

- **Pin to a major version.** Clients should explicitly call `/v1/...` rather
  than relying on the default-unprefixed alias, so that the eventual
  re-aliasing of the unprefixed routes does not silently move the client
  to a new major.
- **Inspect response headers** on every call (or at least periodically):
  - `Deprecation: true` -> begin migration planning.
  - `Sunset: <date>` -> migrate before that date.
  - `Link: ...; rel="successor-version"` -> target for migration.
- **Tolerate unknown response fields** to allow non-breaking additive changes
  to ship without client updates.
- **Do not parse minor/patch versions out of URIs**; use the
  `/healthz` / `/version` endpoint for granular server build info.

### Implementation outline

- `src/api/versioning.py` exports `make_versioned_router(version, *,
deprecated=False, sunset=None)`.
- Internally it returns a `fastapi.APIRouter(prefix=f"/{version}")` with a
  `dependencies=[Depends(_deprecation_headers)]` hook when
  `deprecated=True`. The dependency mutates the outgoing `Response` to set
  `Deprecation: true` and (if provided) `Sunset: <RFC 1123 date>`.
- `src/api/server.py` mounts the canonical routers under both the versioned
  prefix and (for the current major) the default-unprefixed paths. New major
  versions are added by constructing an additional router and mounting it
  alongside the existing one.
- The helper validates the version string against `^v\d+$`. Anything else
  raises `ValueError` so misuse fails fast in tests rather than at runtime
  in production.

## Alternatives Considered

1. **Header-based versioning** (e.g. `Accept: application/vnd.upstream.v2+json`).
   Rejected: less discoverable in logs and curl, harder to cache via CDNs and
   reverse proxies, and harder for the desktop UI to debug.
2. **Query-parameter versioning** (e.g. `?api_version=2`). Rejected: same
   discoverability/caching concerns and trivially defeated by clients that
   strip query strings.
3. **Date-based versioning** (Stripe-style, e.g. `2026-04-30`). Rejected as
   overkill for this project's release cadence; the operational and
   documentation overhead exceeds the benefit when most consumers are
   internal.
4. **Single-version, no prefix** (status quo). Rejected: makes coordinated
   migration impossible and forces breaking changes to be deferred
   indefinitely.

## Consequences

- Positive:
  - Clear, machine-readable deprecation signal for every deprecated endpoint.
  - Simple URL-based routing in FastAPI; no custom content-negotiation logic.
  - The default-unprefixed alias keeps existing clients working during the
    transition to explicit `v1` pinning.
  - One canonical helper to construct routers reduces the chance that a
    contributor forgets to wire deprecation headers.
- Negative:
  - URLs change shape across majors, requiring documentation updates and
    coordinated client releases at major boundaries.
  - The default-unprefixed alias must be re-aliased on major rollover; this
    requires care to avoid surprising clients that did not pin to `v1`.
- Follow-ups:
  - Add a CI lint that flags new routers not constructed via
    `make_versioned_router`.
  - Document the versioning contract in the public API reference under
    `docs/api/`.

## Validation

- Unit tests under `tests/unit/test_api_versioning.py` cover prefix
  construction, version-string validation, and `Deprecation`/`Sunset`
  header injection.
- The CI route-prefix test (`tests/unit/api/test_route_prefixes.py`) is
  extended over time to assert that every router goes through the helper.
- Release checklist requires that any breaking change references this ADR
  and bumps the major version segment.

## References

- RFC 8594, "The Sunset HTTP Header Field":
  https://www.rfc-editor.org/rfc/rfc8594
- RFC 9745, "The Deprecation HTTP Response Header Field":
  https://www.rfc-editor.org/rfc/rfc9745
- Microsoft REST API Guidelines, "Versioning":
  https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md#12-versioning
