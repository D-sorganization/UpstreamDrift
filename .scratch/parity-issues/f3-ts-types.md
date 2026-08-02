# Generate TypeScript API types from Pydantic models in CI (contract drift guard)

## Problem

The web app hand-maintains TypeScript shapes for API payloads (`ui/src/api/schemas.ts` runtime parsing, ad-hoc interfaces in `ui/src/api/*.ts`), while the contract source of truth is the Pydantic models in `src/api/models/`. Hand-synced types are exactly the kind of silent divergence that has already required runtime-validation patches (#7165). As parity work adds many new endpoints (static plots, counterfactuals, exports), hand-syncing will not scale.

## Proposed fix

Per `docs/architecture/dual-gui-architecture-review.md` §3.2 Step 4:

1. Add a generation script (e.g. `scripts/generate_ui_api_types.py`) using `pydantic-to-typescript` (or datamodel-codegen via the FastAPI OpenAPI schema, which avoids a new dependency and covers route signatures too — evaluate both, prefer the OpenAPI route since `src/api/server.py` already exposes the schema).
2. Emit to `ui/src/api/generated/types.ts` (checked in).
3. CI check: regenerate and `git diff --exit-code` so the generated file can never be stale (same pattern as other freshness checks in this repo).
4. Migrate `ui/src/api/*.ts` call sites to the generated types incrementally; new parity endpoints MUST use generated types from day one.
5. Keep runtime validation (`apiFetchParsed`) for boundary safety; generated types complement it.

## Acceptance criteria

- [ ] Generated types file + CI freshness gate
- [ ] At least the engine, simulation, launcher-manifest, and analysis payloads consumed from generated types
- [ ] Contributor doc note in `ui/README` (or CLAUDE.md): never hand-write a payload interface that exists in `src/api/models/`

## References

- #7165 (runtime manifest validation — symptom of hand-synced contracts)
- `src/api/models/requests.py`, `src/api/models/responses.py`
