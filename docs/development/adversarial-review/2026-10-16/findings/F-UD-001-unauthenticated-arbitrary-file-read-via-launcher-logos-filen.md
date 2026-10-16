# F-UD-001

**Unauthenticated arbitrary file read via `/launcher/logos/{filename:path}`**

| Severity    | Category | Confidence | Effort | Status   |
| ----------- | -------- | ---------- | ------ | -------- |
| 🟥 Critical | security | high       | S      | proposed |

**Locations:** `src/api/routes/launcher.py:166`, `src/api/route_registry.py:248`, `src/api/utils/path_validation.py:46`

## Finding

- Severity: Critical - Category: security
- Confidence: high
- Location: src/api/routes/launcher.py:166-186; src/api/route_registry.py:248-258; src/api/utils/path_validation.py:46
- Evidence:
  ```python
  @router.get("/logos/{filename:path}")
  ...
      if ".." in filename:
          raise HTTPException(status_code=400, detail="Invalid filename")
      logo_path = ASSETS_DIR / filename
  ```
  and `_PUBLIC_ROUTERS = frozenset({"auth", "core", "observability", "capabilities", "launcher", "matched_swings", "lifting"})`.
- Failure scenario: `pathlib` discards the left operand when the right is absolute. Demonstrated with `TestClient`:
  `GET /launcher/logos//etc/hostname` → `200 application/octet-stream` with the host's contents; `%2Fetc%2Fhostname` also 200.
  The launcher router is in `_PUBLIC_ROUTERS`, so no bearer token is needed. Default bind is `127.0.0.1`, which limits blast
  radius to local processes and any browser page that can issue a cross-origin GET (CORS does not stop the read, only the JS
  response) — but the cloud/Docker profiles in `docker-compose*.yml` bind wider. `.env` files with API keys sit next to the repo.
- Why a newer engineer misses this: "contains `..`" feels like the traversal check; the absolute-path join is the other half of
  the same bug class, and Starlette's `:path` converter happily hands you a leading slash. Existing tests
  (`tests/api/test_launcher_api.py:250`, `tests/unit/api/test_routes_launcher.py:82`) assert the guard, not the property.
- Proposed change: replace the `..` check with the repo's own helper:
  ```python
  from src.api.utils.path_validation import resolve_contained_path
  logo_path = resolve_contained_path(ASSETS_DIR / filename, [ASSETS_DIR])  # raises 400/404
  ```
  and only fall through to manifest `tile.logo_path` when that path is itself contained in an allow-listed directory.
- Verification: add `test_logo_rejects_absolute_path` with `/launcher/logos//etc/hostname` and `/launcher/logos/%2Fetc%2Fhostname`
  expecting 400/404; also assert `FileResponse.path` is relative to `ASSETS_DIR`.
- Effort: S

## Draft Pull Request

**Title:** `fix(api): contain /launcher/logos paths with resolve_contained_path`

### Why This Matters (The Pitch)

Right now anyone who can reach the API can read any file the server process can read, without a token. GET /launcher/logos//etc/hostname returns 200 with the host's contents, and the percent-encoded form works too. The launcher router sits in \_PUBLIC_ROUTERS, so there is no bearer check in front of it, and the .env files holding API keys live next to the repo. Default bind is 127.0.0.1, which limits the blast radius to local processes and cross-origin browser GETs, but the cloud and Docker profiles in docker-compose\*.yml bind wider. The cruel part is that the repo already owns the correct fix: resolve_contained_path in src/api/utils/path_validation.py is simply not called here, because a '..' substring check feels like traversal defence while pathlib quietly discards the left operand of an absolute right-hand path. This is a size-S change, one helper call plus two regression tests, and it protects every operator who deploys the API beyond localhost. Nothing else in the queue turns a Critical into a closed ticket this cheaply, so it should go first.

### Proposed PR Body

## Problem

`src/api/routes/launcher.py:166-186` serves `/launcher/logos/{filename:path}` and guards only with a `..` substring check, then does `ASSETS_DIR / filename`. `pathlib` discards the left operand when the right is absolute, so `GET /launcher/logos//etc/hostname` returns `200` with the host file; `%2Fetc%2Fhostname` also works. The router is in `_PUBLIC_ROUTERS` (`src/api/route_registry.py:248-258`), so no bearer token is required.

## Change

- [ ] Replace the `..` check with `resolve_contained_path(ASSETS_DIR / filename, [ASSETS_DIR])` (from `src/api/utils/path_validation.py`), which raises 400/404
- [ ] Only fall through to manifest `tile.logo_path` when that path is itself contained in an allow-listed directory

## Verification

Add `test_logo_rejects_absolute_path` requesting `/launcher/logos//etc/hostname` and `/launcher/logos/%2Fetc%2Fhostname`, expecting 400/404; also assert `FileResponse.path` is relative to `ASSETS_DIR`.

## Risk and Rollback

Low: legitimate logos live under `ASSETS_DIR` and still resolve; manifest logo paths outside allow-listed directories will start being refused. Rollback is a plain revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-001

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
