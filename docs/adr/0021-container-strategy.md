# ADR-0021: Root Container Surface Policy

- Status: Accepted
- Date: 2026-05-24
- Decision Makers: dieterolson, codex (AI)
- Related Issues/PRs: #6097, #5768

## Context

UpstreamDrift ships three Dockerfiles at the repo root:

- `Dockerfile`
- `Dockerfile.heavy_test`
- `Dockerfile.modular`

Each file already serves a real workflow, but that role is easy to miss from a
fresh checkout:

- `docker-compose.yml` builds `Dockerfile` for the default backend and training
  services.
- `docker-compose.profiles.yml` builds `Dockerfile.modular` for named profile
  images.
- `scripts/ci/run_local_heavy_tests.sh` and
  `scripts/check_heavy_dep_parity.py` treat `Dockerfile.heavy_test` as the
  heavy-test parity image.
- `docker-smoke.yml`, `docker-size-gates.yml`, and
  `docker-security-scan.yml` already gate different parts of that surface area.

The ambiguity was not that the files were unused; it was that the repo did not
state which of them was canonical for which workflow. That made
`Dockerfile.modular` look half-promoted and invited the "which Dockerfile do I
edit?" failure mode.

## Decision

Keep all three root Dockerfiles, but assign each an explicit and stable role.

| File | Canonical role | Used by |
| --- | --- | --- |
| `Dockerfile` | Default release/runtime/training image | `docker-compose.yml`, `docker-compose.gpu.yml`, `docker-security-scan.yml`, runtime/training release flows |
| `Dockerfile.heavy_test` | Heavy-test parity image only | `scripts/ci/run_local_heavy_tests.sh`, `heavy-tests-opt-in.yml`, `scripts/check_heavy_dep_parity.py` |
| `Dockerfile.modular` | Opt-in modular profile build surface | `docker-compose.profiles.yml`, `docker-smoke.yml`, modular profile matrix in `docker-size-gates.yml`, `docker/profiles.yaml` |

Additional policy:

1. `Dockerfile` remains the canonical root image for the default developer and
   release path.
2. `Dockerfile.modular` is supported, but only for opt-in profile builds. It
   is not a hidden replacement for `Dockerfile`.
3. `Dockerfile.heavy_test` is intentionally separate from the release path and
   exists only to mirror the heavy-test runner environment.
4. `docker/README.md` is the operational policy file for contributors. If any
   workflow role changes, that README and the root Dockerfile headers must be
   updated in the same PR.
5. Promoting `Dockerfile.modular` to replace `Dockerfile`, or retiring it
   entirely, requires a new ADR rather than silent drift.

## Alternatives Considered

1. **Promote `Dockerfile.modular` to replace `Dockerfile` now**.
   Rejected because the current compose files, release path, and security scan
   still treat `Dockerfile` as the default runtime surface.
2. **Retire `Dockerfile.modular` now**.
   Rejected because the repo already has active compose and CI flows that build
   profile images from it, so deletion would remove supported functionality.
3. **Keep all three without a policy document**.
   Rejected because that is the ambiguous state issue #6097 was filed to fix.

## Consequences

- Positive:
  - Contributors get a single "edit routing" source of truth.
  - `Dockerfile.modular` stops looking like an undocumented migration artifact.
  - CI ownership is clearer: release/runtime checks target `Dockerfile`, profile
    smoke/size checks target `Dockerfile.modular`, and heavy parity stays
    isolated.
- Negative:
  - The repo still carries three root Dockerfiles, so policy drift must be
    guarded with documentation and tests.
  - A future container consolidation still needs an explicit decision pass.
- Follow-ups:
  - Revisit whether modular profiles should replace the default runtime image
    only when release, compose, and security-scan paths are ready for a
    coordinated switch.

## Validation

- `tests/unit/repo_hygiene/test_docker_docs.py` locks the ADR, `docker/README`,
  and root Dockerfile header cross-links in place.
- `scripts/check_heavy_dep_parity.py` continues to enforce the documented
  `Dockerfile.heavy_test` contract.
- Existing Docker workflows remain aligned with the policy:
  `docker-security-scan.yml` for `Dockerfile`,
  `docker-smoke.yml` plus the profile matrix in `docker-size-gates.yml` for
  `Dockerfile.modular`.
