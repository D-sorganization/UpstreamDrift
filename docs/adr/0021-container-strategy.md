# ADR-0021: Container Strategy — Three-Dockerfile Policy

Status: Accepted
Date: 2026-05-25
Issue: #6097

## Context

The repository contains three Dockerfiles, each serving a distinct purpose, but
their roles were not formally documented, creating ambiguity about which file CI
targets and which files developers should edit for particular use-cases.

**`Dockerfile`** — multi-stage build (builder → runtime → training). Uses a
pinned `python:3.12-slim` digest for reproducible production images. Runs the
FastAPI server on port 8001. Contains a full dependency stack (Pinocchio, MuJoCo
headless, PyTorch/CUDA training stage). This is the file targeted by
`docker-smoke.yml`, `docker-size-gates.yml`, and `docker-security-scan.yml`.

**`Dockerfile.heavy_test`** — single-stage `python:3.11-bookworm` image
designed to mirror the `d-sorg-fleet-4core` custom GitHub Actions runner. Installs
X11/Xvfb, SDL2, and the full physics/robotics engine stack (Pinocchio, MuJoCo,
Drake, OpenSim, MyoSuite — best-effort). Default CMD runs the heavy integration
test suite under `xvfb-run`. Built and used locally via
`wsl bash run_local_heavy_tests.sh`; kept in sync with
`.github/workflows/heavy-tests-opt-in.yml`.

**`Dockerfile.modular`** — experimental two-stage build that selects features
via named `--build-arg PROFILE=<name>` or `--build-arg FEATURES=<list>`
arguments. Feature resolution is delegated to
`scripts/docker/install_features.py` and
`src/shared/python/feature_registry/features.py`. Explicitly non-canonical
during Phase 2. Targeted by `docker-smoke.yml` (profile capability checks)
and `docker-size-gates.yml` (profile size budget matrix) — but NOT by the
security scan or the canonical size gate that uses `Dockerfile` directly.

The three options considered were:

1. **Consolidate** — merge all functionality into a single Dockerfile using
   multi-stage targets and build-args.
2. **Promote** `Dockerfile.modular` — retire the legacy `Dockerfile` and make
   the modular one canonical.
3. **Keep all three with explicit documented roles** — formalise the status quo
   with clear per-file policy.

## Decision

We adopt **Option 3**: keep all three Dockerfiles with explicit, documented roles.

| File                    | Role                                    | CI scope                                                                          | When to edit                                                       |
| ----------------------- | --------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `Dockerfile`            | Canonical production build              | docker-smoke, docker-size-gates, docker-security-scan                             | Production dependency changes, server config, hardening            |
| `Dockerfile.heavy_test` | Test environment mirroring fleet runner | heavy-tests-opt-in (uses runner directly)                                         | When runner apt packages or test-suite requirements change         |
| `Dockerfile.modular`    | Experimental / modular-build validation | docker-smoke (profile capability checks), docker-size-gates (profile size matrix) | Local exploration of modular profiles only — see constraints below |

Rationale:

- Consolidation (Option 1) would couple three very different concerns into one
  complex Dockerfile with many `--target` arguments, increasing cognitive load
  and the risk of production regressions from test-env changes.
- Promoting `Dockerfile.modular` (Option 2) is premature while Phase 2
  validation is still in progress. The feature registry is not yet stable enough
  to be the sole production build path.
- Option 3 preserves CI stability while keeping each file focused and readable.

## Consequences

- **`Dockerfile` is the authoritative file for production changes.** Security
  patches, dependency upgrades, and server configuration changes go here first.
- **`Dockerfile.heavy_test` must stay in sync with the fleet runner.** Changes
  to the runner's apt packages or Python tool versions should be reflected here.
- **`Dockerfile.modular` is in limited CI scope** (profile smoke and size-budget
  matrix only). It is NOT covered by the security scan or the canonical size
  gate. Developers using it for local experiments should not assume it reflects
  the current production dependency set.
- **Any future modularization effort that promotes `Dockerfile.modular` to
  production must file a new ADR** (superseding this one) rather than silently
  modifying the file or wiring it into CI without governance review.
- Operators should consult `docker/README.md` for a quick-reference table before
  choosing which Dockerfile to build.
