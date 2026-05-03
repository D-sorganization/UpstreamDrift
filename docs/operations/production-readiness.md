# Production Readiness

This document is the source of truth for release deliverables and support
commitments. README files and release automation should link here instead of
repeating version or platform lists.

## Canonical Production Artifacts

UpstreamDrift ships exactly four production artifacts. Everything else in this
repository is a development or test convenience.

| Artifact | Source | Distribution channel | Audience |
| --- | --- | --- | --- |
| `upstream-drift` wheel + sdist | `pyproject.toml` build via `release.yml` | PyPI | Library users, CI/CD pipelines |
| `upstream-drift-api` Docker image | `Dockerfile` build via `release.yml` | GHCR | Self-hosted API operators |
| Tauri desktop app | `ui/` build via `tauri-build.yml` | GitHub Releases | End users, researchers, and practitioners |
| `upstream-physics` Rust crate | `rust_core/upstream-physics/` via `release.yml` | crates.io or bundled native extension | Performance-sensitive embedders |

Items not shipped as production:

- MATLAB Simscape models in `src/engines/Simscape_Multibody_Models/` are
  research and comparison references only.
- `Dockerfile.heavy_test` is a CI-only test image.
- Launchers in `src/launchers/*.py`, except the public console script declared
  in `[project.scripts]`, are development conveniences.

## Compatibility Matrix

| Artifact | OS | Python | Tier(s) supported | Hardware |
| --- | --- | --- | --- | --- |
| Python wheel | Linux x86_64, macOS arm64, Windows 10+ x86_64 | 3.10-3.13 | core; +extras | CPU |
| Docker image (API) | Linux x86_64 | 3.11 | core+extended | CPU; optional CUDA 12 |
| Tauri desktop | Linux x86_64, macOS arm64, Windows 10+ x86_64 | bundled | core+extended | CPU |
| Rust crate | Linux, macOS, Windows | n/a | n/a | CPU |

Supported means a release has a green smoke test in `tests/smoke/` for the
artifact and the combination is on the release checklist. Combinations outside
this matrix are best-effort.

## Release-Blocking Smoke Tests

Each canonical artifact owns a smoke-test directory:

| Artifact | Smoke suite | Artifact input |
| --- | --- | --- |
| Python wheel | `tests/smoke/python_wheel/` | `dist/upstream_drift-*.whl` |
| Docker image (API) | `tests/smoke/docker_api/` | `UPSTREAM_DRIFT_API_IMAGE` |
| Tauri desktop | `tests/smoke/tauri_desktop/` | `UPSTREAM_DRIFT_TAURI_BUNDLE` |
| Rust crate | `tests/smoke/rust_crate/` | `rust_core/upstream-physics/Cargo.toml` |

The existing tag release workflow blocks PyPI and GitHub Release publication
until the built Python wheel passes its matrix smoke tests. Docker, desktop,
and Rust artifact jobs must call their matching smoke suite before their
publish steps when those release jobs are enabled.
