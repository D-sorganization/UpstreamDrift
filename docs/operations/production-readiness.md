# Production Readiness

This document is the source of truth for production release surfaces,
deliverables, version governance, and support commitments. README files and
release automation should link here instead of repeating version or platform
lists.

## Production Surfaces

- Python library and CLI package published as `upstream-drift` on PyPI.
- FastAPI service when deployed by an operator from the released package or
  container image.
- Tauri desktop application when attached to a GitHub release.
- Rust physics kernels when shipped as part of the Python package or as
  buildable workspace crates.

Experimental notebooks, archived engine code, local assessment output,
developer-only scripts, `Dockerfile.heavy_test`, and MATLAB Simscape reference
models are not production surfaces.

## Canonical Production Artifacts

UpstreamDrift ships exactly four production artifacts. Everything else in this
repository is a development or test convenience.

| Artifact                          | Source                                          | Distribution channel                  | Audience                                  |
| --------------------------------- | ----------------------------------------------- | ------------------------------------- | ----------------------------------------- |
| `upstream-drift` wheel + sdist    | `pyproject.toml` build via `release.yml`        | PyPI                                  | Library users, CI/CD pipelines            |
| `upstream-drift-api` Docker image | `Dockerfile` build via `release.yml`            | GHCR                                  | Self-hosted API operators                 |
| Tauri desktop app                 | `ui/` build via `tauri-build.yml`               | GitHub Releases                       | End users, researchers, and practitioners |
| `upstream-physics` Rust crate     | `rust_core/upstream-physics/` via `release.yml` | crates.io or bundled native extension | Performance-sensitive embedders           |

## Version Contract

- `pyproject.toml` `[project].version` is canonical.
- `src/api/_version.py`, `ui/package.json`, root `Cargo.toml`, and
  `rust_core/upstream-physics/pyproject.toml` must match the canonical version.
- The canonical version must match the latest `vX.Y.Z` tag or be ahead of it
  during release preparation.
- `scripts/check_version_consistency.py` is the CI gate for this contract.

## Compatibility Matrix

| Artifact           | OS                                            | Python    | Tier(s) supported | Hardware              |
| ------------------ | --------------------------------------------- | --------- | ----------------- | --------------------- |
| Python wheel       | Linux x86_64, macOS arm64, Windows 10+ x86_64 | 3.10-3.13 | core; +extras     | CPU                   |
| Docker image (API) | Linux x86_64                                  | 3.11      | core+extended     | CPU; optional CUDA 12 |
| Tauri desktop      | Linux x86_64, macOS arm64, Windows 10+ x86_64 | bundled   | core+extended     | CPU                   |
| Rust crate         | Linux, macOS, Windows                         | n/a       | n/a               | CPU                   |

Supported means a release has a green smoke test in `tests/smoke/` for the
artifact and the combination is on the release checklist. Combinations outside
this matrix are best-effort.

## Service Objectives

- API readiness: `/health` returns success within 500 ms p95 on the reference
  deployment profile.
- API launch metadata: launcher listing endpoints return within 500 ms p95 for
  the default bundled engine set.
- Engine determinism: dependency-free validators and core unit tests pass
  across the supported Python matrix.
- Packaging: `pip install upstream-drift==X.Y.Z` resolves in a clean virtual
  environment for every published production release.
- Release integrity: release artifacts include checksums, a CycloneDX SBOM,
  and GitHub artifact attestations.

## Release-Blocking Smoke Tests

Each canonical artifact owns a smoke-test directory:

| Artifact           | Smoke suite                  | Artifact input                          |
| ------------------ | ---------------------------- | --------------------------------------- |
| Python wheel       | `tests/smoke/python_wheel/`  | `dist/upstream_drift-*.whl`             |
| Docker image (API) | `tests/smoke/docker_api/`    | `UPSTREAM_DRIFT_API_IMAGE`              |
| Tauri desktop      | `tests/smoke/tauri_desktop/` | `UPSTREAM_DRIFT_TAURI_BUNDLE`           |
| Rust crate         | `tests/smoke/rust_crate/`    | `rust_core/upstream-physics/Cargo.toml` |

The existing tag release workflow blocks PyPI and GitHub Release publication
until the built Python wheel passes its matrix smoke tests. Docker, desktop,
and Rust artifact jobs must call their matching smoke suite before their
publish steps when those release jobs are enabled.

## Supported Environments

- Python: 3.10, 3.11, and 3.12 for the published package; wheel smoke tests also
  cover 3.13 where release artifacts are built.
- Operating systems: Linux for CI and service deployments; Windows 10+ and
  macOS arm64/x64 for desktop artifacts when produced by the Tauri workflow.
- Native engines: MuJoCo is part of the default package dependency set. Drake,
  Pinocchio, OpenSim, and MyoSuite are optional and only supported when their
  extras and platform prerequisites are installed.
- Rust: the pinned toolchain in `rust-toolchain.toml`.

## Support Window

- Latest two minor releases receive regression fixes.
- Security fixes target the latest minor release first and may be backported
  one minor version when downstream consumers cannot upgrade immediately.
- Pre-release and `.dev0` builds are unsupported outside release validation.

## Release Gates

- CI is green on the release PR.
- `CHANGELOG.md` has a dated release section for the target version.
- `scripts/check_version_consistency.py` passes.
- The release workflow publishes artifacts, checksums, an SBOM, and
  attestations.
- Release-blocking smoke tests pass for every artifact being published.
- PyPI installation is verified after publish by a human release operator.

## Open Operational Follow-up

- Tag creation and PyPI publication are intentionally human-operated and cannot
  be completed from development worktrees.
- Documentation hosting must be checked for each release version at
  `https://upstream-drift.readthedocs.io`.
