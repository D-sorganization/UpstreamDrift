# Release Runbook

This runbook defines the human-operated release process for UpstreamDrift.
`pyproject.toml` `[project].version` is the canonical project version. The
release version must stay aligned with `src/api/_version.py`,
`ui/package.json`, root `Cargo.toml`, and
`rust_core/upstream-physics/pyproject.toml`.

## Release Cadence

- Minor releases: monthly when the main branch has production-ready changes.
- Patch releases: weekly as needed for regressions, security fixes, and
  downstream unblockers.
- Release candidates: optional for changes that alter public API, packaging,
  or native engine behavior.

## Pre-release

1. Verify all PRs targeting the milestone are merged.
2. Verify CI is green on `main`: `ci-standard`, `docker-security-scan`, and
   `docs-ci`.
3. Update `CHANGELOG.md`: move relevant entries from `[Unreleased]` to
   `[v<X.Y.Z>] - YYYY-MM-DD`.
4. Bump `pyproject.toml` to `X.Y.Z`.
5. Align all generated or mirrored version surfaces:
   `src/api/_version.py`, `ui/package.json`, root `Cargo.toml`, and
   `rust_core/upstream-physics/pyproject.toml`.
6. Update `SPEC.md` §1 (Identity), including "Last Spec Update" when the
   release changes public version metadata.
7. Verify `SPEC.md` is current by running
   `python3 scripts/check_spec_paths.py`.
8. Review `SPEC.md` §6 (Component Locations) and §7 (Feature Status) for
   accuracy.
9. Run `python3 scripts/check_version_consistency.py`.
10. Verify all release-blocking smoke tests for artifacts being published.
11. Open a release PR and require CODEOWNERS approval.

## Release

1. Merge the release PR to `main`.
2. Create and push a signed tag from a trusted workstation:
   `git tag -s vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z`.
3. Run schema migrations against the production database before starting the new server version:
   `python3 scripts/db_migrate.py upgrade head`.
4. Confirm `.github/workflows/release.yml` starts from the tag.
5. Verify the workflow builds wheels and source distributions.
6. Verify `SHA256SUMS.txt`, the CycloneDX SBOM, and GitHub artifact
   attestations are attached to the release.
7. Verify the PyPI publish job completes for `upstream-drift`.
8. Verify a clean environment resolves the package:
   `pip install upstream-drift==X.Y.Z`.

## Post-release

1. Open a fresh `[Unreleased]` section in `CHANGELOG.md`.
2. Bump `pyproject.toml` to the next development version, such as
   `X.Y.(Z+1).dev0`, and align mirrored version surfaces.
3. Run `python3 scripts/check_version_consistency.py`.
4. Announce the release to downstream consumers, including
   `Gasification_Model` and Tools maintainers.
5. File follow-up issues for any release workflow warning, SBOM gap, or
   downstream pinning problem.

## Rollback

1. Locate the previous release tag with `git tag --list "v*.*.*"`.
2. Pin downstream consumers to the previous known-good version.
3. Re-run the release workflow for the previous tag only if artifacts are
   missing or corrupted.
4. File an incident issue documenting the regression, affected versions,
   mitigation, and owner.
5. Ship a patch release after the fix is merged and verified.

## SPEC.md Update Triggers

The following changes must trigger a `SPEC.md` update:

1. Any PR that adds, removes, or moves a top-level `src/` package or public
   engine adapter.
2. Any PR that changes the version in `pyproject.toml`.
3. Any PR that changes a CI gate threshold.

See `SPEC.md` §1 (Identity) and §6 (Component Locations) for required updates.

## Operational Limits

This development environment must not create or push release tags and must not
publish to PyPI. Those actions require a human release operator with signing
keys, repository release permissions, and PyPI project authority.
