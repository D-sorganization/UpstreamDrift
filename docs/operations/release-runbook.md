# Release Runbook

This document describes the release process for UpstreamDrift, including pre-release verification steps and post-release tasks.

## Pre-Release Checklist

Before releasing a new version of UpstreamDrift, complete the following verification steps:

### 1. Version Bump

- [ ] Update version in `pyproject.toml`
- [ ] Update version in `SPEC.md` §1 (Identity)
- [ ] Update "Last Spec Update" date in `SPEC.md`
- [ ] Add entry to `CHANGELOG.md` (if applicable)

### 2. SPEC.md Verification

- [ ] **Verify SPEC.md is current** — Run `python3 scripts/check_spec_paths.py` to ensure all paths cited in SPEC.md exist on disk
- [ ] Review §6 (Component Locations) for accuracy
- [ ] Review §7 (Feature Status) for current feature states
- [ ] Confirm SPEC.md ownership block has current owner

### 3. Quality Gates

- [ ] All CI checks passing on main branch
- [ ] Coverage meets minimum thresholds (70% overall, 80% engine adapters)
- [ ] No outstanding security vulnerabilities (pip-audit, bandit)
- [ ] MyPy type checking passes

### 4. Testing

- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Cross-engine validation passing
- [ ] Performance benchmarks within acceptable ranges

### 5. Documentation

- [ ] README.md is up to date
- [ ] API documentation generated
- [ ] User manual reviewed (if applicable)

## Release Steps

### 1. Create Release Tag

```bash
git tag -a v<version> -m "Release v<version>"
git push origin v<version>
```

### 2. Trigger Release Workflows

The `tauri-build.yml` workflow will automatically trigger on tag push to build desktop applications.

### 3. Verify Release Artifacts

- [ ] Python package published to PyPI
- [ ] Docker image pushed to Docker Hub
- [ ] Desktop applications available in GitHub Releases
- [ ] Documentation published to GitHub Pages

## Post-Release Tasks

- [ ] Announce release on project channels
- [ ] Update release notes in GitHub Releases
- [ ] Monitor for any post-release issues

## SPEC.md Update Triggers

The following changes **must** trigger a SPEC.md update:

1. Any PR that adds, removes, or moves a top-level `src/` package or engine adapter
2. Any PR that changes the version in `pyproject.toml`
3. Any PR that changes a CI gate threshold

See `SPEC.md` §1 (Identity) and §6 (Component Locations) for required updates.