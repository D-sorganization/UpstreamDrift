# Release Process Checklist (Issue #3067)

This document addresses gaps in the release.yml workflow and the manual release process.

## Current Release.yml Workflow

The workflow is triggered on version tags matching `v*.*.*` pattern:
```yaml
on:
  push:
    tags:
      - "v*.*.*"
```

### Current Coverage

- [x] Build Python wheel and source distribution
- [x] Generate SHA256 checksums
- [x] Upload artifacts to GitHub Actions
- [x] Publish to PyPI (conditional on tag push)
- [x] Create GitHub Release with artifacts attached

### Known Gaps (Issue #3067)

#### 1. **Missing Docker Artifacts**
- **Status:** Not currently built in CI
- **Action Required:** Add Docker build stage to release.yml
- **Rationale:** Users expect Docker images for containerized deployments

```yaml
docker-build:
  needs: quality-gate
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@...
    - uses: docker/setup-buildx-action@v3
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        push: true
        tags: |
          ghcr.io/d-sorganization/upstream-drift:${{ github.ref_name }}
          ghcr.io/d-sorganization/upstream-drift:latest
```

#### 2. **Missing Tauri App Artifacts**
- **Status:** Desktop app not built/released
- **Action Required:** Add Tauri build for multi-platform releases
- **Rationale:** Users expect native installers for Windows, macOS, Linux

```yaml
tauri-build:
  needs: quality-gate
  strategy:
    matrix:
      platform: [ubuntu-latest, macos-latest, windows-latest]
  runs-on: ${{ matrix.platform }}
  steps:
    - uses: actions/checkout@...
    - uses: tauri-apps/tauri-action@v0
      with:
        releaseBody: ${{ needs.build.outputs.release_notes }}
```

#### 3. **Unconditional PyPI Publish**
- **Status:** Correctly conditional on `push && tag` event
- **Verification:** `if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')`
- **Action:** Verify PYPI_API_TOKEN is configured in repository secrets

#### 4. **Release Notes Generation**
- **Status:** Using softprops/action-gh-release with `generate_release_notes: true`
- **Enhancement Opportunity:** Generate detailed release notes from commits/PRs

#### 5. **Artifact Signing**
- **Status:** SHA256 checksums provided, GPG signing not implemented
- **Enhancement:** Add GPG signature for wheel distribution

## Pre-Release Manual Checklist

Before pushing a version tag:

1. **Update version numbers**
   - [ ] pyproject.toml: `version = "X.Y.Z"`
   - [ ] Any other version references

2. **Update changelog**
   - [ ] Add section for new version
   - [ ] List breaking changes
   - [ ] List new features
   - [ ] List bug fixes

3. **Run final test suite**
   ```bash
   python3 -m pytest -n auto --timeout=60
   python3 -m ruff check .
   python3 -m ruff format --check .
   python3 scripts/check_file_size_budget.py
   ```

4. **Create annotated tag**
   ```bash
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   git push origin vX.Y.Z
   ```

5. **Monitor release workflow**
   - [ ] ci-standard.yml passes
   - [ ] build and publish jobs complete
   - [ ] Artifacts appear on PyPI
   - [ ] GitHub Release is published

## Implementation Priority

### P0 - Fix Immediately
- Verify PYPI_API_TOKEN is configured correctly
- Document token requirement in workflow

### P1 - High Priority
- Add Docker build/push to release workflow
- Add release notes generation from commits

### P2 - Medium Priority  
- Add Tauri desktop app builds
- Add GPG signature generation

### P3 - Future Enhancement
- Automated changelog generation
- Deployment to additional registries (conda-forge, etc.)

## Related Issues

- #3065: Consolidate Jules workflows + kill switch
- #3064: Unify coverage thresholds
- #3066: Pin GitHub Actions to commit SHAs
