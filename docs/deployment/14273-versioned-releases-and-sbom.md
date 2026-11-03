# Versioned Releases With Verifiable Artifacts and SBOM (#14273)

## Overview

As part of the professional deployment and packaging epic ([#14276](https://github.com/D-sorganization/UpstreamDrift/issues/14276)),
UpstreamDrift enforces verifiable release bundles for every versioned release.
Release artifacts must be cryptographically attested, self-contained, reproducible,
and accompanied by multi-tier Software Bill of Materials (SBOM) documents in standard
industry formats.

## Artifact Contract

For any release version `vX.Y.Z`, the release distribution directory (`dist/`) must
contain the following verifiable components:

1. **Python Distribution Wheel**: `upstream_drift-X.Y.Z-py3-none-any.whl`, built with Hatchling,
   containing complete package metadata, root modules, and the compiled frontend bundle
   (`ui/dist/index.html` and assets).
2. **Cryptographic Checksum Manifest**: `SHA256SUMS.txt` containing SHA-256 digests of all
   distributed files. Every file in `dist/` must match its manifest digest; extraneous or untracked
   files fail verification.
3. **Multi-Tier SBOM Documents**: Generated across all dependency scopes defined in
   `scripts/security/sbom_tiers.py` in both CycloneDX and SPDX formats:
   - `core` (`upstream-drift==X.Y.Z`): Minimal server, CLI, and physics dependencies.
   - `extended` (`upstream-drift[all-engines]==X.Y.Z`): Core plus external engine backends.
   - `full` (`upstream-drift[all]==X.Y.Z`): Full suite including optional GUI, analytics, and dev tools.
4. **Build Provenance**: GitHub Actions build provenance attestations (`actions/attest-build-provenance`).

## Verification Workflow

Downstream consumers and CI pipelines verify the integrity of release bundles using
`scripts/ci/verify_release_artifacts.py`:

```bash
# Verify checksums, wheel metadata, and tiered SBOMs
python -m scripts.ci.verify_release_artifacts --dist-dir dist/ --version X.Y.Z

# Emit structured JSON report
python -m scripts.ci.verify_release_artifacts --dist-dir dist/ --version X.Y.Z --json
```

The verifier executes fail-closed checks against:

- Checksum integrity against `SHA256SUMS.txt`
- Wheel metadata validity and UI bundle inclusion
- Structural conformity of CycloneDX (`bomFormat`, `specVersion`, components) and SPDX (`spdxVersion`, `SPDXID`, `packages`) documents
- Complete presence of all required tier artifacts without untracked files
