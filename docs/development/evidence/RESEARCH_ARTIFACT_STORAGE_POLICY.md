# Research Artifact Storage and Evidence Preservation Policy

This policy defines the governance, classification, and durable storage lifecycle for research outputs and scientific evidence across UpstreamDrift (Part of #14894, parent #14893).

## 1. Output Classification Hierarchy

All research artifacts and generated materials must be categorized into one of four classes:

| Class | Description | Storage Location | Retention Requirement |
|---|---|---|---|
| **Source Inputs** | Unmodified input trajectories, laboratory capture data, licensed meshes, hardware logs | Pinned release assets or Git source tree (if < 1 MB) | Permanent |
| **Irreplaceable Evidence** | Optimization run receipts, validation verdicts (ACCEPTED and REJECTED), provenance chains, reviewer sign-offs | Git repository (`docs/development/*/evidence/`) | Permanent |
| **Reproducible Products** | Motion bundles, rendered stills/videos, summary tables generated from deterministic inputs | Pinned release assets or Git LFS | Pinned to release/commit |
| **Disposable Scratch** | Temporary solver logs, intermediate test outputs, local cache directories | `scratch/` or OS temp directories | Ephemeral (auto-cleaned) |

> [!IMPORTANT]
> **Preservation of Negative Results:** A run verdict of `REJECTED` or a failed gate is irreplaceable scientific evidence. Negative results must never be reclassified as disposable scratch or deleted merely because a solver or test failed.

## 2. Provenance and Checksum Binding

Every research artifact that survives beyond an individual run must be cryptographically bound to its complete generation lineage:
1. **Source Model & Spec Hashes:** SHA-256 of the exact input specification files (e.g. `full_body_spec_*.json`).
2. **Engine & Runtime Identity:** Physics engine version, solver tolerances, integrator type, and operating system / architecture.
3. **Execution Recipe:** The exact CLI command or script invocation required to reproduce the output.
4. **Content Address:** SHA-256 checksum and exact byte count recorded in `RELEASE_ASSETS.json`.

## 3. Storage Tiers and Lifecycle

| Storage Tier | Typical Duration | Use Case | Integrity Enforcement |
|---|---|---|---|
| **Git Repository** | Permanent | Text receipts, JSON contracts, small thumbnails (≤ 300 KB) | Git tree SHA |
| **Release Assets** | Permanent | Large archives (> 1 MB), validation image matrices, video evidence | Checksum-indexed via `fetch_release_assets.py` |
| **Actions Artifacts** | 30 days (transient) | CI test logs, temporary compilation wheels | Disposable build cache |

## 4. Fail-Closed Integrity Enforcement

When test suites, CI lanes, or local analysis pipelines depend on external release assets:
- **Offline / Missing Evidence:** If an asset cannot be verified or downloaded, the consumer must fail closed with an explicit error (`AssetIntegrityError` or `AssetMissingError`).
- **Never Synthesize Passes:** Missing or unverified evidence must never be converted into a false pass, zero residue, or default green status.
- **Atomic Materialization:** Downloads are written to temporary sibling files and atomically replaced only after byte count and SHA-256 verification succeed.
