# Source Asset and Selective Git LFS Policy

This policy governs the classification, storage destinations, and CI hydration contracts for binary source assets and model fixtures across UpstreamDrift (Part of #14895, parent #14893).

## 1. Asset Classification Hierarchy

Repository materials are partitioned into four distinct storage tiers:

| Tier | Characteristics | Examples | Storage Destination | Hydration Rule |
|---|---|---|---|---|
| **Ordinary Source Text** | Human-readable, diffable, delta-compressible text | Code, Markdown docs, JSON specs, schemas | Git repository tree | Standard clone (`depth 1` or full) |
| **Small Test Fixtures** | Deterministic golden files (< 1 MB) required across multiple test lanes | Launch monitor CSVs, golden test fixtures, test SVG icons | Git repository tree (`tests/fixtures/`) | Standard clone |
| **Source Assets & Model Fixtures** | Static binary models, physical engine tables, simulation meshes (> 1 MB) | Simscape multibody tables, synthetic sweeps, ONNX/H5 model baselines | Git repository (current) or Selective Git LFS (approved paths) | Explicit targeted hydration; fail closed if absent |
| **Durable Generated Products** | Large run outputs, video evidence, study binaries (> 1 MB) | Rendered swing frames, optimization logs, solver traces | Pinned release assets (`RELEASE_ASSETS.json`) via `fetch_release_assets.py` | Acquired on-demand; governed by `RESEARCH_ARTIFACT_STORAGE_POLICY.md` |

## 2. Reconciliation of `.gitignore` and `.gitattributes`

The repository `.gitignore` contains comments documenting historical intent:
```gitignore
# Binary models/data via LFS
*.h5
*.zip
```

### Governing Principles:
1. **`.gitignore` Does Not Govern LFS:** Adding a pattern to `.gitignore` does not configure Git LFS filters, does not move tracked files out of ordinary Git history, and provides no remote storage.
2. **Tracked Files Precedence:** Files already committed to Git history continue to be tracked regardless of `.gitignore` entries.
3. **Explicit `.gitattributes` Requirement:** Genuine Git LFS tracking requires explicit filter rules in `.gitattributes` (`filter=lfs diff=lfs merge=lfs -text`) backed by a configured remote repository endpoint (`.lfsconfig`).
4. **No Blanket Pattern Conversion:** Broad wildcards (e.g. `*.json`, `*.parquet`, `*.csv`) must never be added to LFS filters. Doing so converts small human-readable fixtures and diffable data into unhydrated pointer text files, breaking blame, line diffs, and text search.

## 3. Storage and Bandwidth Budget Separation

Git LFS usage consumes independent billing and network transfer quotas distinct from standard Git operations:

| Dimension | Git Repository (Blobs) | Git LFS Objects | GitHub Actions Storage |
|---|---|---|---|
| **Storage Account** | Included in repository size cap | Separate LFS storage quota (billed per GB/month) | 30-day transient workflow artifact retention |
| **Transfer Quota** | Included in Git clone bandwidth | Separate LFS download bandwidth quota (billed per GB) | Workflow cache and job upload/download transfer |
| **History Multiplier** | Delta-compressed packfiles | Each unique version pushed stores full uncompressed payload | Ephemeral (expires after retention window) |

## 4. Selective CI Hydration Contracts

Unselective LFS checkouts (`git lfs pull`) in CI waste runner time and consume download bandwidth:

1. **Source-Only Jobs:** Linting, formatting, documentation governance, and pure logic unit tests must skip LFS downloads (`GIT_LFS_SKIP_SMUDGE=1`).
2. **Asset-Dependent Jobs:** Jobs requiring physical simulation models or large fixtures must explicitly hydrate only their declared dependencies.
3. **Fail-Closed Verification:** Asset consumers must verify that on-disk files are genuine hydrated binary payloads rather than unhydrated LFS pointer stubs (~130 bytes text starting with `version https://git-lfs.github.com/spec/v1`). If an asset is missing or unhydrated, the job must fail closed with structured diagnostics via `scripts/assets/source_asset_inventory.py`.

## 5. Migration Safety and History Integrity

No existing Git history is rewritten by this policy:
- Existing tracked binary blobs remain in their established commits.
- Any future historical LFS migration requires explicit owner approval, a coordinated fleet freeze, and a fully reversible rollout plan.
