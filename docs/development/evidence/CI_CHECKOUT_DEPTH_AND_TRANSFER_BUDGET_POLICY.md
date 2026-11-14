# CI Checkout Depth and Transfer Budget Policy

This policy governs GitHub Actions checkout depth configurations, transfer bandwidth budgets, and sparse-state regression prevention across UpstreamDrift (Part of #14896, parent #14893).

## 1. Checkout Depth Qualification Rules

GitHub Actions jobs must not default to full history checkouts (`fetch-depth: 0`) without documented technical justification:

| Checkout Category | Permitted Depth | Eligible Jobs | Justification & Contracts |
|---|---|---|---|
| **Shallow (Default)** | `fetch-depth: 1` | `unit-test-gate`, `rust-quality-gate`, `lint`, `format`, `docs-governance-gates` | Job only requires current HEAD tree; minimizes network payload and runner disk consumption. |
| **Full History** | `fetch-depth: 0` | `spec-check`, `changed-paths`, `pr-size`, `seam-drift-gate`, `release`, `publication-quality` | Requires full commit graph, merge base resolution, tag lineage, or git blame calculation. |
| **Pinned Tools** | `fetch-depth: 1` at gitlink ref | Jobs requiring `vendor/ud-tools` (`pip install -e .`) | Managed via `.github/actions/fetch-pinned-tools` at exact superproject gitlink; avoids full recursive submodule checkout. |

## 2. Sparse-State Regression & Reused Runner Isolation

Self-hosted runners persist their `_work` directories across job invocations:

1. **The Sparse-Checkout Poisoning Risk:** When a job enables `git sparse-checkout` or leaves `skip-worktree` bits in the git index, subsequent jobs running in that reused workspace inherit the sparse configuration. This causes `actions/checkout` or submodule commands to silently omit files (e.g. `package-lock.json`, `vendor/ud-tools` headers), leading to secondary build failures.
2. **Completeness Assertion Contract:** Every job that checks out the repository on self-hosted or reused infrastructure must invoke `scripts/ci/assert_checkout_complete.py` immediately following checkout.
3. **No Global Sparse Toggles:** Sparse checkout must never be enabled globally across a runner workspace. If selective extraction is required, it must occur in an isolated scratch directory or throwaway worktree.
4. **Healing Contract:** Standard checkout actions must verify that `core.sparseCheckout` is disabled when a full checkout is requested, clearing any leftover stale `skip-worktree` bits.

## 3. Cache Integrity and Intentional Purges

Pip and wheel caches provide build acceleration, but on shared runners they introduce cache collision risks:
- **Intentional Pip Purges:** Purging the pip cache in selected unit test lanes is an intentional defense against corrupted ABI wheels or binary cross-contamination across disparate matrix jobs.
- **Tool Cache Isolation:** Python and Rust toolchains must remain isolated using lockfiles, exact toolchain versions, and revision-specific keys.

## 4. Verification and Non-Regression Boundaries

All workflow checkout changes must be validated against `scripts/ci/qualify_checkout_depth.py`:
- Unjustified `fetch-depth: 0` settings will fail CI qualification.
- No historical commits or tags may be omitted from jobs that enforce versioning or provenance gates.
