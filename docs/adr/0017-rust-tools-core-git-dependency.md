> **Note**: This ADR was formerly numbered ADR-0005 before renaming to resolve a numbering collision.

# ADR 0017: Pin `tools-core` as a Git Dependency

- Status: Accepted
- Date: 2026-04-23
- Related issues: #3077

## Context

UpstreamDrift's Rust workspace depends on `tools-core` from the
`D-sorganization/Tools` repository. The previous dependency contract used a
local path:

```toml
tools-core = { path = "../Tools/rust_core/tools-core" }
```

That forced both developers and CI jobs to create a sibling `../Tools`
checkout or symlink before `cargo build` or `maturin develop` could start.
A clean clone of only UpstreamDrift failed early with a raw Cargo manifest
error, and CI carried the same repository-layout knowledge in duplicated
checkout and symlink steps.

## Decision

Pin `tools-core` as a git dependency in the root Cargo workspace:

```toml
tools-core = { git = "https://github.com/D-sorganization/Tools.git", rev = "<pinned-revision>" }
```

Keep the existing `scripts/setup_tools_workspace.sh` helper only for optional
cross-repository Python workflows that still benefit from a local `Tools`
checkout and `PYTHONPATH` wiring.

## Rationale

- A clean `git clone` of UpstreamDrift can now run `cargo build` immediately.
- `maturin develop --features python` no longer depends on a sibling checkout.
- CI no longer needs Rust-only `_tools_dep` checkouts and symlink hacks.
- Pinning a specific revision preserves reproducibility and makes updates
  reviewable in normal PRs.

## Consequences

### Positive

- Rust quickstart instructions become accurate for clean clones.
- Tauri and Rust CI jobs are simpler and less coupled to repo layout.
- Missing sibling workspaces no longer block basic Rust development.

### Negative

- Updating `tools-core` now requires an explicit revision bump in UpstreamDrift.
- Cargo fetches the Tools repository over git for Rust builds, which adds a
  network dependency when the crate is not already cached.

## Validation

- `cargo metadata --no-deps`
- `cargo build`
- `python -m maturin develop --features python`
