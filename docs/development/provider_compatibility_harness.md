# Provider Compatibility Harness

The shared launcher migration uses a single compatibility harness to decide
whether a provider-backed model pack is safe to surface in UpstreamDrift.

## What It Checks

- manifest loads through the shared `ModelPackManifest` contract
- source roots, working directories, artifacts, and extra `python_paths` resolve
- models declare canonical identity metadata and at least one capability
- engine runtimes that are not installed are reported as `runtime_unavailable`
  diagnostics instead of being conflated with malformed metadata

## Tools Provider Authority

Launcher models declaring `provider: tools` use the repository-pinned
`vendor/ud-tools` gitlink as their sole production source authority. The
resolver intentionally ignores legacy `source_root: ../Tools` declarations and
workspace sibling checkouts. If `vendor/ud-tools/src` is absent because the
gitlink is missing or uninitialized, the launcher reports
`provider_unavailable`; it does not run a potentially different Tools revision.

`TOOLS_REPO_PATH` remains available to the separate Sidekick development
bootstrap, but is not accepted by this model-provider slice because the current
override contract validates directory shape rather than exact repository and
commit identity. Adding a commit-validated development override remains open
under #4262.

## Machine-Readable Diagnostics

Diagnostics are emitted with:

- `code`: stable identifier such as `missing_artifact_path`
- `category`: high-level classification such as `malformed_metadata` or
  `runtime_unavailable`
- `message`: human-readable explanation
- `context`: additional machine-readable fields like `model_id`,
  `artifact_path`, or `engine_type`

## CLI Usage

Run the shared harness against a provider repo from UpstreamDrift:

```bash
python scripts/check_provider_compatibility.py \
  --manifest ../Drake_Models/model_pack.yaml \
  --provider-root ../Drake_Models
```

Exit codes:

- `0`: provider manifest is compatible
- `1`: one or more compatibility failures were found

The script prints JSON so provider repos and agent workflows can consume the
same deterministic pass/fail contract in CI.

## Recommended Provider CI Usage

1. Generate or update `model_pack.yaml`.
2. Run `python scripts/check_provider_compatibility.py ...`.
3. Fail CI if the command exits non-zero.
4. Inspect `issues[].code` and `results[].issues[].code` to drive remediation.
