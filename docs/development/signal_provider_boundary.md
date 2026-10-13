# Signal Provider Boundary

Tools owns the overlapping Signal Toolkit implementation. UpstreamDrift's
`_seam_redirect.REDIRECTED_ROOTS` routes its legacy package spelling to the exact
`vendor/ud-tools` provider. The UD-only `signal_processing.py` remains a split
extension on the canonical package path. The gitlink, Python dependency and
Rust dependency must identify the same published Tools revision.

## Public Identity and Compatibility

`signal_toolkit`, `shared.python.signal_toolkit` and
`src.shared.python.signal_toolkit` must resolve to one package and one `Signal`
class. Fitting, I/O and optional widget modules must retain that same class even
when imported during package initialization. Private provider namespaces and
late cache replacement cannot satisfy this contract.

Tools issues #5569, #5571 and #5572 cover optimized-Python structural checks,
fitting compatibility and early import aliases. Preserve the six empty-input
fit guards and four failed-fit array fallback identities from UD #5912. Keep
the provider's numerical formulas, typed model outputs and tolerances. This
integration does not rerun or qualify the frozen shot-pattern simulations.

## Verification

Run from the repository root with the declared development environment and
headless GUI settings:

```bash
python -m pytest tests/unit/repo_hygiene/test_signal_provider_boundary.py -q
python -m pytest tests/unit/shared_python/test_seam_redirect.py -q
python scripts/shared_tools/check_seam_drift.py
```

The boundary test starts a fresh process for each import order, checks provider
origin and all loaded consumers' class references, and verifies that the UD-only
extension remains reachable. Passing these checks proves the import boundary;
the full repository gate and published-provider pin check remain separate
requirements.
