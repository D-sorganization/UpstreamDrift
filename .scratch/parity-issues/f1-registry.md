# Machine-readable feature-parity registry + CI gate (PyQt6 = model)

## Problem

The PyQt6 launcher is the canonical application; the Tauri/React app must track it. Today parity is tracked only in prose docs that go stale (`docs/development/launcher_parity_assessment.md` is dated Feb 2026; `docs/architecture/dual-gui-architecture-review.md` estimates the web UI at ~30% coverage). Nothing stops a new PyQt6 feature from landing with no web counterpart and no record that a gap was created. The existing parity tests (`tests/config/launcher_manifest/test_parity.py`, `tests/api/test_api_parity.py`) only cover launcher tiles and route registration — not feature surface.

## Proposed fix

Implement the "feature flag" approach already recommended in `docs/architecture/dual-gui-architecture-review.md` §3.2 Step 5:

1. Add `src/config/feature_parity.json` — one entry per user-facing feature:
   ```json
   {
     "analysis.static_plots.joint_angles": {
       "pyqt": "src/shared/python/analysis/plot_engine.py",
       "api": null,
       "web": null,
       "status": "gap",
       "issue": 7460
     },
     "sidekick.os_terminal": {
       "pyqt": "src/launchers/launcher_sidekick_sidebar.py",
       "api": null,
       "web": null,
       "status": "exempt",
       "reason": "desktop-native per ADR-0028"
     }
   }
   ```
   Valid `status` values: `parity`, `gap` (requires an open `issue`), `exempt` (requires `reason`).
2. Add a loader with validation (mirror `src/config/launcher_manifest_loader.py` — typed, frozen, DbC).
3. CI test (`tests/config/feature_parity/`) that fails when:
   - an entry has `status: gap` with no open-issue number;
   - referenced `pyqt`/`web` paths don't exist;
   - a launcher-manifest tile has no corresponding registry entry.
4. Generate `docs/development/feature_parity_matrix.md` from the JSON (script in `scripts/`, freshness-checked in CI) so the human-readable matrix can never drift from the registry.
5. Seed the registry from the gap inventory in the parity epic (this issue's parent).

## Acceptance criteria

- [ ] `feature_parity.json` exists and covers all epic-listed features (parity/gap/exempt)
- [ ] CI fails on a `gap` entry without an issue, or on dangling file references
- [ ] Generated matrix doc is up to date in CI (freshness check)
- [ ] CONTRIBUTING/CLAUDE.md note: PRs adding user-facing PyQt6 features must add/update a registry entry (this is the maintainability mechanism that keeps the two apps developable in parallel)

## References

- `docs/architecture/dual-gui-architecture-review.md` §3.2 Step 5 (feature registry proposal)
- `src/config/launcher_manifest.json` + loader — the established single-source-of-truth pattern to follow
