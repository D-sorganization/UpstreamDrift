---
issue: 15097
summary: "Isolate sidekick sys.modules and sys.meta_path state across overlay tests (#15097)"
---

- Add an `autouse=True` fixture in `tests/unit/launcher/test_sidekick_extension_overlay.py` that snapshots and restores `sys.meta_path` and all sidekick-related `sys.modules` entries, uninstalls any remaining `ManifestGatedSidekickFinder`, and invalidates import caches on teardown (#15097).
- Wrap `install_manifest_gated_sidekick_extensions(...)` calls in `try/finally: finder.uninstall()` and add `test_overlay_teardown_leaves_no_finder_or_tmp_sidekick_modules` to catch cross-test stub leakage.
