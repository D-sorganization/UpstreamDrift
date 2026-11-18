---
issue: 15240
summary: "Honor package anchors in safe_import relative module resolution"
---

Use importlib.import_module so safe_import resolves relative names against the documented package argument. Preserve missing-dependency fallback and initialization-error propagation, with direct regression tests for nested and parent-relative module identity.
