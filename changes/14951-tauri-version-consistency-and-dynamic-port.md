---
issue: 14951
summary: "Enforce Tauri Cargo.toml version parity, correct production readiness doc, and add frontend dynamic port resolution"
---

## Handoff

- Slices 3, 7 & 8 of #14951 (Part of #14934):
  - Slice 7: update `scripts/check_version_consistency.py` to audit `ui/src-tauri/Cargo.toml` (`[package] version`) alongside `tauri.conf.json`, align `ui/src-tauri/Cargo.toml` to 2.1.3, and add unit tests covering drift detection and real repo consistency (Oracle 13).
  - Slice 8: correct `docs/operations/production-readiness.md` to state that no bundled Python runtime is shipped today (governed by #14264 and Sidekick packaging hold) and detail local interpreter resolution under the `BackendState` supervisor.
  - Slice 3: add `DEFAULT_BACKEND_URL`, `resolveBackendBaseUrl`, `checkHealth`, `useBackendAutoStart`, and dynamic status fields to `ui/src/api/backend.ts`, with unit tests in `ui/src/api/backend.test.ts`.
