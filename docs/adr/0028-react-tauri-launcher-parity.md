# ADR-0028: React/Tauri launcher parity model

- Status: Accepted
- Date: 2026-06-10
- Decision Makers: UpstreamDrift maintainers
- Related Issues/PRs: #7221

## Context

The PyQt launcher owns embeddable tabs, docks, pop-out windows, and backgrounding
inside one Qt process. The React/Tauri shell currently launches tools through the
local launcher API as separate Python/Qt subprocess windows. Treating the Tauri
surface as a hidden tab host would imply embedding Python/Qt widgets inside a
webview, which is not available through the current manifest/API boundary and
would couple the web shell back to PyQt internals.

## Decision

React/Tauri uses the multi-window paradigm. PyQt remains the canonical host for
embedded tabs and docks. The React/Tauri shell stays manifest-driven through
`useLauncherManifest.ts` and opens tools through the local launcher API as
separate windows/processes.

Common parity work is limited to shell-level affordances that do not require
embedding Qt widgets in the webview:

1. Persist a launcher window registry keyed by manifest tile id.
2. Reconcile that registry with the live manifest so removed or renamed tiles do
   not leave stale launch entries.
3. Provide a window list/focus menu that reuses the existing launch API as the
   best-effort focus/reopen path.

## Alternatives Considered

1. Build a React `<TabbedHost>` and migrate tools into embedded web views.
   Rejected for now because most tools are Python/Qt applications, not web
   routes, and true embedding would require a new IPC/window-management layer.
2. Keep the parity gap undocumented. Rejected because contributors need a stable
   canonical answer before adding more React launcher features.

## Consequences

- Positive: React/Tauri launcher behavior is documented, manifest-driven, and
  testable without adding PyQt dependencies to the web shell.
- Negative: React/Tauri does not gain PyQt-style in-process tabs or docks.
- Follow-ups: If a future tool ships as a web route, it can opt into a web-native
  tab host without changing the decision for Python/Qt subprocess tools.

## Validation

`ui/src/api/launcherWindowRegistry.test.ts`, `ui/src/pages/Dashboard.test.tsx`,
and `ui/src/components/simulation/LauncherDashboard.test.tsx` verify persisted
launch records, manifest reconciliation, and the visible window focus menu.
