# Diagnostics parity: desktop has full diagnostics + integrations-health; web panel is Tauri-only and shallow

## Gap (PyQt6 = model)

Desktop diagnostics (`src/launchers/launcher_diagnostics.py`, 46KB) checks model registry integrity, asset files, engine availability with versions, dependency health (NumPy/SciPy/MuJoCo/pydrake/Pinocchio/OpenSim/MyoSuite), and git metadata; plus a separate Integrations Health dashboard (`src/launchers/integrations_health_panel.py` — MCP/CLI/API probes with status colors, refresh, copy-as-markdown with secret redaction). The web `DiagnosticsPanel` (`ui/src/components/ui/DiagnosticsPanel.tsx`) only shows backend process status and Python env, and parts are Tauri-only — browser mode gets almost nothing.

The probe logic is already headless (`src/launchers/integrations_health_data.py`), so this is mostly an exposure problem.

## Proposed fix

1. Expand `GET /api/diagnostics` (or add `GET /api/v1/diagnostics/full`) to return the same structured report the desktop diagnostics produce: engine availability+versions, dependency health, asset/registry checks, git metadata. Reuse the existing desktop probe functions — single implementation.
2. Add `GET /api/v1/integrations/health` returning the integrations-health probe results (already headless in `integrations_health_data.py`); include the secret-redaction guarantee server-side.
3. Web DiagnosticsPanel: render both reports with the same status taxonomy (healthy/configured/warning/error/unconfigured), a refresh button, and copy-diagnostics-as-markdown — in both Tauri and browser modes (backend lifecycle controls stay Tauri-only).
4. Parity test: the set of probe categories rendered on desktop == categories served by the API.

## Acceptance criteria

- [ ] Browser-mode web app can answer "which engines are installed and at what version, and what's broken" identically to desktop
- [ ] One probe implementation feeding both UIs
- [ ] Secrets redacted in API responses (test with a fake env secret)

## References

- `src/launchers/launcher_diagnostics.py`, `src/launchers/integrations_health_data.py`
- `src/api/routes/core.py` (existing `/api/diagnostics`)
