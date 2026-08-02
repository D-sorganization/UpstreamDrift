# Web app has no settings/preferences surface — desktop has a 9-tab settings dialog + persisted state

## Gap (PyQt6 = model)

Desktop: `src/launchers/settings_dialog.py` (Layout, Configuration, Diagnostics, MCP Servers, Appearance, Startup, Notifications, Performance, Processes tabs), app-wide zoom, theme selection, and layout persistence to `~/.config/upstream-drift/launcher/layout.json`. Web: **no settings page at all** — theme is server-pushed (`GET /api/v1/themes/active`), sidebar/help state is session-only, and nothing persists except the launcher window registry.

## Proposed scope (deliberately a subset — not every desktop tab makes sense in a browser)

1. Add a `/settings` route with:
   - **Appearance**: theme selection (list themes via API — add `GET /api/v1/themes` + `PUT /api/v1/themes/active` if missing) and font-size/zoom preference. Coordinate with #7423 (theming tokens) — this issue is the _settings UI + persistence_, #7423 is the token infrastructure.
   - **Notifications**: toast duration/verbosity (pairs with #7428 toast hardening).
   - **Simulation defaults**: default engine, default duration/timestep (mirrors desktop Configuration tab; pairs with #7424 persistence fix).
   - **Diagnostics**: link to the diagnostics panel (see diagnostics parity issue).
2. Persistence: server-side per-profile settings endpoint (`GET/PUT /api/v1/settings`) stored next to other user config (`~/.upstreamdrift/`), so settings survive browser storage clears and are shared between Tauri and browser modes. localStorage only as cache.
3. Desktop-only tabs (MCP Servers, Processes, Docker/startup) → record as `exempt` in the feature-parity registry with reasons (see exemptions issue) unless/until the web app gains those capabilities.

## Acceptance criteria

- [ ] `/settings` route with the four sections above, persisted server-side
- [ ] Theme choice round-trips and applies without reload
- [ ] Settings keys documented and shared schema validated (Pydantic model + generated TS types)
- [ ] Registry updated: settings.appearance/notifications/sim-defaults = parity; desktop-only tabs = exempt

## References

- `src/launchers/settings_dialog.py`, `src/launchers/app_zoom.py`, `src/launchers/launcher_theme.py`
- UI/UX epic #7444 issues #7423/#7424/#7428 — adjacent, not duplicates
