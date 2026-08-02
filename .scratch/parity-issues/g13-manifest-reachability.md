# Define and test web reachability for every launcher-manifest tile (browser mode vs Tauri mode)

## Gap (PyQt6 = model)

`launcher_manifest.json` is the shared tile source of truth, and the web dashboard renders it — but tiles without a `web_route` fall back to `POST /api/launcher/launch/{tile_id}`, which spawns a Python/Qt subprocess **on the machine running the API server**. In Tauri mode that's the user's machine (fine, per ADR-0028 multi-window). In pure browser mode (remote or dev-server use) the window opens on the server — invisible to the user — or fails headlessly. The manifest schema doesn't distinguish these cases, so the web dashboard can't render honest affordances, and there is no parity test that every tile is _meaningfully_ reachable from the web app.

## Proposed fix

1. Extend the manifest tile schema with an explicit launch contract, e.g. `web: {mode: "route"|"native-window"|"unavailable", route?: string, reason?: string}` (validated by `launcher_manifest_loader.py` and the TS schema parser in `ui/src/api/schemas.ts`).
2. Web dashboard behavior:
   - `route` → in-app navigation (current behavior);
   - `native-window` → enabled only when running under Tauri/localhost (detectable today via the Tauri bridge); otherwise rendered with a "desktop app only" badge instead of a dead/launches-nowhere button;
   - `unavailable` → badge + reason.
3. Server-side guard: `POST /launcher/launch/{tile_id}` refuses native-window launches when the request is non-local (honest 409 with reason) — prevents invisible server-side Qt windows.
4. Extend `tests/config/launcher_manifest/test_parity.py`: every tile must declare its web contract; every `route` must exist in the React router (cross-check route table); CI fails on a new tile with no declaration.

## Acceptance criteria

- [ ] No tile in the web dashboard can be clicked into a no-op or a server-side window
- [ ] Manifest schema validated on both Python and TS sides
- [ ] Parity test enforces declaration + route existence for all current and future tiles

## References

- `docs/adr/0028-react-tauri-launcher-parity.md` (multi-window decision this builds on)
- `src/config/launcher_manifest.json`, `src/api/routes/launcher.py`, `ui/src/api/useLauncherManifest.ts`
