# UpstreamDrift UI

The UpstreamDrift UI is a React 19 + TypeScript + Vite application with an optional Tauri desktop shell. It talks to the local-first Python API over REST and WebSockets, renders 3D scenes with `@react-three/fiber`, and keeps shared client state in Zustand stores.

## Local development

Start the Python API from the repository root:

```bash
python start_api_server.py --port 8000
```

In a second terminal, start the UI from `ui/`:

```bash
cd ui
npm install
npm run dev
```

The Vite dev server listens on port `5180` by default. The browser UI expects the API server to be reachable on the same machine, and the WebSocket clients in `ui/src/api/client.ts` connect through `/api/ws/simulate/{engineType}`.

## Architecture at a glance

- `ui/src/pages/` - route-level screens such as the dashboard and simulation views.
- `ui/src/components/` - reusable UI building blocks and domain components, including `visualization/Scene3D.tsx`.
- `ui/src/api/` - client utilities and hooks that wrap REST and WebSocket flows.
- `ui/src/stores/` - Zustand state for engines, simulation state, and UI session data.
- `ui/src/integration/` - integration-focused UI tests.
- `ui/src-tauri/` - Rust packaging and native shell support for the desktop build.

## API contracts the UI depends on

The UI consumes HTTP routes defined under `src/api/routes/` and relies on WebSocket flows implemented in:

- `src/api/routes/simulation_ws.py`
- `src/api/routes/chat_ws.py`

Frontend call sites and hooks live under `ui/src/api/`. For transport details, start with `ui/src/api/client.ts`, which currently connects to `/api/ws/simulate/{engineType}`. The chat service protocol lives in `src/api/routes/chat_ws.py` under `/ws/chat/{session_id}`.

## Desktop build

Install the Rust toolchain and the Tauri build prerequisites for your platform, then run:

```bash
cd ui
npm install
npm run tauri:build
```

For local desktop iteration, use `npm run tauri:dev`.

## Testing and linting

From `ui/`:

```bash
npm run test
npm run lint
```

`npm run test` uses Vitest. Integration-oriented coverage lives under `ui/src/integration/`.

## Contributing

- Repository contribution workflow: [`../CONTRIBUTING.md`](../CONTRIBUTING.md)
- Agent and repository policy: [`../CLAUDE.md`](../CLAUDE.md)
- UI parity assessment notes: [`../docs/assessments/issues/REACT_UI_PARITY_ISSUES.md`](../docs/assessments/issues/REACT_UI_PARITY_ISSUES.md)

The real ESLint configuration for this package lives in `ui/eslint.config.js`; do not copy guidance from the stock Vite scaffold.

## Known gaps

This README is a narrow developer entry point, not the full UI handbook. Active UI follow-up work is being tracked under `#3160`, including the chat UI, scaffolding pages, HelpPanel content, and model-authoring gaps called out in that track.
