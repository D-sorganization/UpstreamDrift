# Unified `pydantic-settings` migration (issue #6565)

This tracks the incremental migration of scattered `os.getenv` /
`os.environ.get` reads onto the single canonical typed settings class
`src.shared.python.config.typed_settings.Settings`.

## Foundation (this PR)

- Added `pydantic-settings` to core `pyproject.toml` dependencies.
- Created `src/shared/python/config/typed_settings.py` with a `Settings`
  (`pydantic_settings.BaseSettings`) class. Each field reads the **same env
  var name** with the **same default** as the legacy accessor it replaces,
  via `validation_alias`. `get_settings()` returns a **fresh** instance per
  call (no process caching) so runtime `os.environ` mutations are observed —
  matching the legacy functional accessors in `config.environment`.
- Validators added only where clearly safe and value-preserving (port range
  `1..65535`, already enforced by the legacy `get_server_port`).

## Proof-of-concept slice migrated

`src/api/config.py` (API server cluster):

| Env var         | Accessor              | Default                       |
| --------------- | --------------------- | ----------------------------- |
| `API_HOST`      | `get_server_host()`   | `127.0.0.1`                   |
| `API_PORT`      | `get_server_port()`   | `8000` (validated `1..65535`) |
| `ALLOWED_HOSTS` | `get_allowed_hosts()` | documented default list       |
| `CORS_ORIGINS`  | `get_cors_origins()`  | documented default list       |

Public function signatures and behavior (including the `ValueError` raised by
`get_server_port` with the exact `Invalid API_PORT value: <raw>` message) are
unchanged; they now delegate to `Settings` internally.

## Known env-var / default divergence (do NOT silently consolidate)

`src/api/config.py` reads the **legacy** `API_HOST` / `API_PORT`, while
`src/shared/python/config/environment.py` (`get_api_host` / `get_api_port`)
reads the **canonical** `GOLF_API_HOST` / `GOLF_API_PORT`. Both default to
`127.0.0.1` / `8000`, but they are distinct variables. This pre-existing
divergence (issue #2068) is **preserved exactly** — retiring the legacy names
requires a design decision and is out of scope here.

## Remaining subsystems to migrate (follow-up PRs)

~125 `os.getenv` / `os.environ.get` sites remain across ~85 files. Suggested
cohesive clusters, each its own PR, migrating onto `Settings`:

- [ ] `config/environment.py` core accessors (secret key, DB URL/pool, env,
      log level, API/realtime host+port, golf port/mode, headless/display).
      This is the largest cluster and the natural backbone — migrate the
      functional accessors to delegate to `Settings` fields.
- [ ] `realtime/` cluster — `ws_pubsub.py`, `transport_file.py`,
      `file_pubsub.py`, `api.py` (`UD_REALTIME_BACKEND`, realtime host/port).
- [ ] `api/` cluster — `database.py`, `auth/security.py`, `rate_limit.py`,
      `debug_guard.py`, `local_server.py`, `task_manager_durable.py`,
      `cors.py`, routes (`chat_ws.py`, `data_explorer.py`, `physics.py`,
      `video.py`).
- [ ] `shared/python/ai/` cluster — adapters (`anthropic_adapter.py`,
      `openai_adapter.py`, `bitnet_adapter.py`), integrations (`notion.py`,
      `linear.py`, `obsidian.py`, `affine.py`, `github_mcp/*`), `mcp/*`,
      `memory_manager.py`, provider config widgets, credentials.
- [ ] `launchers/` cluster — `upstream_drift_launcher.py`,
      `settings_dialog.py`, `launcher_process_manager.py`, `docker_manager.py`,
      `launcher_diagnostics.py`, `launcher_constants.py`,
      `external_tools_adapter.py`, `exercise_dashboard.py`,
      `integrations_health_data.py`.
- [ ] `model_generation/` cluster — `library/repository.py`,
      `library/github_importer.py`, converters, `api/rest_api.py`,
      `model_registry.py`, `model_source_providers.py`.
- [ ] `engines/` cluster — simscape adapter/pool, drake/mujoco GUI + sim,
      opensim muscle analysis, pose_studio engine controller.
- [ ] `security/` — `env_validator.py` (`_assert_production_secrets`),
      `secure_subprocess.py`.
- [ ] Misc — `docker_config.py`, `cors.py`, `body_part_viz/asset_library.py`,
      `motion_pipeline/matching/*`, `motion_matching/leaderboard.py`,
      `pendulum_simulator/native_backend.py`, `engine_core/engine_availability.py`,
      `pose_estimation/openpose_estimator.py`, sidekick process calculators.

When migrating each cluster: keep public function signatures/behavior
identical (delegate to `Settings`), never rename an env var or change a
default, and add tests pinning the env-var/default contract.
