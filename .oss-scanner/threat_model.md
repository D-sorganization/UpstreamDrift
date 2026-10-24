# Threat Model

UpstreamDrift is a research platform for golf-swing biomechanics. It loads
motion-capture recordings and multibody models, runs physics engines, and
serves the results through a FastAPI backend to a React/Tauri desktop UI and
a PyQt6 launcher. Paths below are relative to the repository root.

## What This Project Does and Where Untrusted Input Enters

The API server has two deployment modes:

- **Cloud/server mode.** `src/api/server.py` binds `127.0.0.1` by default
  (`src/shared/python/config/typed_settings.py`). It adds
  `TrustedHostMiddleware`, a CORS allowlist that refuses `*`, and slowapi rate
  limits (`src/api/rate_limit.py`). It requires a JWT bearer token (HS256,
  `src/api/auth/security.py`) on every auto-registered router
  (`src/api/route_registry.py`). Treat every request as untrusted.
- **Local/desktop mode.** `src/api/local_server.py` is what the Tauri shell
  starts. It binds `127.0.0.1` and sets `GOLF_SUITE_MODE=local`, which
  disables HTTP auth on purpose (`GOLF_AUTH_DISABLED`;
  `src/shared/python/config/environment.py`).
  - Mutating launcher routes are guarded only by the
    `X-Launcher-CSRF-Token` and Origin check (`_enforce_launcher_mutation_guard`).
  - WebSockets need a loopback Origin plus the launcher token
    (`src/api/auth/ws_auth.py`).
  - The attacker here is a malicious web page in the user's browser: DNS
    rebinding, cross-site "simple" POSTs and multipart uploads, WebSocket
    hijacking. A malicious data file the user opens is the other attacker.

Untrusted input enters through:

- **File uploads (`UploadFile`).** Sizes are capped by
  `src/api/middleware/upload_limits.py`.
  - `src/api/routes/motion_capture.py` (C3D)
  - `src/api/routes/data_explorer.py`
  - `src/api/routes/video.py`
  - `src/api/routes/necromatcher.py`
  - `src/api/routes/launch_monitor_analytics_sessions.py`
  - `src/shared/python/motion_pipeline/api.py`
- **Filesystem paths taken from request bodies or query strings.**
  - `src/api/routes/opencap.py` (`session_dir`, only an `is_dir()` check)
  - `src/api/routes/necromatcher.py` (`source_path` on `/swings/{id}/models` and
    `/swings/{id}/profiles`, then video and zip exports)
  - `src/api/routes/motion_capture.py` (`matching_model_urdf`)
  - Contained by allowlists, but worth confirming:
    - `src/api/routes/model_explorer.py` (`resolve_contained_path`)
    - `src/api/routes/dataset.py` (input and output roots)
    - `src/api/routes/launcher.py` (`resolve_file_within`)
    - `src/api/routes/recordings.py` (`validate_recording_id`)
- **WebSockets.**
  - `src/api/routes/chat_ws.py`
  - `src/api/routes/simulation_ws.py`
  - `src/api/routes/realtime.py`
  - `src/shared/python/chat/router_factory.py`
  - `src/shared/python/realtime/ws_pubsub.py`
- **Standalone services.**
  - `src/shared/python/motion_pipeline/api.py`: its `__main__` binds
    `0.0.0.0` with no auth, and accepts uploads plus a URDF path.
  - `src/shared/python/calc_backend/app.py`: no auth.
    `routers/symbolic_solver.py` passes request strings to sympy `parse_expr`
    (eval-based). The only guard is `validate_expression` from `safe_eval`,
    which resolves from the pinned Tools tree in `vendor/ud-tools`.
  - `src/shared/python/model_generation/api/rest_api_core.py`: open when
    `MODEL_GEN_API_KEY` is unset.
- **Model and motion-capture parsers** (malicious file opened by the user or
  uploaded).
  - The Rust C3D/TRC/BVH reader is `rust_core/upstream-mocap-io/src/`
    (`c3d.rs`, `trc.rs`, `bvh.rs`, PyO3 `bindings.rs`). The C3D header drives
    allocation sizes, e.g. `n_frames * n_markers * 3`.
  - The ezc3d fallback goes through the C3D adapter.
  - URDF, MJCF, SDF and OSIM XML parsing is meant to use `defusedxml`. Any
    parse through stdlib `xml.etree` (for example `src/tools/model_explorer/`)
    is in scope.
  - The only Rust `unsafe` block is `rust_core/upstream-pinocchio-id/src/bindings.rs`.
- **Archives and deserialization.**
  - `src/shared/python/training/portable.py`: `extractall` after a
    manifest-only checksum.
  - `src/shared/python/myofullbody/assets.py`: tar.
  - `src/shared/python/model_generation/library/repository.py`: `_safe_extract_zip`.
  - `src/shared/python/security/security_utils.py`.
  - `np.load(allow_pickle=True)` in
    `src/shared/python/motion_matching/parity_report.py` and
    `src/shared/python/motion_matching/visualization/simulation_viewer.py`.
- **Outbound fetches (SSRF).**
  - `src/shared/python/model_generation/library/model_library.py`
    (`urlretrieve(entry.source_url)`)
  - `src/shared/python/model_generation/library/repository.py` (HTTPS host
    allowlist)
  - `src/shared/python/motion_pipeline/sources/opencap_download.py`
- **LLM agent tools.** Prompt injection can come from chat input or from
  content the agent reads. The registry is built in
  `src/api/services/chat_service.py` from `src/shared/python/ai/sample_tools.py`
  and includes:
  - `shell_execute` (`src/shared/python/ai/tools/cli_tools.py`): an allowlist
    of ls, cat, grep, find and similar, with no path restriction.
  - `claude_ask` and `codex_generate`, which run a CLI with an LLM-chosen
    prompt.
  - The agent-control tools in `src/shared/python/ai/tools/agent_control.py`.
  - `file_ops.py` (`read_file`, `list_directory`), which the PyQt assistant
    panel registers.
- **Frontend.** `ui/` is React, Vite and Tauri 2. Its CSP is in
  `ui/src-tauri/tauri.conf.json`, and IPC is limited by
  `ui/src-tauri/capabilities/main.json`. The Tauri shell spawns the backend
  with a fixed argv (`ui/src-tauri/src/lib.rs`).

Trusted: the repository's own config under `src/config/`, maintainer-authored
model files under `shared/models/`, and the local user's own shell.

## Components That Matter Most / Least

Most important:

1. Local-mode API exposure to a browser attacker: missing auth, Host checks
   and CSRF coverage on upload and path routes.
2. Path handling in `necromatcher.py`, `opencap.py` and `motion_capture.py`,
   and any read or write outside the intended data roots.
3. The LLM tool surface (`cli_tools.py`, `agent_control.py`, `file_ops.py`):
   file disclosure or code execution through prompt injection.
4. The `calc_backend` symbolic solver (`parse_expr`) and anything else that
   evaluates request strings.
5. The Rust mocap parsers in `upstream-mocap-io` and their PyO3 boundary:
   crashes, unbounded allocation, out-of-bounds reads.
6. Archive extraction and pickle-enabled loads reachable from imported files.
7. JWT and API-key handling in `src/api/auth/` and the WebSocket auth in
   `ws_auth.py`.

Less important but in scope: the standalone `motion_pipeline` and
`model_generation` services (developer tools, not shipped as defaults), the
PyQt6 desktop tools under `src/tools/` and `src/launchers/`, and the
remaining `rust_core/` crates.

Out of scope:

- Vendored third-party code and submodules: `vendor/` (including
  `vendor/ud-tools`, which belongs to the D-sorganization/Tools repository and
  is scanned there), `shared/models/opensim/opensim-models`,
  `shared/models/myosuite/myo_sim` and `src/shared/tools/human-gazebo`.
- Data and assets: `data/`, `assets/`, `candidates/`, `evidence/`, `output/`,
  `reports/`, `knowledge/`, `notebooks/`. Use the `.c3d` files in `data/` as
  seed inputs only.
- Generated files: `ui/dist`, `ui/src-tauri/gen`, `schemas/`, `stubs/`,
  `src/api/migrations/`.
- MATLAB/Simscape models under `src/engines/Simscape_Multibody_Models/`.
- Documentation (`docs/` and `evidence/`, both left out of the scan image,
  and `manuals/`), CI workflows (`.github/`), and agent
  configuration (`.claude/`, `.gaai/`, `.agents/`, `.jules/`, and similar).

## How to Exercise It

The image is built from `.oss-scanner/Dockerfile`.

- **What is installed.** The image has:
  - the checkout at `/src`
  - the pinned Tools tree in `/src/vendor/ud-tools`
  - a venv at `/opt/venv` with `requirements-dev.lock`, which includes
    FastAPI, uvicorn, pytest, hypothesis, ezc3d, mujoco and PyQt6
  - the Rust toolchain and crate dependencies
  - the `upstream_mocap_io` and `upstream_physics` wheels
- **What is not installed.** Drake, Pinocchio, OpenSim, MyoSuite, MATLAB,
  PyTorch and the RL stack are left out to keep the image buildable. Code
  that needs them skips or degrades. Do not report a missing optional engine
  as a finding.
- **The network is off.** Install nothing.

To exercise it:

- **API tests:** `python -m pytest tests/api -q --no-cov -p no:cacheprovider`.
  `tests/api/test_auth_security.py` covers auth. Use `fastapi.testclient`
  against `src.api.server:app` (cloud mode) or
  `src.api.local_server.create_local_app()` (set `GOLF_SUITE_MODE=local`) to
  drive routes in-process.
- **Run a server:**
  `python -m uvicorn --factory src.api.local_server:create_local_app --host 127.0.0.1 --port 8001`.
  It is headless; never launch the GUI.
- **Mocap tests:** `python -m pytest tests/motion_capture -q --no-cov`.
- **Rust parser:** `cargo test --offline -p upstream-mocap-io` (fixtures are in
  `rust_core/upstream-mocap-io/tests/`). From Python, call
  `upstream_mocap_io` on crafted C3D, TRC or BVH bytes. Seed inputs are
  `data/C3D_TA_Driver.c3d`, `data/C3D_TA_Iron.c3d` and `data/cmu_mocap/`.
- **Headless environment:** `QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg` and
  `MUJOCO_GL=egl` are preset.
- **Default test filters:** `pyproject.toml` `[tool.pytest.ini_options]`
  defines the markers that are skipped by default.

## How You Rate Severity

- **Critical.** Code execution reachable from the network or from a browser
  page in local mode. Examples: an eval or `parse_expr` bypass, pickle or
  unsafe deserialization of request data, a path write that lands an
  executable or config. Also authentication bypass in cloud/server mode.
- **High.**
  - Arbitrary file read or write, or path traversal, through an API route or
    an archive.
  - Cross-site or DNS-rebinding access to the local API that reads data or
    triggers state changes.
  - LLM tool abuse that discloses files or runs commands.
  - A memory-safety bug in the Rust parsers or PyO3 bindings (out-of-bounds
    access, use-after-free) reachable from a crafted mocap file.
  - SSRF to internal addresses.
  - JWT or API-key forgery.
- **Medium.**
  - Denial of service: unbounded allocation or CPU from a crafted file or
    request, upload-limit bypass, a parser panic that kills the server.
  - Information leaks through error messages or debug endpoints when the
    environment is not production.
  - Missing rate limiting on auth.
- **Low.** Issues that need local shell access or a maintainer-controlled
  config file, hardening gaps with no demonstrated impact, and crashes in
  developer-only tools.

Raise severity by one level when the issue is reachable in the default local
desktop configuration with no user action beyond visiting a web page. Lower it
by one level when it needs the user to import a file they would not normally
open.

## Anything to Leave Alone

- Auth being disabled in local mode is a documented design choice
  (`src/api/local_server.py`). Report what an attacker can actually do across
  that boundary (browser, rebinding, CSRF), not the flag itself.
- Debug endpoints that `src/api/debug_guard.py` enables outside production are
  expected on developer machines; report them only if they are reachable in
  production mode.
- Physics and numerical accuracy, NaN propagation, solver convergence and
  tolerance choices are correctness work tracked elsewhere, not security
  findings.
- Do not report CVEs in pinned dependencies by version alone. Dependabot,
  pip-audit (`scripts/config/pip_audit_waivers.json`) and Trivy already cover
  them. Report one only with a reachable exploit path in this code.
- Missing optional engines, GPU or display, and MATLAB in the scan image are
  expected.
- Never launch GUI entry points (PyQt6 launchers, `launch_golf_suite.py`,
  `launch_upstream_drift.py`). Exercise GUI code only through offscreen tests.
