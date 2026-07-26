# Sidekick PyQt6 startup test matrix

Tracking: UpstreamDrift #8102, Tools #3936 and #3938, Tools PR #3937.

This matrix is for the classic PyQt6 launcher. React/Tauri is not the
acceptance target for this work.

## Source ownership

Sidekick, chat, and shared launcher components are owned by the Tools
repository. UpstreamDrift must consume the pinned `vendor/ud-tools` revision;
it must not repair a copied file under `src/shared/python/`.

The 2026-07-25 audit compared warning-headered UpstreamDrift Python files with
the Tools revision used for verification:

| Scope                              | Exact payload match | Payload drift | No same-relative Tools source |
| ---------------------------------- | ------------------: | ------------: | ----------------------------: |
| `chat`                             |                  13 |            26 |                             1 |
| `sidekick`                         |                  49 |           207 |                            46 |
| `gui_launcher`                     |                   1 |             5 |                             4 |
| All warning-headered shared Python |                 167 |           523 |                           158 |

These counts document pre-existing migration debt; they are not permission to
edit the child copies. The startup fix changes the canonical Tools chat source,
pins that Tools revision, and installs its import paths before the first
Sidekick import. The direct copied-chat edit found in UpstreamDrift PR #8096
was restored to `main`; its intended disconnect guidance was added to Tools PR
#3937 with regression coverage.

Existing controls are:

1. Warning headers on child copies.
2. CODEOWNERS review for `src/shared/python/`.
3. Repository hygiene tests for warning headers, approved shadows, and known
   Tools counterparts.
4. A PR-diff hygiene gate that rejects every non-deletion edit to a file that
   carried the Tools child-copy warning at the branch merge base, plus every
   newly added file that carries the warning.
5. Startup contract tests proving that pinned vendor direct-package paths
   precede both legacy alias shims and a mutable sibling Tools checkout, and
   are installed before the first Sidekick import.
6. A gitlink update to the reviewed Tools revision whenever canonical Sidekick
   behavior changes.

## Acceptance matrix

| ID           | Scenario                                         | Expected result                                                                                      | Automated evidence                                                                                                                                                  | UI evidence                                                                                                                                                                                    |
| ------------ | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SK-START-001 | Port 8000 is free                                | Launcher exports one port to API and chat                                                            | `test_configure_sidekick_runtime_exports_consistent_port_and_capability`                                                                                            | Not exercised; occupied-port path was the live target                                                                                                                                          |
| SK-START-002 | Port 8000 is occupied by an unrelated API        | Launcher selects a different free loopback port and does not attach to the stale API                 | `test_select_loopback_port_avoids_occupied_default`                                                                                                                 | **Pass** — API/Chat used 15067 and 8747 in repeat runs while port 8000 remained owned by unrelated processes                                                                                   |
| SK-START-003 | Unrelated API returns HTTP 200 from `/readyz`    | Readiness remains false unless the public instance ID matches                                        | `test_check_sidekick_api_readiness_requires_matching_instance`                                                                                                      | **Pass** — launcher ignored port 8000 and accepted only the matching dynamic child                                                                                                             |
| SK-START-004 | Launcher creates the API child                   | Child inherits a random capability and public instance ID; secrets are redacted                      | `test_generated_capability_is_nonempty_and_redacted`, `test_install_launcher_capability_copies_environment_to_app_state`                                            | N/A                                                                                                                                                                                            |
| SK-START-005 | Native Qt chat opens its WebSocket               | Socket sends a loopback Origin and encoded launcher capability                                       | Tools `test_native_websocket_contract.py`                                                                                                                           | **Pass** — status reached `Connected`; a real chat request crossed the socket                                                                                                                  |
| SK-START-006 | Chat API is still starting or unavailable        | Files and other local tabs install immediately; Chat alone reports degraded state                    | `test_deferred_install_does_not_gate_local_sidebar_on_api`                                                                                                          | **Pass** — Files browsed the worktree independently while readiness monitoring continued                                                                                                       |
| SK-START-007 | API child exits during startup                   | Launcher retries at most twice, then reports a visible degraded state                                | `test_dead_api_process_receives_bounded_restart`                                                                                                                    | Pending                                                                                                                                                                                        |
| SK-START-008 | API port environment variables conflict          | Launcher fails closed and cannot accept a stale API as its child                                     | `test_configure_sidekick_runtime_rejects_conflicting_explicit_ports`, `test_missing_runtime_contract_cannot_accept_unrelated_api`                                   | Pending                                                                                                                                                                                        |
| SK-START-009 | Readiness response is returned                   | Response exposes the public instance ID and never the launcher capability                            | `test_readyz_reports_launcher_instance_without_secret`                                                                                                              | N/A                                                                                                                                                                                            |
| SK-START-010 | Sidebar installation is requested twice          | Existing sidebar is made visible; no duplicate splitter pane is added                                | `test_sidebar_manager_install_is_idempotent`                                                                                                                        | Automated only                                                                                                                                                                                 |
| SK-START-011 | Vendored Tools and sibling Tools both exist      | Pinned vendor source wins                                                                            | `test_vendored_tools_precedes_mutable_sibling_checkout`, `test_vendored_direct_packages_precede_legacy_alias_shims`                                                 | **Pass** — default launch used the pinned chat implementation and connected on the exported dynamic port                                                                                       |
| SK-START-012 | Unexpected WebSocket disconnect                  | Status names the Sidekick API and points to `UD_CHAT_WS_URL`                                         | Tools `test_close_event_disables_reconnect`                                                                                                                         | Pending                                                                                                                                                                                        |
| SK-START-013 | Host closes after Sidekick Terminal starts       | Sidebar stops its PTY, shell, bridge, API child, and launcher; unrelated services remain             | Tools `test_host_window_close_shuts_down_live_runtime`, Upstream `test_launcher_shutdown_delegates_to_sidekick_runtime_owner`                                       | **Pass** — all nine tracked launcher descendants and ports 8747/8781 exited; both port-8000 listeners remained                                                                                 |
| SK-START-014 | User switches through every default Sidekick tab | Each tab renders without a Python crash dialog                                                       | Existing Tools sidebar/runtime tab suites                                                                                                                           | **Pass** — Chat, Files, Workspace, Terminal, Python REPL, Calculator, Data Explorer, Units, Notes, and Reporting rendered in the live sweep                                                    |
| SK-START-015 | Connected chat sends a real prompt               | Transport accepts the request; provider failures are surfaced without crashing/disconnecting         | Tools chat protocol suites                                                                                                                                          | **Transport pass / provider blocked** — request was accepted and Chat remained connected; configured Ollama/model request timed out with an actionable provider message                        |
| SK-START-016 | API child exits after Chat has connected         | Chat degrades, the launcher recreates the child within the bounded retry policy, and Chat reconnects | `test_readiness_monitor_passes_current_instance_identity`, `test_dead_api_process_receives_bounded_restart`, `test_closed_launcher_stops_sidekick_liveness_monitor` | **Pass** — Chat changed from `Connected` to `Connecting...`; API PID 48628 was replaced by PID 20084 on the same isolated port and public instance contract, then Chat returned to `Connected` |

## Computer-control procedure

1. Leave an unrelated API listening on port 8000; record its PID and do not
   terminate it.
2. Initialize `vendor/ud-tools` at the gitlink revision under test.
3. Disable onboarding for the test profile and launch
   `python launch_upstream_drift.py --classic`.
4. Confirm the main window and Sidekick pane render without a startup freeze.
5. Confirm the Files tab can browse the UpstreamDrift worktree before Chat is
   ready.
6. Confirm the Chat status transitions to `Connected` and does not show the
   unavailable/retry banner.
7. Confirm the child API listens on a port other than 8000 and `/readyz`
   returns the launcher instance ID.
8. Reconfirm the pre-existing port-8000 process is still running.
9. Stop only the launcher's verified API process tree after Chat connects.
   Confirm Chat reports the API unavailable, a new API child and dynamic
   listener appear, and Chat reconnects.
10. Close the launcher, verify its child API exits, and confirm the unrelated
    port-8000 process is untouched.
11. Record pass/fail, screenshot evidence, tested SHAs, and any new issue
    numbers in this matrix.

## 2026-07-25 computer-control result

- UpstreamDrift branch: `fix/8102-sidekick-startup`.
- Canonical Tools PR head: `2a4300cbb1b57695ae07a8375e8796c977b2939d`.
- Pinned Tools source was verified instead of either UpstreamDrift child copies
  or the mutable sibling Tools checkout.
- Chat reached `Connected` on dynamic ports while unrelated port-8000
  listeners remained active.
- `/readyz` returned `status=ready` and a public Sidekick instance ID; no
  launcher capability was displayed or written to the test record.
- Files browsing and every default Sidekick tab rendered without a Python
  crash dialog.
- A real chat prompt crossed the WebSocket. The configured AI provider later
  timed out and produced its expected actionable error; this is recorded as an
  environment/provider dependency result, not a startup transport failure.
- Opening the Terminal tab exposed a separate lifecycle bug (Tools #3938).
  After the parent Tools shutdown fix and host delegation were applied, closing
  the launcher left zero tracked descendants and zero dynamic listeners.
- A post-connect API-loss injection exposed a second recovery gap: Chat
  degraded correctly, but readiness monitoring stopped after its first success,
  so the launcher did not recreate the child. The regression was captured as
  SK-START-016. After the bounded liveness monitor was added, the retest
  replaced API PID 48628 with PID 20084 on port 35981, preserved the public
  instance identity, and returned Chat to `Connected`. Closing the host then
  removed the replacement child and both dynamic listeners while the unrelated
  port-8000 service remained active.

## Required automated command

```powershell
python -m pytest -n 0 `
  tests/unit/api/test_launcher_capability.py `
  tests/unit/api/test_routes_observability.py `
  tests/unit/api/test_websocket_origin_guard.py `
  tests/unit/api/test_chat_ws.py `
  tests/unit/launcher/test_sidekick_readiness.py `
  tests/unit/launcher/test_sidekick_runtime_startup.py `
  tests/unit/launcher/test_sidekick_startup_coordination.py -q
```

No acceptance row may be marked passed solely because the launcher says
`System Ready`; Chat connectivity and at least one local Sidekick tab must be
verified independently.
