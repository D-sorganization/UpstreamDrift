# Sidekick PyQt6 startup test matrix

Tracking: UpstreamDrift PyQt6 QA epic #8062, original Sidekick defect #8075,
startup/recovery issue #8102, Tools #3936 and #3938, and merged Tools PR #3937
(`4744422d3`).

This matrix is for the classic PyQt6 launcher. React/Tauri is not the
acceptance target for this work.

## Source ownership

Sidekick, chat, and shared launcher components are owned by the Tools
repository. Deployed UpstreamDrift must consume the pinned `vendor/ud-tools`
revision; an explicitly configured development checkout must be a valid Tools
root and remains authoritative. UpstreamDrift must not repair a copied file
under `src/shared/python/`.

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

The follow-up exact-pin audit initially classified the Sidekick/chat portion
more precisely: 233 drifting Python files had a same-relative canonical file,
while 47 files had no parent source. Contributing the standalone, persistence,
and WebSocket protocol surfaces to Tools and removing those downstream copies
reduced the active exception inventory to 33 UpstreamDrift production
extensions. The previously packaged force-plate test now lives under `tests/`.
Parent-backed files are frozen migration candidates: they cannot receive another
UpstreamDrift implementation edit and must be deleted in usage-audited batches.
The remaining exceptions require file-level ownership metadata rather than an
assumed Tools owner.

Existing controls are:

1. Warning headers on child copies.
2. CODEOWNERS review for `src/shared/python/`.
3. Repository hygiene tests for warning headers, approved shadows, and known
   Tools counterparts.
4. A PR-diff hygiene gate that rejects every non-deletion edit to any Python
   file with a same-relative path in the exact pinned Tools gitlink, regardless
   of warning header or top-level shadow allow-list. It also retains the
   base/current warning-header checks. Protected CI fetches the PR base, then
   sparse-checks out only `src/shared/python` from the exact gitlink revision;
   missing authoritative inventory fails closed. A direct base comparison is
   used conservatively when shallow history cannot supply a merge base.
5. Startup contract tests proving that pinned vendor direct-package paths
   precede both legacy alias shims and a mutable sibling Tools checkout, while
   a valid `TOOLS_REPO_PATH` precedes every fallback and an invalid explicit
   root fails closed. A direct import probe resolved
   `sidekick.ui.tools_sidebar` from canonical Tools commit `9d95f7c2b`, not
   from the UpstreamDrift child tree or vendor fallback.
6. A gitlink update to the reviewed Tools revision whenever canonical Sidekick
   behavior changes.
7. The repository shadow-module gate is evaluated against the exact pinned
   Tools tree. The new direct-package surface exposed 12 pre-existing
   UpstreamDrift shadows (`calc_backend`, `compatibility.py`, `config`,
   `deprecation.py`, `file_watcher`, `logging_pkg`, `programmatic_pid`,
   `reporting`, `rotation_transforms`, `safe_pandas_eval.py`, `scripting`, and
   `sidekick`). They are classified under migration issue #5623 with an
   enforced 2026-12-31 sunset instead of being silently accepted. Sidekick
   runtime imports still resolve from the pinned Tools direct package first.
8. The documented focused regression command runs with
   `--tools-mode vendored`, so its imports exercise the same canonical source
   precedence as the production launcher instead of silently testing stale
   child copies.
9. `scripts/config/shared_python_ownership_exceptions.yaml` records every
   warning-headered Sidekick/chat file without a same-relative Tools source.
   The hygiene test requires an owner state, rationale, tracking issue, and
   unexpired review date for all 33 paths; missing, stale, extra, or newly
   canonicalized exceptions fail the gate.
10. Wheel builds verify that `vendor/ud-tools` is checked out at the exact
    superproject gitlink and package the parent-owned `shared` graph,
    `chat`/`sidekick` compatibility shims, `utils`, and DbC contracts. Only
    non-conflicting Upstream-owned Chat/Sidekick extensions supplement that
    graph. Tools' canonical alias finder coalesces `chat.*`,
    `shared.python.chat.*`, and `src.shared.python.chat.*` (and the equivalent
    Sidekick spellings) to one module identity, so an installed application
    cannot silently execute a stale nested child copy.
11. The tagged Python release checks out the exact Tools submodule and builds
    the wheel directly from that verified checkout. It deliberately does not
    publish an unverifiable wheel rebuilt from an unpacked source archive.
12. Source-mode startup uses a manifest-gated exact-module finder for the 33
    retained production extensions. It never widens canonical package paths;
    unclassified, unresolved, preloaded-child, or newly conflicting modules
    fail closed. Isolated regression coverage proves canonical standalone and
    persistence modules resolve from Tools while `engine_core`,
    `motion_matching`, the API route, and approved biomechanics extensions
    continue to resolve from UpstreamDrift.

## Acceptance matrix

| ID           | Scenario                                         | Expected result                                                                                                                 | Automated evidence                                                                                                                                                  | UI evidence                                                                                                                                                                                             |
| ------------ | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SK-START-001 | Port 8000 is free                                | Launcher exports one port to API and chat                                                                                       | `test_configure_sidekick_runtime_exports_consistent_port_and_capability`                                                                                            | Not exercised; occupied-port path was the live target                                                                                                                                                   |
| SK-START-002 | Port 8000 is occupied by an unrelated API        | Launcher selects a different free loopback port and does not attach to the stale API                                            | `test_select_loopback_port_avoids_occupied_default`                                                                                                                 | **Pass** — API/Chat used 15067 and 8747 in repeat runs while port 8000 remained owned by unrelated processes                                                                                            |
| SK-START-003 | Unrelated API returns HTTP 200 from `/readyz`    | Readiness remains false unless the public instance ID matches                                                                   | `test_check_sidekick_api_readiness_requires_matching_instance`                                                                                                      | **Pass** — launcher ignored port 8000 and accepted only the matching dynamic child                                                                                                                      |
| SK-START-004 | Launcher creates the API child                   | Child inherits a random capability and public instance ID; secrets are redacted                                                 | `test_generated_capability_is_nonempty_and_redacted`, `test_install_launcher_capability_copies_environment_to_app_state`                                            | N/A                                                                                                                                                                                                     |
| SK-START-005 | Native Qt chat opens its WebSocket               | Socket sends a loopback Origin and encoded launcher capability                                                                  | Tools `test_native_websocket_contract.py`                                                                                                                           | **Pass** — status reached `Connected`; a real chat request crossed the socket                                                                                                                           |
| SK-START-006 | Chat API is still starting or unavailable        | Files and other local tabs install immediately; Chat alone reports degraded state                                               | `test_deferred_install_does_not_gate_local_sidebar_on_api`, `test_delayed_readiness_rechecks_running_child_without_relaunch`                                        | **Pass** — Files browsed the worktree independently while readiness monitoring continued                                                                                                                |
| SK-START-007 | API child exits or cannot be launched            | Launcher retries at most twice, then reports a visible degraded state                                                           | `test_dead_api_process_receives_bounded_restart`, `test_child_launch_failure_exhausts_retry_budget_observably`                                                      | Automated pass; runtime post-connect child replacement is exercised separately in SK-START-016                                                                                                          |
| SK-START-008 | API port environment variables conflict          | Launcher fails closed and cannot accept a stale API as its child                                                                | `test_configure_sidekick_runtime_rejects_conflicting_explicit_ports`, `test_missing_runtime_contract_cannot_accept_unrelated_api`                                   | **Pass** — with `API_PORT=8123` and `GOLF_API_PORT=9123`, startup logged the explicit conflict, opened no listener on either port, and kept Chat visibly degraded instead of accepting an unrelated API |
| SK-START-009 | Readiness response is returned                   | Response exposes the public instance ID and never the launcher capability                                                       | `test_readyz_reports_launcher_instance_without_secret`                                                                                                              | N/A                                                                                                                                                                                                     |
| SK-START-010 | Sidebar installation is requested twice          | Existing sidebar is made visible; no duplicate splitter pane is added                                                           | `test_sidebar_manager_install_is_idempotent`                                                                                                                        | Automated only                                                                                                                                                                                          |
| SK-START-011 | Vendored Tools and sibling Tools both exist      | Pinned vendor source wins                                                                                                       | `test_vendored_tools_precedes_mutable_sibling_checkout`, `test_vendored_direct_packages_precede_legacy_alias_shims`                                                 | **Pass** — default launch used the pinned chat implementation and connected on the exported dynamic port                                                                                                |
| SK-START-012 | Unexpected WebSocket disconnect                  | Status names the Sidekick API and points to `UD_CHAT_WS_URL`                                                                    | Tools `test_close_event_disables_reconnect`                                                                                                                         | **Pass** — after the verified API pair stopped on port 22732, Chat immediately displayed the exact Sidekick/`UD_CHAT_WS_URL` guidance and returned to `Connected` within two seconds                    |
| SK-START-013 | Host closes after Sidekick Terminal starts       | Sidebar stops its PTY, shell, bridge, API child, and launcher; unrelated services remain                                        | Tools `test_host_window_close_shuts_down_live_runtime`, Upstream `test_launcher_shutdown_delegates_to_sidekick_runtime_owner`                                       | **Pass** — all nine tracked launcher descendants and ports 8747/8781 exited; both port-8000 listeners remained                                                                                          |
| SK-START-014 | User switches through every default Sidekick tab | Each tab renders without a Python crash dialog                                                                                  | Existing Tools sidebar/runtime tab suites                                                                                                                           | **Pass** — Chat, Files, Workspace, Terminal, Python REPL, Calculator, Data Explorer, Units, Notes, and Reporting rendered in the live sweep                                                             |
| SK-START-015 | Connected chat sends a real prompt               | Transport accepts the request; provider failures are surfaced without crashing/disconnecting                                    | Tools chat protocol suites                                                                                                                                          | **Transport pass / provider blocked** — request was accepted and Chat remained connected; configured Ollama/model request timed out with an actionable provider message                                 |
| SK-START-016 | API child exits after Chat has connected         | Chat degrades, the launcher recreates the child within the bounded retry policy, and Chat reconnects                            | `test_readiness_monitor_passes_current_instance_identity`, `test_dead_api_process_receives_bounded_restart`, `test_closed_launcher_stops_sidekick_liveness_monitor` | **Pass** — repeat run changed Chat from `Connected` to `Connecting...`; API PID 45416 was replaced by PID 31088 on isolated port 57765, then Chat returned to `Connected` in under five seconds         |
| SK-START-017 | Built wheel is installed outside the checkout    | Direct, `shared.python`, and legacy `src.shared.python` spellings share parent-owned module identities; standalone CLI executes | Import-alias architecture tests, build-hook and console-script packaging suites                                                                                     | **Final-pin source pass / artifact retest pending** — the protected Tools merge is pinned exactly and the 140-test vendored source gate passes; rebuild and clean-install smoke remain required         |

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
- Initializing the new Tools gitlink made the broader repository shadow gate
  detect 12 previously unclassified overlaps, including `sidekick`. The
  migration allow-list now records each under #5623 with a mandatory sunset;
  the guard passes against the exact pinned parent tree and will fail again if
  another unclassified shadow appears.

## 2026-07-26 focused fault-injection result

- Repeated the post-connect API-loss case on isolated port 57765. The verified
  API grandchild PID 45416 exited, Chat degraded, and the bounded liveness
  monitor created replacement PID 31088 on the same runtime contract. Chat
  returned to `Connected` in under five seconds.
- Launched the classic PyQt6 application with conflicting explicit
  `API_PORT=8123` and `GOLF_API_PORT=9123`. Startup logged the contract error,
  and Chat displayed the exact degraded guidance:

  > `Sidekick API unavailable — retrying in 3s. Set UD_CHAT_WS_URL if the local API is external.`

  No process listened on 8123 or 9123.

- Launched UpstreamDrift commit `c0ab9cfaf` against canonical Tools commit
  `9d95f7c2b` and waited for Chat to reach `Connected` on isolated port 22732.
  After verifying and stopping API PID 33316 and its worker PID 11300, Chat
  immediately displayed the exact guidance above. The launcher created
  replacement API PID 30244 and Chat returned to `Connected` within two
  seconds.
- Closed each isolated launcher through its window and verified that no tested
  launcher process or dynamic listener remained.
- Built and installed `upstream_drift-2.1.1-py3-none-any.whl` in clean
  temporary virtual environments. Successive probes exposed stale
  nested-package precedence, a missing canonical `chat_contracts` dependency,
  and split module identities between direct and `src.shared` imports. The
  parent-owned Tools alias graph now supplies the dependency closure and
  coalesces all three supported spellings; clean-install identity and
  byte-fidelity probes pass. The stronger `python -m sidekick` probe then found
  a missing `runpy` loader method in Tools.
- Rebuilt the candidate wheel after adding the `runpy` fix. The full
  clean-environment smoke test then exposed one more installed-only fault:
  Python could not attach the intermediate `src.shared` namespace even though
  the canonical leaf modules were present. A RED architecture test reproduced
  the fault in Tools; the parent alias installer now binds both intermediate
  namespaces. All nine alias-architecture tests and Ruff pass.
- Rebuilt UpstreamDrift candidate `e56b9cdd7` against the exact Tools source
  tree published as PR #3937 head `7396c2071`. Installation outside the source
  checkout produced one module identity for direct, `shared.python`, and
  `src.shared.python` Chat and Sidekick imports, loaded the canonical dependency
  closure, and completed `python -m sidekick --help`. This row remains pending
  only for the protected Tools merge, final gitlink repin, and a no-substitution
  rerun from that merged revision.
- Repeated the artifact probe in a fresh dependency-isolated virtual
  environment using the declared `gui-tools` extra. The complete clean install
  and PyQt6 import/CLI probe passed in 943.11 seconds. The GUI extra is part of
  this gate by contract; a core-only wheel install is not expected to provide a
  Qt binding.

## 2026-07-27 protected Tools merge and exact repin

- Tools PR #3937 merged through the protected workflow as
  `4744422d39aea03f5b6f59c8908f5e79ce246d92`.
- The scheduled UpstreamDrift `main` sync and local parent-ownership migration
  shared no changed files; a normal merge preserved both histories without
  rewriting either commit.
- `vendor/ud-tools` now points at the exact protected Tools merge. The focused
  140-test vendored source/startup/security suite passes at this pin.
- The installed-wheel smoke and final computer-controlled classic PyQt6
  acceptance remain open gates; this entry must be updated with their results
  before #8102 is marked complete.

## Required automated command

```powershell
python -m pytest -n 0 --tools-mode vendored `
  tests/unit/api/test_launcher_capability.py `
  tests/unit/api/test_routes_observability.py `
  tests/unit/api/test_websocket_origin_guard.py `
  tests/unit/api/test_chat_ws.py `
  tests/unit/launcher/test_sidekick_readiness.py `
  tests/unit/launcher/test_sidekick_runtime_startup.py `
  tests/unit/launcher/test_sidekick_startup_coordination.py `
  tests/unit/test_launch_upstream_drift_bootstrap.py `
  tests/unit/test_build_hooks.py `
  tests/unit/test_src_package_bootstrap.py `
  tests/unit/packaging/test_sidekick_console_script.py `
  tests/unit/repo_hygiene/test_tools_child_copy_contract.py `
  tests/unit/repo_hygiene/test_no_shadow_of_tools_shared.py `
  tests/ci/test_ci_infrastructure.py::TestCIEnvironmentCompatibility::test_unit_gate_fetches_pr_base_before_child_copy_guard `
  tests/ci/test_ci_infrastructure.py::TestCIEnvironmentCompatibility::test_unit_gate_sparse_checks_out_pinned_tools_for_ownership_guard -q
```

The installed-artifact contract is a separate release gate because importing
from a source checkout cannot detect package-layout split brain:

```powershell
$env:SKIP_UI_BUILD = "1" # Python-only QA build; tagged releases must include UI
Remove-Item -Recurse -Force -LiteralPath dist -ErrorAction SilentlyContinue
python -m build --wheel --no-isolation
$wheels = @(Get-ChildItem -LiteralPath dist -Filter "upstream_drift-*.whl" -File)
if ($wheels.Count -ne 1) {
  throw "Expected exactly one newly built UpstreamDrift wheel; found $($wheels.Count)"
}
$env:UPSTREAM_DRIFT_WHEEL = $wheels[0].FullName
python -m pytest -n 0 `
  tests/smoke/python_wheel/test_wheel_artifact.py::test_sidekick_uses_one_parent_owned_alias_graph -vv
```

No acceptance row may be marked passed solely because the launcher says
`System Ready`; Chat connectivity and at least one local Sidekick tab must be
verified independently.
