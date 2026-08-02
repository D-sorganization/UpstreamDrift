# Decide and document desktop-only exemptions in the feature-parity registry (pose editing, document library, Docker, MCP config, sidekick terminal/REPL/Jupyter)

## Problem

Several large PyQt6 features have no web counterpart and probably should not get one under the current architecture — but today that's implicit, so every parity review re-litigates them and contributors can't tell deliberate scope from accidental drift. ADR-0028 set this precedent for tabs/docks (multi-window decision); the remaining features need the same treatment, recorded in the feature-parity registry (registry issue in this epic) as `exempt` with reasons, or converted into roadmap issues.

## Features to decide (each: exempt-with-reason OR roadmap issue)

| Feature                                                                | Desktop location                                                                      | Considerations                                                                                                                          |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Interactive pose editing / keyframes / Pose Studio                     | `src/tools/pose_studio/`, MuJoCo GUI pose tabs                                        | Heavy native-viewer coupling; a web version is a major project. Likely roadmap-later.                                                   |
| Document library (PDF/LaTeX indexing + viewer)                         | `src/launchers/library_widget.py`                                                     | SQLite + local files; could be a thin web reader later.                                                                                 |
| Docker environment management                                          | `src/launchers/docker_manager.py`, `docker_dialog.py`                                 | Local-machine concern; exposing Docker control over HTTP is a security risk. Likely exempt (Tauri-mode-only candidate).                 |
| MCP servers preferences                                                | `src/launchers/mcp_servers_preferences.py`                                            | Desktop assistant config. Likely exempt.                                                                                                |
| Sidekick OS terminal / Python REPL / Workspace / Jupyter / Skills tabs | `src/launchers/launcher_sidekick_sidebar.py`                                          | Remote shell from browser = non-starter under the local-server trust model. Exempt (see chat-context issue for what web chat DOES get). |
| Embedded tool tabs/docks/pop-out re-docking                            | `src/launchers/embedded_host.py`                                                      | Already decided: ADR-0028 multi-window. Registry entry should cite the ADR.                                                             |
| Exercise dashboard + per-engine dashboards                             | `src/launchers/exercise_dashboard.py`, `{mujoco,drake,pinocchio,jaxsim}_dashboard.py` | Web Simulation page partially covers this; decide whether exercise routing is a web feature.                                            |
| MATLAB/Simscape suite dialog                                           | `src/launchers/matlab_suite_dialog.py`                                                | Requires local MATLAB; probably exempt for browser, possible in Tauri mode.                                                             |

## Acceptance criteria

- [ ] Each row has a registry entry: `exempt` + one-sentence reason (+ ADR reference where applicable) or a linked roadmap issue
- [ ] A short ADR (or amendment to ADR-0028) capturing the security rationale for the exempt-by-trust-model items (terminal/Docker)
- [ ] Future parity audits can diff against the registry instead of rediscovering these

## References

- `docs/adr/0028-react-tauri-launcher-parity.md` (the pattern to follow)
- Feature-parity registry issue (this epic) — blocker for this issue
