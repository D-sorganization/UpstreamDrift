# ADR-0031: Web feature-parity exemptions (desktop-only features)

- Status: Accepted
- Date: 2026-06-12
- Decision Makers: UpstreamDrift maintainers
- Related Issues/PRs: #7460, #7445, epic #7462

## Context

The PyQt6 desktop app is the canonical model and the Tauri/React web app must
match it (epic #7462). Several large desktop features have no web counterpart,
and until now that was implicit: every parity review re-litigated them, and
contributors could not tell deliberate scope from accidental drift. ADR-0028
set the precedent for one such decision (embedded tabs/docks vs. the web
multi-window model); the remaining features needed the same treatment.

The web shell runs against a **local server trust model**: the FastAPI backend
is a localhost process serving a browser UI. Anything reachable from the
browser is reachable from any page the browser visits (CSRF/DNS-rebinding
class risks) and from any user on a shared machine. The browser surface must
therefore never expose arbitrary code execution or host-administration
capabilities.

## Decision

The feature-parity registry (`src/config/feature_parity.json`, #7445) is the
**enforcement mechanism**: every exempt feature carries a one-sentence reason
citing this ADR, and CI (`tests/config/feature_parity/`) gates the registry
plus the generated matrix doc. Parity audits diff against the registry instead
of rediscovering these decisions.

Decisions made in #7460, in three groups:

### 1. Trust-model exemptions (never web features under the local-server model)

- `sidekick.terminal_repl_jupyter_skills` — an OS terminal, Python REPL,
  Jupyter kernel, or skills runner exposed over HTTP is a remote-shell
  endpoint. That is a non-starter: any browser page could pivot through it to
  full code execution on the host. Web chat instead gets structured context
  via #7453.
- `launcher.docker_management` — Docker daemon control is root-equivalent on
  most hosts; exposing start/stop/build over HTTP would hand container escape
  primitives to the browser surface. Candidate for Tauri mode only, where the
  shell is a trusted native process.
- `launcher.mcp_config` — writes local MCP configuration files for desktop AI
  integrations; file-system configuration of the user's machine is a
  local-machine concern outside the web trust boundary.

### 2. Capability exemptions (depend on local resources the browser cannot have)

- `tools.matlab_suite` — requires a licensed local MATLAB installation;
  browser-exempt, possible later in Tauri mode.
- `engines.dashboards` and `biomech.exercise_injury_dashboards` — experimental
  desktop dashboards whose equivalent web surface is the Simulation page;
  exercise routing is not a web feature.
- `launcher.embedded_tabs_docks` — already decided in ADR-0028 (web uses the
  multi-window model).

### 3. Roadmap exemptions (plausible web features, deliberately deferred)

Marked `exempt` with reason "roadmap — revisit after epic #7462 core lands"
rather than opened as issues now, so the gap list stays an actionable backlog:

- `tools.pose_editing` (Pose Studio) — heavy native-viewer coupling; a web
  pose editor is a major project.
- `docs.document_library` — indexes local files via SQLite; a thin web reader
  is plausible later.
- `simulation.golf_suite_batch` — parameter sweeps/batch runs could become
  API-backed web jobs.
- `optimization.swing_optimizer` — long-running trajectory optimization suits
  an API-backed job model.

## Alternatives Considered

1. Open roadmap issues now for group 3. Rejected: issues without a committed
   horizon go stale and dilute the gap backlog; the registry reason preserves
   the intent and the revisit trigger (epic #7462 core landing).
2. Expose terminal/Docker behind authentication on the web surface. Rejected:
   localhost auth does not remove the CSRF/rebinding attack surface, and a
   compromised browser session would still gain host code execution.
3. Leave exemptions implicit (status quo). Rejected: every parity audit
   re-litigated the same features.

## Consequences

- Positive: parity audits diff against the registry; scope decisions are
  explicit, cited, and CI-enforced; the security boundary of the local server
  is written down.
- Negative: roadmap items have no tracking issue until they are picked up;
  the revisit trigger is procedural (post-epic review), not automated.
- **Changing any exemption requires updating this ADR** (and the registry
  entry's reason) in the same PR — the registry description and CLAUDE.md
  state this rule.

## Validation

`tests/config/feature_parity/test_registry.py` (exempt entries require a
reason; no `pending_decision` entries remain) and
`tests/config/feature_parity/test_matrix_freshness.py` (generated matrix
matches the registry byte-for-byte).
