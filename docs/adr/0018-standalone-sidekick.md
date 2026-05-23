# ADR-0018: Standalone Sidekick Application

- Status: Accepted
- Date: 2026-05-23
- Decision Makers: dieterolson, claude (AI)
- Related Issues/PRs: #5969 (epic), #5984 (T6), #5985 (T7), #5986 (T8), #5987 (T9)

## Context

Sidekick currently runs only inside the UpstreamDrift launcher or the
React/Tauri shell — users cannot launch it without first installing the full
UpstreamDrift suite.  Several users and downstream projects need a lightweight,
self-contained Sidekick window that requires nothing beyond a Python or PyInstaller
install.

The standalone build must:
1. Remain consistent with the embedded version (same calculators, same theme
   contract, same headless `sidekick run` interface).
2. Not duplicate the existing embeddable-tool contract or theme system.
3. Ship a one-file binary for macOS, Linux, and Windows for users who cannot
   or will not install Python.

## Decision

### Packaging (T6 / T7)

Add a `sidekick = "sidekick.__main__:main"` console script to the
**repo-root `pyproject.toml`**.  The canonical library package
(`sidekick.*`) continues to live in `vendor/ud-tools/`; the standalone shell
(`sidekick.standalone.*`) lives here.  This separation avoids shipping vendor
content in the wheel and keeps `vendor/ud-tools/` as the source-of-truth for
shared utilities.

A `sidekick.spec` PyInstaller spec produces a one-file binary excluding heavy
physics engines (`pybullet`, `mujoco`, `pydrake`) and ML training frameworks.
Binary size is capped at 250 MB; growth > 10 % between releases requires PR
justification.

### Profile model (T8)

Two mutually exclusive profiles are supported in v1:

| Profile | Default panel | Sidebar default |
|---------|--------------|-----------------|
| `chat-first` | `AIAssistantPanel` | Collapsed |
| `calc-first` | `ToolsSidebar` | Expanded |

The profile is stored in `StandalonePreferences` backed by `FileSessionStore`
(a plain JSON file in `platformdirs.user_config_dir("sidekick")`).  Tests
inject `InMemorySessionStore` so nothing touches the user's real `~/.config`.

### Persistence boundary (T8)

`StandaloneSessionStore` / `FileSessionStore` owns *only* user-editable
preferences (4 keys: profile, theme, data_dir, llm_provider).  Ephemeral
session state (chat history, calculator intermediate values) is held in memory
and discarded on exit.  This keeps the on-disk footprint minimal and avoids
migration complexity.

### Onboarding sentinel (T8)

A single empty file `onboarded` in `platformdirs.user_config_dir("sidekick")`
signals that the 3-step first-run wizard has been completed.  The wizard is
skipped when:
- the sentinel exists, or
- `--skip-onboarding` is passed on the command line (CI smoke tests).

### Standalone-vs-embedded round-trip invariant

A calculator that works in embedded Sidekick (via `sidekick.process_calculators`)
must also work via `sidekick run --calculator <name>`.  The headless runner
(`sidekick.standalone.runner`) is the contract boundary: it accepts JSON inputs
and emits JSON outputs without any GUI or display dependency.

### Documentation and discoverability (T9)

- This ADR is the authoritative design record.
- `docs/sidekick/standalone.md` is the user-facing getting-started guide.
- `docs/sidekick/README.md` cross-links `standalone.md`.
- `AGENTS.md` tells contributors to use `sidekick.standalone.*` rather than
  writing a new shell.

## Alternatives Considered

1. **Separate standalone repository** — rejected; would fragment the codebase
   and break the shared-calculator invariant.
2. **Tauri/Electron wrapper** — rejected for v1; adds Node.js/Rust build
   complexity and a much larger binary.  Revisit when the UI needs richer
   web-native interactivity.
3. **vendor/ud-tools as the console-script home** — rejected; the standalone
   shell window is UpstreamDrift-specific and should not pollute the shared
   library package.

## Consequences

- Positive:
  - Users can `pip install upstream-drift && sidekick --help` immediately.
  - CI smoke-tests the binary on every release tag.
  - The headless `sidekick run` interface enables scripting and CI use.
- Negative:
  - Binary size budget (250 MB) adds a CI gate to every release.
  - `sidekick.standalone.*` is UpstreamDrift-only; cross-project sharing
    requires extracting it to `vendor/ud-tools/` later.
- Follow-ups:
  - Code signing / notarisation (#TBD).
  - Auto-update mechanism (#TBD).
  - `sidekick` PyPI publication (separate ops decision).

## Validation

- `tests/unit/packaging/test_sidekick_console_script.py` — script declared,
  module importable, standalone package exists.
- `tests/unit/packaging/test_pyinstaller_spec.py` — spec includes/excludes.
- `tests/unit/sidekick/standalone/test_preferences.py` — DbC round-trip.
- `tests/unit/sidekick/standalone/test_onboarding.py` — sentinel logic.
- `tests/unit/repo_hygiene/test_sidekick_docs.py` — ADR accepted, cross-links.
- `release-sidekick-binary.yml` smoke tests — `--help` < 5 s, `run` exits 0.
