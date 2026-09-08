# Workflow Tracking Document: Golf Modeling Suite

This document lists all active GitHub Workflows in this repository hub.

| Workflow Name                | Filename                         | Status   | Purpose                                                                                                                                                                                                                                                                            |
| :--------------------------- | :------------------------------- | :------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Control Tower**            | `Jules-Control-Tower.yml`        | Active   | Orchestrates agentic workers.                                                                                                                                                                                                                                                      |
| **PR Compiler**              | `Jules-PR-Compiler.yml`          | Active   | Compiles PR info for fleet management.                                                                                                                                                                                                                                             |
| **CI Standard**              | `ci-standard.yml`                | Active   | Core lint/test lane; does not claim full optional-engine coverage.                                                                                                                                                                                                                 |
| **Release**                  | `release.yml`                    | Active   | Tag-driven wheel/sdist build, PyPI publish, GitHub release; `build` compiles `ui/dist` with Node 24 before `python -m build` because `build_hooks.py` refuses to package without it (UD #9449); wheel smoke asserts the UI bundle and version on Python 3.11–3.12 only (RM #1507). |
| **Vendor Freshness**         | `vendor-freshness.yml`           | Active   | Submodule staleness + Cargo/pyproject Tools-pin consistency (`check_tools_pins.py`, UD #9406).                                                                                                                                                                                     |
| **Seam Drift Gate**          | `ci-standard.yml` (job)          | Active   | `seam-drift-gate`: enforces `docs/shared_tools/seam_rulings.v1.json` (UD #9406).                                                                                                                                                                                                    |
| **CI Fast Tests**            | `ci-fast-tests.yml`              | Active   | Runs unit and integration tests (non-slow).                                                                                                                                                                                                                                        |
| **Nightly Cross-Engine**     | `nightly-cross-engine.yml`       | Active   | Dedicated native-engine validation lane with strict import checks.                                                                                                                                                                                                                 |
| **Cross-Engine Equivalence** | `cross-engine-equivalence.yml`   | Active   | Cross-engine equivalence + canonical conformance gate; its tool-cache ownership repair is scoped to `$RUNNER_TOOL_CACHE` only — it must never `chown -R` a runner `_work` tree, which invalidates git's index stat cache and breaks the next job's checkout (UD #9443, RM #1507).  |
| **Critical Files Guard**     | `critical-files-guard.yml`       | Active   | Prevents accidental deletion of core files.                                                                                                                                                                                                                                        |
| **Assessment Generator**     | `Jules-Assessment-Generator.yml` | Active   | Automated architecture & quality audits.                                                                                                                                                                                                                                           |
| **Auto-Repair**              | `Jules-Auto-Repair.yml`          | Disabled | Automatically fixes CI failures (Disabled via `if: false`).                                                                                                                                                                                                                        |
| **Test Generator**           | `Jules-Test-Generator.yml`       | Active   | Generates unit tests for new Python changes.                                                                                                                                                                                                                                       |
| **Doc Scribe**               | `Jules-Documentation-Scribe.yml` | Active   | Maintains CodeWiki and documentation updates.                                                                                                                                                                                                                                      |
| **Scientific Auditor**       | `Jules-Scientific-Auditor.yml`   | Active   | Peer reviews physics and math correctness.                                                                                                                                                                                                                                         |
| **Conflict Fix**             | `Jules-Conflict-Fix.yml`         | Active   | Resolves merge conflicts agentically.                                                                                                                                                                                                                                              |
| **Tech Debt Assessor**       | `Jules-Tech-Debt-Assessor.yml`   | Active   | Tracks and reports technical debt weekly.                                                                                                                                                                                                                                          |

---

## Maintenance

Update this document whenever a new workflow is added or the status of an existing workflow changes. For global standards, see `Repository_Management/docs/architecture/WORKFLOW_GOVERNANCE.md`.

## Notes

- `ci-standard.yml` is the default core PR lane. It is intentionally fast and
  honest about optional-engine coverage.
- `ci-standard.yml` concurrency: the group is per-ref on branches/PRs (a newer
  push cancels the superseded run) but per-commit on `main`, so consecutive
  merges queue instead of cancelling each other and `main` always finishes a
  run (RM #1507, #9409). `cancel-in-progress: true` stays literal because
  `lint-workflow-files.yml` greps for it.
- `ci-standard.yml` `tests (3.x)` budget: `timeout-minutes: 150`, measured
  2026-09-03 from run 33779933815 (RM #1507, UD #9431). The lane runs serially
  (`-n 0`, since 2026-06-13) with coverage over 40,488 selected tests at
  ~5.27 tests/s, so a full pass needs ~135 min. The previous 35-minute budget
  predated the serialisation and killed every `main` run at exactly 35
  minutes, failing `quality-gate` without reporting any assertion. Re-measure
  this budget whenever the lane's parallelism or selection changes; per-test
  hangs are bounded separately by `--timeout=60 --timeout-method=thread`.
- `ci-standard.yml` `tests (3.x)` failure reporting: each pytest invocation in
  the lane writes JUnit XML to `$RUNNER_TEMP/junit/` and the
  `Upload JUnit Test Results` step publishes it as the `junit-tests-<python>`
  artifact (`if: always()`, 14-day retention, RM #1507, UD #9474). Fetch it
  with `gh run download <run-id> -R D-sorganization/UpstreamDrift -n junit-tests-3.12`.
  The XML is a second copy of the terminal summary, not a replacement for it:
  both are written during pytest's terminal-summary phase, so a run that dies
  before that phase produces neither. `main` @ `4ec3da33` did exactly that -
  a test replaced `builtins.__import__` with a hook that raised for every
  unrecognised name, pytest's failure formatter could not complete its own
  lazy import, and the session aborted with `INTERNALERROR` (exit code 3)
  after counting 305 failures without naming one. The guard against that
  regression is `tests/unit/repo_hygiene/test_no_non_delegating_import_hook.py`.
- `nightly-cross-engine.yml` is the repo's dedicated cross-engine lane and is
  the right place to expand stricter native-engine validation over time.
- `release.yml` job `companion-protected-main` always runs on pushes to
  `main` (no path filter) and publishes the attested artifact
  `upstreamdrift-companion-<sha>` (30-day retention) plus
  `upstreamdrift-companion-evidence-<sha>`. Payloads: `manifest.json`
  (stable consumer name, byte-identical to `upstreamdrift-companion.v1.json`),
  `capabilities.json`, `screenshots.json` (metadata-only, `pending`), the
  three matching schemas, the acquisition schema, the compatibility policy,
  and a `.sha256` sidecar per file. Tag pushes attach the same set to the
  draft release. The job sets `PYTHONPATH` to the workspace and pins
  `jsonschema`/`pyyaml` so the import-free builder runs on any runner
  (RM #1507, #9416). `ci-standard.yml` runs
  `scripts.companion_publication check` on code PRs.
