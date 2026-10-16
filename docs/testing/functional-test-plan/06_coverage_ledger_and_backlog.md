## Coverage Ledger

Measured against `origin/main` `c523c07b5` with all sibling model-pack repos
resolved. Reproduce with
`QT_QPA_PLATFORM=offscreen python3 scripts/testing/enumerate_test_surface.py --summary`.

| Surface                            | Population | Rows covering it                  | Covered | Coverage |
| ---------------------------------- | ---------: | --------------------------------- | ------: | -------: |
| Native runtime tiles               |         75 | `NAT-01`–`46`, `PRV-01`–`06`      |      75 |     100% |
| Shared-manifest tiles              |         75 | `NAT-*` + `MAN-01`–`24` + `PRV-*` |      75 |     100% |
| Union of both tile inventories     |         99 | `NAT-*` + `MAN-*` + `PRV-*`       |      99 |     100% |
| React routes                       |         17 | `SH-01`, `WEB-01`–`21`            |      17 |     100% |
| Backend route families (non-`/v1`) |         39 | `API-01`–`19`                     |      39 |     100% |
| Backend paths (non-`/v1`)          |        154 | `API-01`–`19` (by family)         |     154 |     100% |
| Launcher menu entries              |        105 | `QT-13`–`QT-32`                   |     105 |     100% |
| Feature-parity entries             |         37 | `PAR-01`–`04`                     |      37 |     100% |

Prior coverage on the same populations, before this revision: 43/75 native
(57%), 35/75 shared (46%), 14/17 React routes (82%), 0/154 backend paths,
0/105 menu entries via any specific row, 0/37 parity entries.

### Deliberately Out of Scope

| Surface                                               | Reason                                                                                           |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `/api/v1/*` individually (146 mirrored routes)        | Byte-identical mirror of `/api/*`; sampled by `API-03` instead of duplicated.                    |
| `POST /api/capabilities/{name}/install`               | Installs software. Prohibited for an unattended agent; verify the button exists, never press it. |
| `/api/auth/register`, credential and API-key creation | Creates persistent secrets. `API-04` asserts unauthenticated denial only.                        |
| Individual theme/plot-style entries (26 + 27)         | `QT-23`/`QT-24` sample three each; per-entry rows would add 51 rows of near-zero defect yield.   |
| `Help → Report a Bug…` submission                     | Would file spurious issues. `QT-31` stops at "the form opens pre-populated".                     |
| Rust-kernel-strict paths (`NAT-25`)                   | Requires a `maturin develop` build; recorded `BLOCKED` per `PF-04` when the kernel is absent.    |
| Real participant / production data                    | Every row uses the `PF-03` disposable fixture set.                                               |

### Top Uncovered Risk That Remains

1. **Two divergent tile inventories** (`SH-10`). 24 native-only and 24 web-only
   ids means a user-visible tile can regress on one surface with the other
   green. The matrix now detects the divergence; it cannot fix it.
2. **`MAN-02`, `MAN-04`, `MAN-06`–`MAN-14`, `MAN-18`, `MAN-20`–`MAN-24`** advertise
   no reachable web route (`—`), so the tile is visible but unclickable-to-value.
   That is the open `#7461` parity gap and is why `WEB-*` cannot reach them.
3. **Provider children on unavailable runtimes** (`PRV-04`): 21 of 28 are
   `runtime_unavailable` on a default machine, so `PRV-05` gives real coverage
   for MuJoCo only. Drake/OpenSim/Pinocchio provider paths stay effectively
   untested until those runtimes install.
4. **File-dialog automation** — several `NAT-10`/`QT-08` rows have historically
   ended `NOT RUN` because the Windows accessibility bridge does not expose
   owned file pickers. Rows depending on a picker should be given a
   fixture-path CLI or environment override so they become automatable.

## Improvement Backlog

- Wire `scripts/testing/enumerate_test_surface.py` into CI as the `SH-12` gate:
  a new visible tile with no `NAT-*`/`MAN-*`/`PRV-*` row fails the build.
- Convert P0 rows into deterministic automated smoke tests that run against a
  built UI and real local API; retain this manual plan for visual/desktop and
  optional-engine coverage.
- Give every fixture-consuming tool a non-interactive input path (CLI flag or
  env var) so file-dialog rows stop landing on `NOT RUN`.
- Add a test-fixture registry with checksums and capability tags so agents can
  select safe data without guessing.
- Publish a lightweight per-run JUnit/Markdown artifact from the computer
  agent (`$EVIDENCE/results.jsonl` already has the shape), then trend P0 pass
  rate, failures by feature, and blocked dependency rate in CI/release reviews.
- Reconcile `models.yaml` and `launcher_manifest.json` into one inventory, or
  publish the web-only/native-only exception list this plan currently has to
  infer.
