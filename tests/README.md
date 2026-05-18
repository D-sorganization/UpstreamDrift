# Tests

Each subdirectory of `tests/` corresponds to a test layer or codebase concern.
The default mapping is `src/<package>/<module>.py` to
`tests/<package>/test_<module>.py`. Cross-cutting tests use the topical
directories below.

## Subdirectories

- `acceptance/` - end-to-end user-flow tests.
- `ai/` - AI adapter, chat service, tool registry, and workflow tests.
- `analysis/` - biomechanics, comparative analysis, injury risk, and load tests.
- `architecture/` - DbC, dependency direction, portability, and repository guards.
- `benchmarks/` - performance benchmarks and regression baselines.
- `ci/` - tests for CI workflow and repository automation behavior.
- `cross_engine/` - engine-pair comparison tests.
- `docker/` - Docker build, compose, and entrypoint tests.
- `engines/` - engine-specific tests that do not fit a narrower layer.
- `heavy_integration/` - live or slow integration tests gated by markers.
- `imports/` - import-time and lazy-loading behavior.
- `integration/` - multi-module workflows run in the core CI lane.
- `launchers/` - GUI and CLI launcher tests.
- `parity/` - result parity and cross-implementation comparisons.
- `physics_validation/` - scientific checks against known references.
- `regression/` - issue-specific repro tests; file docstrings should reference the issue.
- `rust_bindings/` - Python-facing Rust extension contracts.
- `scripts/` - tests for repository maintenance scripts.
- `shared_contracts/` - contracts against the vendored Tools repository.
- `tools/` - tests for developer and model-generation tools.
- `ui/` - UI rendering, layout, and interaction tests.
- `unit/` - fast unit tests with the narrowest practical fixture scope.

## Conftests

| Path                                                      | Scope            | Purpose                                                                                              |
| --------------------------------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------- |
| `conftest.py`                                             | session          | Imports MuJoCo before collection on Windows to avoid DLL initialization crashes.                     |
| `tests/conftest.py`                                       | session/function | Adds `--tools-mode`, controls Tools path precedence, and isolates protected engine modules per test. |
| `tests/heavy_integration/conftest.py`                     | session/function | Marks heavy tests as `live_simulation` and provides headless display plus engine fixtures.           |
| `tests/integration/conftest.py`                           | package          | Re-exports shared fixture-library helpers for integration tests.                                     |
| `tests/parity/conftest.py`                                | module/function  | Provides FastAPI client and fresh pendulum engine fixtures.                                          |
| `tests/shared_contracts/conftest.py`                      | session          | Resolves real, vendored, or sibling Tools checkouts for shared contract tests.                       |
| `tests/unit/conftest.py`                                  | function         | Mocks optional native dependencies and resets those mocks between unit tests.                        |
| `tests/unit/dashboard/conftest.py`                        | package          | Documents dashboard test path handling; no fixtures.                                                 |
| `tests/unit/engines/*/conftest.py`                        | package          | Documents engine test path handling or mocks missing optional engine modules.                        |
| `tests/unit/engines/simscape/*/conftest.py`               | package          | Documents Simscape test path handling; no fixtures.                                                  |
| `tests/unit/plotting/conftest.py`                         | package          | Pre-mocks PyQt6 modules to avoid Qt DLL crashes in plotting tests.                                   |
| `tests/unit/tools/humanoid_character_builder/conftest.py` | function         | Skips torch-dependent tests when local torch binaries are unavailable.                               |

## Layout Guard

`scripts/check_test_layout.py` enforces that Python test files live under topic
subdirectories and prevents new `src/**/tests` directories. It retains a legacy
allowlist only for pre-existing in-tree test directories that still need staged
migration; root-level `tests/test_*.py` files are not allowed.
