# Impact Provider Import Compatibility Turnover

Parent #9700; focused prerequisite #9735; related provider PR Tools #5077 and
qualified lumped-impact Tools #5082. The full impact/acoustics program
remains active. A1 (AffineDrift #4258) and U1 (#9706) are merged; no distributed
shaft, acoustic or experimental completion is implied.

## Workspace and Claim

`C:/Users/diete/Repositories/UpstreamDrift-impact-provider`, branch
`fix/9735-impact-provider-imports`, base `dbc6727aa4f0d422b7adaf6957e658e8997f7f29`.
The claim was free and the codex lease succeeded, session
`impact-acoustics-01a07d8a-provider`, expiry 2026-09-08T04:17:08Z.
Production is unchanged. The provider test setup and origin helper are corrected,
with two new origin regressions. No-vendor checks completed before initializing
the existing exact vendor pin for supplementary validation; the gitlink is unchanged. Read repository AGENTS/CLAUDE and
seam handoff before a fix; preserve all protected research records.

## Reproduction and Positive Control

Tools provider source is `C:/Users/diete/Repositories/Tools-impact-coupling`;
implementation `31ebe4993` and reproduction head `eb78179b5`. Its ignored `.venv`
is Python 3.12.10 with system packages plus pytest-qt 4.5.0. The exact provider
checkout has no production changes after these commits, only turnover updates.

Set TOOLS_REPO_ROOT to that provider, REQUIRE_REAL_TOOLS_REPO=1, and
QT_QPA_PLATFORM=offscreen. Match CI PYTHONPATH order: provider root,
provider/src/shared/python, provider/src, provider/src/python/src.
Run the existing test
`tests/shared_contracts/test_tools_provider_contracts.py::test_fresh_provider_import_preserves_downstream_modules`
with the Tools `.venv/Scripts/python.exe`, `-m pytest -n 0 -q -o addopts='' --timeout=60 --tb=short`.
A 90-second subprocess boundary guarded collection. It returned normally:
**1 failed in 4.67 s**, the same missing `src.shared.python.logging_pkg` as CI.
Full local failure is in system temp `impact-provider-repro.log`; authoritative
CI job: https://github.com/D-sorganization/Tools/actions/runs/34163726663/job/101870569015.

In a separate plain Python process with the same provider paths, importing
`src.shared.python.cli_utils` **succeeds**. Its module is the downstream copy,
`shared.python` is the Tools copy, and `src.shared.python.logging_pkg.logging_config`
is the Tools copy. This proves that the existing fallback can work; do not
blindly change the production import or broaden Tools namespace takeover.
A plain import of the entire perturbation adapter passed this logging stage but
then lacked `bunkershot3d`, because the downstream package was not installed in
that isolated environment. That is not evidence of an impact-adapter defect.

## Next Discriminating Tests

Inspect `tests/conftest.py` path ordering and cached `shared.python` before the
shared-contract conftest promotes Tools paths. Root test setup puts downstream
src and shared paths first and mutates package paths. `src/__init__.py` already
contains an installed/vendored fallback, ownership protection and initialization
ordering logic; `_installed_tools_spec` resolves a canonical name that may itself
be the downstream shadow in this test context.

The current hypothesis is test-bootstrap contamination. Establish it with a
minimal fresh-process regression and real installed-provider coverage, then fix
the smallest responsible setup boundary. Preserve the test's actual requirement:
refreshing a Tools provider must not evict downstream-owned module identities.
Do not weaken that assertion, initialize vendor merely to hide the failure, or
copy logging code. Keep normal local-suite behavior intact. The helper already
clears canonical shared names within `_fresh_provider_import`; distinguish
preparing a fresh provider context from behavior being asserted inside it.

Run full provider/vendoring contracts, focused import-order/CLI behavior, lint,
file budgets and manual governance. Update SPEC and the root handoff in the fix
commit, and attach Fixes #9735. The program matrix remains in Tools
`docs/development/impact-acoustics/PROGRESS.md`.

## Implemented Correction and Evidence

The identity test now prepares the actual downstream consumer inside a fresh
provider context before testing a second refresh. It still asserts the exact
same downstream module object survives and now checks its physical downstream
source origin. The scope of cache eviction is unchanged. Provider-origin checks
use the configured TOOLS_REPO_ROOT rather than guessing ownership from directory
names. Two new regressions reject an unrelated Tools-named directory and accept
a legitimately configured checkout with a different name.

RED: the identity test and both origin regressions failed (3 failed, 7 deselected,
4.97 s). GREEN: all 13 provider/vendoring contracts passed without an initialized
vendor (23.10 s), after installing the required h5py dependency in the isolated
local environment. Temporarily adding src.shared to the eviction prefixes made
the unchanged identity requirement fail at its KeyError in 12.58 s; the working
source was restored. This verifies that the setup correction did not weaken
that protection. Mutation log: system temp impact-provider-identity-mutation.log.

The local `.venv` uses Python 3.12 with system packages, pytest-qt 4.5.0 and h5py;
the downstream is installed editable with SKIP_UI_BUILD=1, matching the existing
consumer lane's local-editable mode. A CI/release-mode install correctly refused
the absent vendor and was not used as release evidence. The provider wheel
ud_tools-1.16.1-py3-none-any.whl was built from Tools head 2f975d06e and installed
without dependencies. SHA256:
`b1a38c958d71239a51fad817869aacc67adb62e5d7367af3c1536ae75d1dcc97`.

With both TOOLS_REPO_ROOT and PYTHONPATH removed, independent plain-process
assertions verified the canonical and logging modules came from this wheel's
site-packages directory, the CLI module and src package came from this checkout,
no vendor existed, and CLI parser/logging-argument behavior worked. This is
installed-provider software verification, not release or scientific approval.

After those checks, initialized only the already-recorded Tools gitlink
`eab74a901a7c8467e1997049a73e2cfd2df74428` for the pinned-provider route. No vendor
files were edited. The wider vendor/provider/CLI/fallback run passed 72 tests in 33.40 s.

Pinned Ruff 0.15.17 reports all 6,649 files formatted and no lint errors.
Design-manual governance passes with its existing release block retained.
PR #9745 is open at implementation commit `163239a6b`; all local commit and push hooks passed. Protected CI is running. Resolve its checks before normal merge, then retry the relevant Tools downstream lane against merged consumer code. T3 work is active in the separate Tools-impact-shaft worktree; the full program remains open.

## CI Mode Qualification

At head `66c2918df`, CI's shared-tools job selected `--tools-mode=vendored` but
workspace setup exported TOOLS_REPO_ROOT for the separate `_tools_dep` checkout.
The stronger source assertion exposed seven mismatches; local reproduction with
a different external provider and the initialized vendor likewise failed seven
contracts (20.95 s). The shared conftest now honors root conftest's explicit
TOOLS_REPO_PATH override and vendored mode, reports that selected root, and
promotes paths after root configuration. It does not accept arbitrary Tools
trees or change runtime imports. Both actual modes pass the original 13
contracts: vendored 18.46 s, external with vendor present 16.38 s. An additional
contract asserts that explicit mode and reported origin agree.

The phantom guard initially matched only the issue's illustrative runtime path.
Issue #9735 now records the located test/fixture paths and diagnosis, preserving
its original smallest-responsible-boundary acceptance criteria. No override
label, tolerance relaxation or production edit was used to bypass that gate.

## Provider CI and Main Synchronization

At `94034b56e`, the actual shared-tools-consumer-contracts job passes
(run 34183574996, job 101927443723). Remaining failures include launcher parity
(provider_unavailable versus ready statuses for seven tiles) and a repository-wide
security scan reporting existing files outside this PR, including the sidekick
Python REPL and MJCF parser. No fixture path is among the inspected findings.
Main advanced to `6bb246e38` (#9740); merge preserves that model-generation
implementation unchanged and resolves only the adjacent SPEC rows, retaining
both #9735 and #9740. Revalidate provider contracts and normal protected CI;
do not infer completion from the earlier narrower checks.
