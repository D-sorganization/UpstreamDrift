# Explicit No-Probe Preparation Turnover (#12353)

## Scope

Add an explicit preparation-only mode that skips the bounded native capability probe. The public API remains probe-enabled by default; the CLI skip option is valid only with `moco-native --prepare-only`. Existing source/input/body checks and native model construction/finalization/state audits are unchanged.

## Contract

A skipped probe is recorded as requested=false and runtime viability `not-attempted`, without a probe-completed finding, phase, or observed time window. The software-solve readiness property independently requires the probe request, even if a diagnostic assessment is otherwise admitted. CLI not-ready status is 2. The flag does not bypass any other blocker, change the passive policy, or grant physical qualification. No forward integration runs on this preparation path. Existing audits may finalize connections, restore state, or initialize the model; the flag does not promise to avoid all native initialization.

## Validation

SDK-free unit tests exercise the real preparation wrapper with fake native bindings, the assessment/report serialization path, independent readiness gating, CLI default/skip/invalid-mode behavior, and non-boolean rejection before preflight. They do not establish runtime viability, solver execution, physiology, capture matching, replay, or parity.

From the repository root on Windows, the focused test command was:

```powershell
$env:PYTHONPATH = "."
& C:\Users\diete\Repositories\_venvs\mpp-domain-12233\Scripts\python.exe -m pytest -q tests/unit/engines/opensim/test_native_moco_probe_mode.py tests/unit/engines/opensim/test_native_path_optimizer_domains.py
```

Result: 31 passed with one existing `state_manager` deprecation warning. Ruff check and format checks passed for the four changed Python files. The design-manual governance check passed: 62 QMD sources, zero registered calculations, and publication remains blocked.

The agent-context views were refreshed and checked with pinned `vendor/ud-tools` commit `6b6e4be55e7086a33db43b36d2632765be619bde`:

```powershell
$env:PYTHONPATH = "vendor/ud-tools/src"
& C:\Users\diete\Repositories\_venvs\mpp-domain-12233\Scripts\python.exe -m agent_context --root . render
& C:\Users\diete\Repositories\_venvs\mpp-domain-12233\Scripts\python.exe -m agent_context --root . check
```

Both context commands passed. No SDK or native execution was performed.

## Current State

Branch: `feat/native-no-probe-12353-20261012`. Changes are awaiting root review. The design-manual calculation inventory remains empty because this adds no new calculation. Canonical contract documentation is in chapter 56.

## Integrated Maintenance Validation (2026-10-12)

The combined source keeps the exact body-inertia audit serializer in
`native_body_inertia.py` and retains `_physical_audit_bytes` as a preparation
module alias. This keeps the combined preparation module under the 1,200-line
file budget without changing audit bytes or digest semantics. The combined
focused selection passed 35 tests and skipped 10 OpenSim-dependent tests in
the SDK-free QA environment. Ruff check/format, architecture and file-size
budgets passed. Agent-context render/check passed after adding the receipt,
inertia adapter, and probe-mode test to the registered component. No SDK or
native execution was performed.
