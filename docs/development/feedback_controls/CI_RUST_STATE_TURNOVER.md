# Unit-Gate Rust State Turnover — #11977

## Problem and Scope

Feedback-control merge-group runs 37944318366 and 37952802412 stopped during
Rust installation before unit tests. The latter logged an existing stable
toolchain with an unreadable version and a missing manifest, despite the
workspace-local Rust homes introduced by #11595. Its runner was GitHub-hosted;
shared-runner concurrency is not an established cause.

## Implementation and Contracts

`scripts/ci/prepare_rust_environment.py` allocates a unique temporary parent and
empty `rustup` and `cargo` directories under the actual runtime `RUNNER_TEMP`.
It appends bindings to `GITHUB_ENV` for subsequent steps. Paths containing line
breaks or NUL, missing temporary roots, and invalid destinations fail explicitly.
The destination opens before allocation; existing installations are neither
selected nor deleted. The Actions runner owns temporary-state cleanup.

Only `unit-test-gate` adopts the new preparation step. Other Rust jobs retain
their existing isolation. Toolchain action, version verification, Cargo cache,
Rust wheel build and fail-closed import probe remain in place. No model,
controller, parity denominator or scientific acceptance requirement changes.

## Validation and Handoff

TDD first demonstrated missing-helper collection failure, then passing behavior
tests. The existing workflow suite also exposed three workspace-only assertions;
these now accept the tested preparation command before installation, with
negative cases for missing, late, replaced or overridden preparation.

Run `python -m pytest --noconftest -o addopts= tests/ci/test_prepare_rust_environment.py tests/ci/test_unit_gate_rust_kernel.py tests/ci/test_check_workflow_contexts.py`.

Local tests cover preserved partial state, concurrent/repeated job isolation,
environment publication, invalid inputs and workflow ordering. Protected CI must
still demonstrate successful Rust installation, kernel build and unit execution.
Do not requeue unchanged dependent PRs repeatedly while this failure persists.
