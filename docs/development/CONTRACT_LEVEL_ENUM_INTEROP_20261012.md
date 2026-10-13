# Contract-Level Enum Interoperability

## Problem

The repository loads contract-level enums through distinct module identities.
The active Tools-canonical `shared.python.contracts.ContractLevel` and the
UD-owned `src.shared.python._contracts_level.ContractLevel` are different
Python enum classes despite sharing member values. Passing a foreign enum to
the canonical setter leaves that enum stored; a failed `require` call can then
return when the foreign value is `enforce`.

## Change

The UD-owned alias installer wraps the public setter and normalizes a
compatible enum value to the canonical module's own `ContractLevel` before
delegating. This behavior applies through the UD alias-installation path; the
canonical Tools setter source remains unchanged. The UD shared-level and
core-level setters also normalize values to their local enum types. A shared
helper prepares per-module alias values and enabled flags, and the shared
setter validates all loaded alias conversions before mutating state.

Direct calls to the canonical Tools setter without the UD wrapper retain the
separately reproduced defect. That Tools-owned follow-up is tracked in #5577;
this change does not claim to fix direct canonical calls.

## Evidence and Limits

Regression tests exercise the UD-installed public wrapper, UD core/shared
setters, and compatibility-alias synchronization for all three levels. They
verify that unsupported values leave state unchanged, enforcement raises,
warning mode logs, and off mode is silent. A fresh-process probe reproduced
the unwrapped canonical Tools setter defect.

The direct API defect is confirmed, but its relationship to reported
merge-group CI failures is unproven. Scoped tests, Ruff, mypy, and commit hooks
are recorded in the turnover note. No full repository gate or native runtime
was run before the separate full-gate receipt.
