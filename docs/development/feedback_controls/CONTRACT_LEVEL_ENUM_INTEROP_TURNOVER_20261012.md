# Contract Enum Interop Turnover

Issue #12403 addresses a confirmed cross-module enum identity defect. The
Tools-canonical `shared.python.contracts` and UD-owned
`src.shared.python._contracts_level` expose different enum classes. A fresh
process showed that passing the UD `ENFORCE` member directly to the canonical
Tools setter left the foreign member stored, and a failed `require` call did
not raise.

This change leaves the canonical Tools setter source untouched. The UD alias
installer wraps that setter and normalizes a compatible enum to the canonical
module's enum before delegation. The UD shared-level and core-level setters
also normalize to their local enum types. Direct calls to the unwrapped
canonical setter remain a separate Tools-owned issue, #5577.

TDD covers the UD-installed public wrapper, core/shared/alias interop, each
supported level, invalid input with state preservation, enforcement, warning
logging, and off-mode silence. The focused contract and fragment tests passed
56 tests. Scoped Ruff check/format and mypy passed on the four changed Python
files. Normal hooks passed for implementation commit
`e60afb2b4d61b2776106f28f1184ff524aa2d99b`; the corrective commit restores
`src/shared/python/contracts.py` to its exact parent blob.

A merge-group failure later reported
`TestContractLevelGetSet.test_get_contract_level_returns_level` at head
`ee2c2c395b3ff4ae2a846f0054b3c5e457b75608` (1 failed, 26,245 passed,
1,152 skipped, 8 xfailed). This is consistent with the enum invariant but
does not prove the interop defect triggered that CI failure. The full
repository gate is run separately against the fetched main and recorded in its
own receipt. No native run or qualification claim is made.
