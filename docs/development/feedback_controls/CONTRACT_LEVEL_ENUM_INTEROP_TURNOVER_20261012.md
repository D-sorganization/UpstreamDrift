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
logging, and off-mode silence. The first focused validation passed 56 tests;
a later targeted run passed 17 tests after adding the required unit marker and
updating the alias assertion to compare the canonical module enum identity and
shared value. Scoped Ruff check/format and mypy passed on the four changed
production/test Python files. Normal hooks passed for implementation commit
`e60afb2b4d61b2776106f28f1184ff524aa2d99b`; the corrective commit restores
`src/shared/python/contracts.py` to its exact parent blob.

The full canonical pre-PR gate at `49b82bd37bf1fa08f71382f20822e0ac4cac34b7`
returned exit 1. A read-only run of the same three affected nodes on main
`30e55be1af3bcba1ebc4edd0c4285cefdf781925` reproduced two failures: the
optimized-mode default assertion and the top-level package collision ratchet.
The alias propagation node passed on main and failed only on the feature because
the assertion compared enum instances from distinct module classes; the
corrected assertion now checks canonical identity and shared value. The two
remaining baseline failures are not fixed or waived by this change. The
optimized assertion conflicts with `contracts.py`'s existing `OFF` default
under `python -O` and is tracked separately in #12404. The collision ratchet
reports vendor roots already listed in `pyproject.toml` and is tracked in
#12405; neither failure is fixed by broadening the collision allowlist. After the unit-marker and alias-assertion corrections, the final full
canonical gate at `b080f2b973b174d76dc11d10e375b2dcad9eb229` returned exit 1:
48 affected tests passed and two failed, exactly the two failures reproduced on
main above. Policy and fragment validators passed. The full-gate stdout is
retained in the external planning receipt; no passing full-gate claim is made.

A normal push at that head stopped in the pre-push mypy hook before publication.
Mypy reported four `union-attr` errors because the alias-sync update loop reused
a `Module | None` variable. The follow-up narrows the update tuple to
`ModuleType` and uses a distinct target variable. The exact configured mypy hook
then passed, as did the 17 focused interop and alias tests. This type-only
follow-up has not had a full canonical gate run yet. The remote topic branch
remained absent after the blocked push. No native run or qualification claim is
made.

A merge-group failure later reported
`TestContractLevelGetSet.test_get_contract_level_returns_level` at head
`ee2c2c395b3ff4ae2a846f0054b3c5e457b75608` (1 failed, 26,245 passed,
1,152 skipped, 8 xfailed). This is consistent with the enum invariant but
does not prove the interop defect triggered that CI failure. The full
repository gate is run separately against the fetched main and recorded in its
own receipt. No native run or qualification claim is made.
