# In-tree test policy (`src/**/tests`)

Issue #7126 was that plain `pytest` only collected `tests/` while tracked
package tests under `src/**/tests` were skipped implicitly. The default pytest
configuration now lists the intentional in-tree Python test directories in
`pyproject.toml [tool.pytest.ini_options].testpaths`.

The policy is:

- New Python tests should still prefer the top-level `tests/` topic layout.
- Existing tracked Python tests under `src/**/tests` must be covered by
  `testpaths` unless they are explicitly removed from the tree.
- `scripts/check_pytest_intree_testpaths.py` is the blocking guard that compares
  tracked `src/**/tests` and `src/**/test*.py` files against the configured
  pytest paths.
- `make test-in-tree` runs that guard as a named local command.

This keeps in-tree tests visible to default collection without recursively
collecting all of `src/`, which also contains engines, MATLAB bridges, generated
assets, and non-Python fixtures.
