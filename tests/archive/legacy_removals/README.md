# Legacy Test Removals Archive

This directory contains standard `.patch` files documenting the exact state of all tests and files removed during the 2026 Test Suite Remediation (Epic #4927).

If future agents or developers determine that any of the permanently-skipped tests or empty test files removed during the P0 and P1 remediation phases were valuable and need to be restored, they can be recovered directly from these patches.

## Contents

- `P0_empty_files_deleted.patch`: Contains the 10 completely empty test stub files that were deleted.
- `P1_dead_skips_removed.patch`: Contains the 686 test functions and classes that were deleted because they were permanently skipped without an issue tracker link.

## Recovery Instructions

To review the removed tests without restoring them:

```bash
cat P1_dead_skips_removed.patch
```

To fully restore the deleted tests from a patch:

```bash
git apply -R P1_dead_skips_removed.patch
```

(Note: The `-R`/`--reverse` flag reverses the deletion patch, effectively restoring the code).
