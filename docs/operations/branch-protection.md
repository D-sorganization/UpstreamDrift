# Branch Protection

This document is the repository contract for `main` and release tag protection.
Reviewers should compare the GitHub repository settings against this document
when security or release workflows change.

## `main`

`main` requires pull requests before merge.

Required review rule: at least one approving review for every pull request.
CODEOWNERS review is required for affected owned paths, including
`.github/workflows/`, `scripts/`, and `/src/`.

The branch requires all `ci-standard` jobs to pass before merge. Required
status checks include lint, formatting, type checks, file-size and module-size
guards, coverage, pip-audit, Bandit, Semgrep, and Trivy filesystem scanning.

Branches must be up-to-date with `main` before merge. The repository uses
linear history for normal changes, so squash or rebase merges are allowed and
merge commits are not the default path.

Force-push to `main` is denied. Deleting `main` is denied.

## Release Tags

Tags `v*.*.*` are protected. Only the release workflow may create or update
version tags, and direct pushes to protected version tags are denied.

## Verification

The GitHub branch-protection API was not accessible to the automation token
used for issue #3844 remediation. A repository administrator must verify the
live settings against this contract after merge.
