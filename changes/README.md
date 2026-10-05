# Change Fragments (RM-5)

This directory contains per-PR change fragments following the RM-5 specification.

## Overview

To avoid merge conflicts on shared documentation files (`SPEC.md`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`) under a GitHub merge queue, pull requests introduce their own change fragment `changes/<issue>-<slug>.md` instead of editing the shared files directly.

## Usage

Create a new fragment for your pull request:

```bash
python shared_scripts/changes_fragment.py new --issue <ISSUE_NUMBER> --summary "<SPEC row summary>"
```

Optional arguments:

- `--slug <SLUG>`: Short hyphen-delimited slug for the fragment filename.
- `--dl-state <STATE>`: Development log entry state (`in_progress`, `in_review`, `shipped`, `parked`, `abandoned`).
- `--next-step <TEXT>`: Required when `--dl-state` is an active state.
- `--title <TEXT>`: Title for newly created development log entries.
- `--owner <OWNER>`: Agent or user owner identifier.
- `--branch <BRANCH>`: Branch name.
- `--paths <PATHS>`: Comma-separated paths touched.
- `--handoff <TEXT>`: Markdown handoff continuation notes.

## Validation

Validate fragments locally or in CI:

```bash
python shared_scripts/changes_fragment.py validate
```

## Post-Merge Collation

When a pull request merges to `main`, the `.github/workflows/collate-changes.yml` workflow automatically runs `python shared_scripts/changes_fragment.py collate --pr <PR_NUMBER>`, folding the fragment into `SPEC.md` and the development log, and deleting the fragment file.
