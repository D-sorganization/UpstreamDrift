# Contributing to Movement Optimizer

Thank you for your interest in contributing! This document provides guidelines for contributing to Movement Optimizer.

## Development Setup

```bash
git clone https://github.com/D-sorganization/Movement-Optimizer.git
cd Movement-Optimizer
pip install -e ".[dev]"
```

## Code Style

- **Formatter**: `ruff format` (do not use Black)
- **Linter**: `ruff check`
- **Type checker**: `mypy --ignore-missing-imports`
- **Line length**: 100 characters

Run before committing:

```bash
ruff check src/ tests/ --fix
ruff format src/ tests/
```

## Design Principles

- **DBC (Design by Contract)**: All public methods must check preconditions and raise `ValueError` or `TypeError` on violation. Do not use `assert` for input validation in production code.
- **DRY**: Factor shared logic into helpers. Constants live in `constants.py`.
- **LoD (Law of Demeter)**: Callers interact only through the public API.
- **Logging**: Use `logging.getLogger(__name__)` in `src/`. Never use `print()`.

## Adding a New Exercise

1. Define start/end joint angles in `constants.py` or a new module under `exercises/`.
2. Create a config factory function (see existing examples in `models.py`).
3. Add a test in `tests/` that optimizes the exercise and asserts convergence.
4. Wire the exercise into the GUI in `gui/main_window.py`.
5. Run the full test suite.

## Testing

```bash
python3 -m pytest tests/ -v --tb=short
```

- Use fixtures from `conftest.py` for shared body model instances.
- Seed all RNGs for determinism.
- Optimizer tests use loose tolerances since L-BFGS-B results vary across platforms.

## Pull Requests

1. Create a feature branch from `main`.
2. Make your changes with clear, atomic commits.
3. Ensure all tests pass and linting is clean.
4. Open a PR with a description of what changed and why.
5. All commits must include a `Signed-off-by:` line (DCO; see below).

## Developer Certificate of Origin (DCO)

By contributing to this project, you agree to the following terms:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

### Sign-off Requirement

All commits must be signed off using `git commit -s`. This adds a
`Signed-off-by:` line to your commit message:

```bash
git commit -s -m "feat: your description here"
```

If you have already made commits without sign-off, amend them before
opening a PR:

```bash
git rebase --signoff HEAD~N
```

Replace `N` with the number of commits to amend, then force-push to
your feature branch.

## Reporting Issues

Open a GitHub issue with:

- A clear title and description
- Steps to reproduce (if applicable)
- Expected vs. actual behavior
- Python version and OS
