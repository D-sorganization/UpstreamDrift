# Running the PyQt6 GUI Tests on Linux From a Windows Checkout

Issue #9684. CI runs the GUI suite (`pytest -m ui`) on Linux and it passes.
On some Windows development hosts the same tests die the moment a PyQt6
list view or rich-text widget is created under pytest: exit code
`0xC0000409` (fail-fast), no traceback, no output, even with `-s`,
`--noconftest`, plugins disabled or `faulthandler` on. Inline Python on the
same host creates the same widgets fine. Bisecting Windows for that is not a
good use of time; running the tests the way CI runs them is.

## One Command

From PowerShell or Git Bash on the Windows side, in the repository root:

```bash
wsl -d Ubuntu -- bash scripts/dev/wsl_qt_tests.sh
```

Optional pytest arguments replace the default `-m ui`:

```bash
wsl -d Ubuntu -- bash scripts/dev/wsl_qt_tests.sh tests/tools/capture_rig/test_gui.py -q
```

## What the Script Does

1. Uses `uv` inside WSL to create `.wsl-venv/` (Python 3.12, the CI
   interpreter) at the repository root on first run; the directory is
   git-ignored and reused afterwards.
2. Installs the project in editable mode with the Qt and test dependencies.
3. Runs `xvfb-run pytest` with `QT_QPA_PLATFORM=offscreen` and `src/` on
   `PYTHONPATH`, exactly as the test suite expects.

Requirements inside the WSL distribution: `uv`
(`curl -LsSf https://astral.sh/uv/install.sh | sh`) and
`xvfb` (`sudo apt-get install -y xvfb libgl1 libegl1 libxkbcommon0`). The
script says which one is missing and exits with code 2.

## Notes

- The checkout is read through `/mnt/c`, so file watching and I/O are slower
  than a native Linux clone; the GUI suite is small and this does not matter.
- `.wsl-venv/` is created beside `.venv/` and never touched by the Windows
  interpreter.
- If a GUI test fails only on Windows and passes here and in CI, treat it as
  the host problem described above, not as a test defect.
