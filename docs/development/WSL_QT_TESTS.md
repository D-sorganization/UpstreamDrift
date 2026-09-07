# Running the PyQt6 GUI Tests on Linux From a Windows Checkout

Issue #9684. CI runs the GUI suite (`pytest -m ui`) on Linux; this lane runs
it the same way from a Windows checkout, so a Windows-only symptom can be
told apart from a real defect in minutes.

The case that motivated it: PyQt6 GUI tests that died the moment a widget
was created, with exit code `0xC0000409` on Windows and no traceback. The
Linux lane turned that into a `SIGABRT` with a Python stack, which pointed
at the real cause in the test itself: the helper created the `QApplication`
and returned it without keeping a reference, so Python collected it together
with its C++ object and the next `QWidget` aborted the process. GUI tests
must hold the application at module level (see `tests/tools/capture_rig/test_gui.py`).

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
- A GUI test that aborts with no traceback on Windows usually aborts here
  with one; read the stack before suspecting the host.
