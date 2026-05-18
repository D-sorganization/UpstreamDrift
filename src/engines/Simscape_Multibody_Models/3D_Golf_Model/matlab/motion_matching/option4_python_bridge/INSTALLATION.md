# Installation — MATLAB Engine for Python on Windows

The MATLAB Engine API for Python is the **only** non-trivial install for Option 4. Everything else (numpy, pytest, matplotlib) is already in the repo's `requirements.lock`.

The single biggest gotcha is **version pinning**: the `matlabengine` pip package version, the MATLAB release, and the Python minor version are a tightly-coupled triple. Get one wrong and you spend half a day debugging cryptic `ImportError`s.

This document is Windows-first because that is the host platform per [`CLAUDE.md`](../../../../../../../CLAUDE.md). Linux and macOS are best-effort; the steps are similar but the paths and PowerShell commands differ.

## Version pinning

The MathWorks compatibility matrix maps each MATLAB release to a small range of supported Python minor versions and to one `matlabengine` pip package version. **Pick a row first, then install matching versions of all three.**

| MATLAB release | Supported Python          | `matlabengine` pip version                                              | Notes                                                    |
| -------------- | ------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------- |
| **R2025b** ✅  | 3.9, 3.10, 3.11, **3.12** | `matlabengine==25.2.*`                                                  | Verified working on the dev box (May 2026).              |
| R2024a         | 3.9, 3.10, **3.11**       | `matlabengine==24.1.*`                                                  | Originally recommended row.                              |
| R2023b         | 3.9, 3.10, 3.11           | `matlabengine==23.2.*`                                                  | Also fine if R2024a not available.                       |
| R2023a         | 3.8, 3.9, 3.10            | `matlabengine==9.14.*`                                                  | Pre-versioning-rename.                                   |
| R2022b         | 3.8, 3.9, 3.10            | `matlabengine==9.13.*`                                                  | Older but still supported by the `matlabengine` package. |
| ≤ R2022a       | varies                    | use `cd "$matlabroot/extern/engines/python" && python setup.py install` | The pre-pip era; out of scope.                           |

> **Heads up — Python 3.13 / 3.14 are NOT yet supported** by any released `matlabengine` wheel as of MATLAB R2025b. If your repo's default interpreter is 3.13 or newer, install `matlabengine` into a 3.12 or 3.11 sidecar interpreter and run the Option-4 tests with that (e.g. `py -3.12 -m pytest ...` on Windows). The repo's `pyproject.toml` requires `>=3.10`, but only `<=3.12` interpreters can drive the bridge.

> **Always** verify the row against [https://www.mathworks.com/support/requirements/python-compatibility.html](https://www.mathworks.com/support/requirements/python-compatibility.html) at install time — the matrix is updated with each MATLAB release.

The repo's recommended row is **MATLAB R2024a + Python 3.11 + `matlabengine==24.1.*`**. The dev box at the time of issue #4077 was actually **MATLAB R2025b + Python 3.12 + `matlabengine==25.2.*`** — the procedure below is identical aside from the version triple.

### Install attempt log — issue #4077 dev box (2026-05-06)

For future agents debugging install pain. The dev box had MATLAB R2025b at `C:/Program Files/MATLAB/R2025b/` and three Python interpreters available (3.10, 3.11, 3.12, 3.13, 3.14). The repo `pyproject.toml` says `requires-python = ">=3.10"`, so `where python` first picked **3.14**.

Attempt 1 — Python 3.14 + `pip install <matlabroot>/extern/engines/python`:

```text
UserWarning: MATLAB Engine for Python supports Python version 3.9, 3.10, 3.11, and 3.12,
             but your version of Python is 3.14
error: could not create 'dist\matlabengine.egg-info': Access is denied
```

Two failures stacked: (1) Python 3.14 is past the supported window for `matlabengine 25.2`, (2) running setup from `C:/Program Files/...` triggers the Windows ACL on the read-only install dir. Both go away when you switch to a supported interpreter and use the PyPI wheel.

Attempt 2 — Python 3.12 + `python -m pip install matlabengine` (PyPI):

```powershell
py -3.12 -m pip install matlabengine --user
# Successfully installed matlabengine-25.2.2
py -3.12 -c "import matlab.engine; print('OK')"
# OK
```

Attempt 3 — verify Simscape Multibody license:

```powershell
py -3.12 -c "import matlab.engine; e = matlab.engine.start_matlab('-nodesktop -nosplash'); \
    print('Simscape_Multibody:', e.eval(\"license('test','Simscape_Multibody')\", nargout=1)); \
    e.quit()"
# Simscape_Multibody: 0.0       <-- license missing on this dev box
```

Result: `matlab.engine` is importable and an engine starts in ~5 seconds, so the round-trip and lifecycle tests pass. The forward-sim tests skip with a "Simscape Multibody license not available" reason — see [RUNBOOK.md § Smoke test](RUNBOOK.md#1-smoke-test-issue-4077-surface).

## Prerequisites

Before starting:

- **MATLAB** is installed and activated. Confirm: `matlab -nodesktop -nosplash -r "disp('OK'); quit"` prints `OK` and exits. If it prompts for activation, finish that first.
- **Simulink** and **Simscape Multibody** licenses are checked out by your activation. Confirm in MATLAB: `license('test', 'Simulink')` returns `1` and `license('test', 'Simscape_Multibody')` returns `1`.
- **Python 3.11** (or whichever version the matrix row specifies) is installed. Confirm: `python --version` prints `Python 3.11.x`. If you have multiple Pythons, check `where python` (Windows) and prefer the one your repo's venv uses.
- **The repo's Python venv is active.** All `pip install` commands below install into the active interpreter.

## Step-by-step (Windows, recommended row)

### 1. Confirm Python and MATLAB versions match

```powershell
python --version                                    # expect 3.11.x
matlab -batch "ver" | Select-String "MATLAB Version"   # expect 24.1 (R2024a)
```

If those don't match the matrix row above, fix it before continuing.

### 2. Activate your repo venv

```powershell
cd C:\Users\diete\Repositories\UpstreamDrift
.\.venv\Scripts\Activate.ps1
```

(Use whatever your venv path is — `python -c "import sys; print(sys.executable)"` confirms.)

### 3. Install `matlabengine`

```powershell
python -m pip install "matlabengine==24.1.*"
```

This is the modern, recommended path. The package autodetects the installed MATLAB and links to its libraries.

If pip cannot find the right wheel:

```powershell
python -m pip install matlabengine --no-cache-dir --verbose
```

The verbose log will show which Python ABI it is targeting and which MATLAB it is binding to.

### 4. Confirm the install

```powershell
python -c "import matlab.engine; print(matlab.engine.__version__)"
```

Should print something like `24.1.x`. If it prints an `ImportError` mentioning a MATLAB DLL, the version triple is wrong — go back to § 1.

### 5. Smoke-test a real engine

```powershell
python -c "import matlab.engine; eng = matlab.engine.start_matlab('-nodesktop -nosplash'); print(eng.eval('1+1', nargout=1)); eng.quit()"
```

Should print `2.0` after a 10–30 s pause. If it hangs:

- Check Task Manager for a `matlab.exe` process — if it spawned, the issue is the Python ↔ engine handshake (firewall, antivirus, locale).
- Re-run with `-batch` to confirm MATLAB itself starts headlessly: `matlab -nodesktop -nosplash -batch "disp('hi')"`.

### 6. Confirm the Simscape model loads

```powershell
python -c "import matlab.engine; eng = matlab.engine.start_matlab('-nodesktop -nosplash'); eng.load_system('src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/GolfSwing3D_Kinetic.slx', nargout=0); print(eng.bdroot()); eng.quit()"
```

Expected: `GolfSwing3D_Kinetic`. If it fails, the .slx itself has a problem (missing toolbox, Simscape-only block) — that's an upstream issue, not an Option 4 issue.

### 7. Run Option 4's smoke tests

```powershell
python -m pytest src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\tests\test_lifecycle.py -v --timeout=120
```

If this passes, the install is good and you can proceed to [RUNBOOK.md](RUNBOOK.md).

## Common failure modes and fixes

### `ImportError: matlab.engine package not found`

- The `matlabengine` pip package isn't installed in the active Python. Check `python -m pip show matlabengine`.
- The active Python isn't your venv. Confirm: `python -c "import sys; print(sys.executable)"`.

### `ImportError: DLL load failed while importing matlabengine`

- Almost always a version mismatch. Re-check the matrix row in § Version pinning.
- Re-install with `--force-reinstall`: `python -m pip install --force-reinstall "matlabengine==24.1.*"`.

### `RuntimeError: Unable to launch the MATLABWindow application`

- MATLAB tried to open a GUI window. Pass `-nodesktop -nosplash` (the adapter does this automatically) or set `MW_DISABLE_DESKTOP=1` in the environment.

### `License Manager Error -8` / `License Manager Error -39`

- Floating license is exhausted. Either wait for another user to release, or downgrade `pool_size` in [RUNBOOK.md § 5](RUNBOOK.md#5-run-a-parallel-sweep-with-simscapeadapterpool).
- Verify the license file is current: `matlab -batch "license('checkout', 'Simulink'); license('checkout', 'Simscape_Multibody'); disp('ok')"`.

### `engine.start_matlab()` hangs > 60 s

- Antivirus / EDR is scanning `matlab.exe` on first run. Add an exception or increase `startup_timeout_s` in `SimscapeAdapter(startup_timeout_s=180)`.
- A previous engine process didn't exit cleanly and is holding a port. `Get-Process matlab | Stop-Process -Force`, then retry.

### `matlab.engine` works in PowerShell but fails in `pytest`

- `pytest` is being launched from a different Python (often the system one). Confirm: `python -m pytest --version` vs `pytest --version`.
- Always launch with `python -m pytest ...` to use the active venv's pytest.

### `pip install matlabengine` succeeds but `import matlab.engine` raises `ImportError: numpy.core.multiarray failed to import`

- numpy mismatch — `matlabengine` was built against a different numpy ABI. Pin numpy: `python -m pip install "numpy>=1.24,<2.0"`. (numpy 2.0 broke ABI for some MATLAB versions; the matrix row will reflect this.)

## Linux / macOS notes (best-effort)

The steps are the same except:

- The matlabengine wheel must match the OS — make sure pip downloaded `matlabengine-24.1.*-linux_x86_64.whl` not `win_amd64`.
- `matlab` must be on `PATH`. On Linux this is usually `/usr/local/MATLAB/R2024a/bin/matlab`.
- License Manager errors look the same as on Windows but the fix is `flexlm` log inspection: `tail -f /var/log/lmgrd.log` (location varies by license install).
- CI on Linux requires a self-hosted runner with MATLAB installed and licensed. Free GitHub-hosted runners do not have MATLAB.

## CI integration

CI pipelines (`.github/workflows/`) currently do **not** run Option 4 tests. To add Option 4 to CI:

1. Procure a self-hosted runner with MATLAB R2024a installed and the Simulink + Simscape Multibody licenses checked out for the runner's user.
2. Add a workflow job that runs only on that runner: `runs-on: [self-hosted, matlab]`.
3. Use the `requires_matlab` pytest marker (defined in `option4_python_bridge/tests/conftest.py` per [TESTING.md § markers](TESTING.md#markers-and-skip-policy)) so the existing GitHub-hosted runners continue to skip these tests cleanly.

This is **not** in scope for issues #036–#040 — file a follow-up issue when the deployment target is decided.

## Verifying everything before opening a PR

Before you push a PR that touches Option 4:

```powershell
# 1. Lint (must pass for any PR per CLAUDE.md)
python -m ruff check src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\
python -m ruff format --check src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\

# 2. Unit tests that don't need MATLAB (must pass everywhere)
python -m pytest src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\tests -v -m "not requires_matlab"

# 3. Integration tests that need MATLAB (must pass on your dev box)
python -m pytest src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\tests -v -m "requires_matlab" --timeout=180

# 4. File-size budget
python scripts\ci\check_file_size_budget.py
```

All four must be green before you mark the PR ready for review.
