# F-UD-029

**Windows new-console launch builds a `cmd /c "..."` string with unescaped cmd metacharacters**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | security | medium     | S      | proposed |

**Locations:** `src/launchers/launcher_process_manager.py:148`, `src/launchers/launcher_process_manager.py:558`

## Finding

- Severity: Low - Category: security
- Confidence: medium
- Location: src/launchers/launcher_process_manager.py:148-151; src/launchers/launcher_process_manager.py:558-566
- Evidence:
  ```python
  inner = subprocess.list2cmdline(args)
  return f'cmd /c "{inner}"'
  ```
- Failure scenario: `list2cmdline` quotes for `CreateProcess` argument splitting, not for `cmd.exe` parsing; a model or script
  path containing `&`, `|`, `^` or `%VAR%` (user-chosen directories, OneDrive folders with `&`) is interpreted by `cmd`. Not
  remotely exploitable, but it is the one place the launcher runs a shell string after `managed_popen` forbids `shell=True`.
- Why a newer engineer misses this: `list2cmdline` has "cmdline" in its name.
- Proposed change: escape `^&|<>%` with `^` in each argument, or avoid `cmd` by using `CREATE_NEW_CONSOLE` directly on the
  Python executable with `-c "import runpy; ...; input()"` for keep-open.
- Verification: launch a script from a directory named `a&b` on Windows CI.
- Effort: S

## Draft Pull Request

**Title:** `fix(launcher): escape cmd metacharacters in Windows console launch`

### Why This Matters (The Pitch)

On Windows the launcher builds cmd /c with the output of subprocess.list2cmdline, which quotes for CreateProcess, not for cmd.exe. A script or model path containing &, |, ^ or %VAR%, such as a OneDrive folder with an ampersand, gets interpreted by cmd. It is not remotely exploitable, but it is the one place that runs a shell string after managed_popen forbids shell=True. Escaping those characters, or avoiding cmd entirely, is a small hardening fix.

### Proposed PR Body

## Problem

`src/launchers/launcher_process_manager.py:148-151` returns `f'cmd /c "{inner}"'` with `inner = subprocess.list2cmdline(args)` (used at `:558-566`); `cmd.exe` metacharacters in paths are not escaped. Confidence is medium.

## Change

- [ ] Escape `^&|<>%` with `^` in each argument, or use `CREATE_NEW_CONSOLE` on the Python executable directly

## Verification

Launch a script from a directory named `a&b` on Windows CI.

## Risk and Rollback

Low: affects only the Windows new-console path. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-029

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
