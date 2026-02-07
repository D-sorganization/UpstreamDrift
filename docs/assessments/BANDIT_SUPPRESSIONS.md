# Bandit Security Suppressions

This document tracks all Bandit security findings that have been reviewed and suppressed with justification.

## Summary

| Issue                          | Count | Action                                            |
| ------------------------------ | ----- | ------------------------------------------------- |
| B108 - Hardcoded tmp directory | 6     | Suppressed - Docker X11 sockets and test fixtures |
| B104 - Bind all interfaces     | 1     | False positive - environment variable check       |
| B314 - XML parsing             | 2     | Suppressed - test code with known data            |
| B310 - urllib.urlopen          | 8     | Reviewed - HTTPS GitHub API only                  |
| B608 - SQL injection           | 1     | Already suppressed - parameterized internally     |

## Detailed Suppressions

### B108: Hardcoded Temporary Directory

**Location:** Multiple files using `/tmp/.X11-unix`

**Files:**

- `src/engines/physics_engines/mujoco/docker/gui/deepmind_control_suite_MuJoCo_GUI.py:1106`
- `src/engines/physics_engines/mujoco/python/humanoid_launcher.py:939`
- `src/launchers/_archive/golf_launcher_pre_refactor_ce85e6ec.py:2905`

**Justification:** These paths are Docker volume mounts for X11 forwarding. This is the standard, required path for Unix domain sockets used by X11. Not a security risk as:

1. The path is well-known and documented
2. Used only for display forwarding in Docker containers
3. Not used for storing sensitive data

**Action:** No change needed - standard Docker practice.

---

**Location:** Test fixtures using `/tmp/pytest` and `/tmp/test.c3d`

**Files:**

- `src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/c3d_reader.py:740`
- `src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/tests/test_headless_gui.py:32`

**Justification:** These are:

1. Detection logic checking if running in pytest context
2. Test fixtures with hardcoded paths for unit testing

**Action:** No change needed - test infrastructure.

---

### B104: Binding to All Interfaces

**Location:** `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/meshcat_adapter.py:48`

**Code:**

```python
if os.environ.get("MESHCAT_HOST") == "0.0.0.0":
```

**Justification:** This is checking an environment variable value, not binding a socket. False positive.

**Action:** No change needed - false positive.

---

### B314: XML Parsing with ElementTree

**Location:** `src/engines/physics_engines/mujoco/python/tests/test_urdf_io.py:183, 201`

**Justification:** Test code parsing known, hardcoded XML strings for unit testing. Not processing untrusted external data.

**Action:** No change needed - test code with known data.

---

### B310: urllib.urlopen

**Location:** `src/tools/model_generation/library/model_library.py` and `repository.py`

**Justification:** These functions download URDF models from the GitHub API:

1. URLs are constructed from known GitHub domains only
2. Only `https://` URLs are used
3. Used for downloading trusted open-source robotics models

**Recommendation:** Add URL scheme validation to ensure only HTTPS is used.

**Action:** Added URL validation in PR (see below).

---

### B608: SQL Injection

**Location:** `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/recording_library.py:629`

**Code:**

```python
cursor.execute(
    f"SELECT DISTINCT {field} FROM recordings WHERE {field} != ''"
)  # nosec B608
```

**Justification:** Already suppressed with `# nosec B608`. The `field` parameter is:

1. Validated against a whitelist of known column names
2. Not user-controllable in the exposed API
3. Internal helper function for dropdown population

**Action:** Existing suppression is appropriate.

---

## Validation

Run Bandit to verify suppressions:

```bash
bandit -r src -ll -ii
```

Expected: 0 high-severity issues, <10 medium-severity issues (all documented above).

---

Last updated: 2026-02-03
