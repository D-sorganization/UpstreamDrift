# Security Audit Findings (MEDIUM Severity)

**Date:** 2026-01-20
**Reporter:** SENTINEL (Jules)
**Priority:** Medium
**Labels:** security, jules:sentinel

## Summary
The following medium severity security issues were identified during the automated security audit.

## Findings

### 1. XML Parsing Vulnerabilities (B314)
**Issue:** Using `xml.etree.ElementTree.fromstring` or `parse` to process untrusted XML data is vulnerable to XML attacks (e.g., entity expansion).
**Remediation:** Replace with `defusedxml.ElementTree` or ensure `defusedxml.defuse_stdlib()` is called.

**Affected Files:**
- `engines/physics_engines/mujoco/python/tests/test_urdf_io.py`: Lines 186, 204
- `shared/python/myoconverter_integration.py`: Line 190
- `tests/test_urdf_generator.py`: Lines 131, 196
- `tests/test_urdf_tools.py`: Lines 51, 78
- `tests/unit/test_physical_constants_xml.py`: Lines 25, 55, 74, 94, 128, 164, 179, 195, 206, 218, 233, 261
- `tests/unit/test_urdf_io.py`: Line 115
- `tools/urdf_generator/mujoco_viewer.py`: Lines 69, 431

### 2. Insecure Temporary File Usage (B108/B377)
**Issue:** Probable insecure usage of temp file/directory.
**Remediation:** Use `tempfile` module securely (e.g., `tempfile.NamedTemporaryFile` with appropriate permissions).

**Affected Files:**
- `engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/c3d_reader.py`: Line 740
- `engines/Simscape_Multibody_Models/3D_Golf_Model/python/tests/test_headless_gui.py`: Line 32
- `engines/physics_engines/mujoco/docker/gui/deepmind_control_suite_MuJoCo_GUI.py`: Line 1106
- `engines/physics_engines/mujoco/python/humanoid_launcher.py`: Line 841
- `launchers/golf_launcher.py`: Line 2892
- `tests/integration/test_engine_manager_coverage.py`: Lines 32, 35
- `tests/integration/test_phase1_security_integration.py`: Line 80
- `tests/unit/test_common_utils.py`: Lines 24, 30, 36, 39, 44
- `tests/unit/test_openpose_estimator.py`: Lines 49, 65
- `tests/unit/test_opensim_physics_engine.py`: Lines 65, 72

### 3. Binding to All Interfaces (B104)
**Issue:** Possible binding to all interfaces (`0.0.0.0`). This may expose the service to the network unexpectedly.
**Remediation:** Bind to localhost (`127.0.0.1`) unless external access is explicitly required and secured.

**Affected Files:**
- `api/server.py`: Line 758
- `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/meshcat_adapter.py`: Line 47
- `start_api_server.py`: Line 94

### 4. URL Open Scheme (B310)
**Issue:** Audit url open for permitted schemes. `urllib.request.urlopen` might support `file://` or other schemes that could lead to local file inclusion if user input is allowed.
**Remediation:** Validate the URL scheme (e.g., ensure it starts with `http://` or `https://`) before calling `urlopen`.

**Affected Files:**
- `shared/python/standard_models.py`: Line 151
- `tools/urdf_generator/model_library.py`: Line 233

### 5. Potential SQL Injection (B608)
**Issue:** Possible SQL injection vector through string-based query construction.
**Remediation:** Use parameterized queries.

**Affected Files:**
- `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/recording_library.py`: Line 628

### 6. Subprocess with Shell=True (B604)
**Issue:** Function call with `shell=True` parameter identified.
**Remediation:** Set `shell=False` and pass arguments as a list.

**Affected Files:**
- `tests/integration/test_phase1_security_integration.py`: Line 124
- `tests/unit/test_secure_subprocess.py`: Line 117

### 7. Use of exec (B102)
**Issue:** Use of `exec` detected.
**Remediation:** Avoid `exec` if possible. If necessary, strictly validate input.

**Affected Files:**
- `tests/test_pinocchio_ecosystem.py`: Line 64

### 8. Permissive Chmod (B103)
**Issue:** Chmod setting a permissive mask (0o755).
**Remediation:** Restrict permissions (e.g., 0o600 or 0o644) if execution is not required for others.

**Affected Files:**
- `tests/integration/test_phase1_security_integration.py`: Line 51
