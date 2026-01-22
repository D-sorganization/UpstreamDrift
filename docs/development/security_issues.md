# Security Issues Identified

## MEDIUM Severity Findings

### 1. XML Injection Risk (B314)
**Description**: Usage of `xml.etree.ElementTree.fromstring` or `parse` detected. This is vulnerable to XML Entity Expansion attacks.
**Remediation**: Replace with `defusedxml.ElementTree` or ensure `defusedxml.defuse_stdlib()` is called.
**Affected Files**:
- `engines/physics_engines/mujoco/python/tests/test_urdf_io.py`: 186, 204
- `shared/python/myoconverter_integration.py`: 190
- `tests/test_urdf_generator.py`: 131, 196
- `tests/test_urdf_tools.py`: 51, 78
- `tests/unit/test_physical_constants_xml.py`: Multiple lines
- `tests/unit/test_urdf_io.py`: 115
- `tools/urdf_generator/mujoco_viewer.py`: 69, 431

### 2. Insecure Temporary File Usage (B108)
**Description**: Probable insecure usage of temp file/directory. This can lead to race conditions or information disclosure.
**Remediation**: Use `tempfile.NamedTemporaryFile` with appropriate permissions or `tempfile.mkstemp`.
**Affected Files**:
- `engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/c3d_reader.py`: 740
- `engines/Simscape_Multibody_Models/3D_Golf_Model/python/tests/test_headless_gui.py`: 32
- `engines/physics_engines/mujoco/docker/gui/deepmind_control_suite_MuJoCo_GUI.py`: 1106
- `engines/physics_engines/mujoco/python/humanoid_launcher.py`: 841
- `launchers/golf_launcher.py`: 2892
- `tests/integration/test_engine_manager_coverage.py`: 32, 35
- `tests/integration/test_phase1_security_integration.py`: 80
- `tests/unit/test_common_utils.py`: Multiple lines
- `tests/unit/test_openpose_estimator.py`: 49, 65
- `tests/unit/test_opensim_physics_engine.py`: 65, 72

### 3. Binding to All Interfaces (B104)
**Description**: Potential binding to all interfaces (0.0.0.0). This exposes the service to the network.
**Remediation**: Bind to localhost (127.0.0.1) unless external access is explicitly required and secured.
**Affected Files**:
- `api/server.py`: 758
- `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/meshcat_adapter.py`: 47
- `start_api_server.py`: 94

### 4. SQL Injection Risk (B608)
**Description**: Possible SQL injection vector through string-based query construction.
**Remediation**: Use parameterized queries.
**Affected Files**:
- `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/recording_library.py`: 628

### 5. URL Open Risk (B310)
**Description**: Audit url open for permitted schemes. `urllib` might allow file access.
**Remediation**: Validate URL schemes (http/https) before opening.
**Affected Files**:
- `shared/python/standard_models.py`: 151
- `tools/urdf_generator/model_library.py`: 233

### 6. Subprocess Shell=True (B604)
**Description**: Function call with `shell=True` parameter identified. This is a command injection risk.
**Remediation**: Set `shell=False` and pass arguments as a list.
**Affected Files**:
- `tests/integration/test_phase1_security_integration.py`: 124
- `tests/unit/test_secure_subprocess.py`: 117

### 7. Exec Usage (B102)
**Description**: Use of `exec` detected. This allows arbitrary code execution.
**Remediation**: Avoid dynamic code execution if possible.
**Affected Files**:
- `tests/test_pinocchio_ecosystem.py`: 64

### 8. Chmod Permissive Mask (B103)
**Description**: Chmod setting a permissive mask (e.g., 0o777).
**Remediation**: Use restrictive permissions (e.g., 0o600 or 0o644).
**Affected Files**:
- `tests/integration/test_phase1_security_integration.py`: 51
