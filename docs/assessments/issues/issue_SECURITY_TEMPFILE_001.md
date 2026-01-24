---
title: "Security: Insecure Temporary File Usage"
labels: ["security", "jules:sentinel"]
---

## Description
Code uses insecure methods for creating temporary files or directories, which can be vulnerable to symlink attacks or information disclosure.

**Findings:**
- `engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/c3d_reader.py:740` (B108)
- `engines/physics_engines/mujoco/docker/gui/deepmind_control_suite_MuJoCo_GUI.py:1106` (B108)
- `engines/physics_engines/mujoco/python/humanoid_launcher.py:841` (B108)
- `launchers/golf_launcher.py:2892` (B108)
- `tests/integration/test_engine_manager_coverage.py:32` (B108)
- `tests/integration/test_phase1_security_integration.py:80` (B108)
- `tests/unit/test_common_utils.py:24` (B108)
- `tests/unit/test_openpose_estimator.py:49` (B108)
- `tests/unit/test_opensim_physics_engine.py:65` (B108)

## Remediation
Use the `tempfile` module (e.g., `tempfile.NamedTemporaryFile`, `tempfile.mkstemp`) which creates files securely.
