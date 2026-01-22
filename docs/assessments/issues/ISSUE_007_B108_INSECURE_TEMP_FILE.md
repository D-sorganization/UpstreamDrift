# Security: Insecure Temporary File Usage (B108)

**Labels:** security, jules:sentinel

## Description
Probable insecure usage of temp file/directory. This usually involves creating temporary files with predictable names or insecure permissions, which can lead to race conditions or information disclosure.

## Locations
- `engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/c3d_reader.py:740`
- `engines/physics_engines/mujoco/docker/gui/deepmind_control_suite_MuJoCo_GUI.py:1106`
- `engines/physics_engines/mujoco/python/humanoid_launcher.py:841`
- `launchers/golf_launcher.py:2892`

## Remediation
Use `tempfile.NamedTemporaryFile` or `tempfile.TemporaryDirectory` which handle secure creation automatically. Avoid using the global temporary directory directly with predictable filenames.
