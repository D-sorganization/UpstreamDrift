---
issue: 12065
summary: "Hold the native C3D/OpenSim runtime boundary inside the pytest process: tests/helpers/native_runtime_boundary skips a test with an explicit no-evidence reason before ezc3d loads beside OpenSim (require_ezc3d, ezc3d_installed) or OpenSim loads beside ezc3d (requires_opensim marker), reusing the diagnostic's public RUNTIME_ROLES and the Tools DLL probe; validated against real OpenSim 4.6 + ezc3d 1.7.2 where both import orders fail"
---
