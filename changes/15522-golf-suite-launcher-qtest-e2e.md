---
issue: 15522
summary: "Headless QTest launch-and-click E2E for the Golf Suite Launcher (MuJoCo/Drake/Pinocchio buttons)."
---

Test-only. Drives the real `UpstreamDriftLauncher` window with `QTest` mouse
clicks; only `subprocess.Popen` and the modal error dialog are stubbed.
