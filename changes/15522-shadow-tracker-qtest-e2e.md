---
issue: 15522
summary: "Headless PyQt6 QTest launch-and-interact E2E for the Shadow Tracker review workbench (#15522)"
---

- Add `tests/tools/shadow_tracker/test_shadow_tracker_qtest_e2e.py`: launches `python -m src.tools.shadow_tracker` offscreen and drives the workbench with real `QTest` mouse and key events (frame scrubbing, worst-frame triage, honest Auto-Fit refusal, save/open/export through patched dialogs).
