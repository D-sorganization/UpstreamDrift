---
issue: 15522
summary: "Headless QTest E2E for the Config Setup Wizard widget (#15522)"
dl_state: "in_review"
next_step: "Independent acceptance review at the PR head; epic #15522 stays open for other GUI slices."
branch: "test/15522-claude-qtest-e2e-d"
---

- Add `tests/launchers/test_config_setup_wizard_gui_e2e_15522.py`: launches the real `ConfigSetupWizardWidget` offscreen and drives Back / Validate / Next with real key and mouse events (#15522).
- Cover the blocked first step, partial-config validation, the full walkthrough to Review and back, invalid-JSON and non-object recovery, malformed gravity, and `cleanup()` (#15522).
- Drop the stale `# pragma: no cover` from `ConfigSetupWizardWidget` now that the GUI smoke exists (#15522).
