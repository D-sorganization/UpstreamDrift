---
issue: 15522
summary: "Headless QTest launch-and-interact E2E tests for the Ground Reaction Plots GUI (Refs #15522)"
dl_state: "in_review"
next_step: "Independent acceptance at the new head."
branch: "test/15522-ground-reaction-qtest-e2e"
---

- Add `tests/tools/ground_reaction_plots/test_gui_e2e_15522.py`: launches the tool through `main()` and drives the real "Open Ground Reaction JSON..." and "Import Plate Trial..." buttons with `QTest` clicks (file dialog stubbed), asserting status line, unqualified-measurement banner and the six-panel figure (Refs #15522).
- Test-only change; no production code is modified.
