---
issue: 15522
summary: "Add native headless QTest coverage for Starting-Pose Matcher transforms, export, rotation locks and CSV playback; stop standalone playback on close."
dl_state: "in_review"
next_step: "Independent review and CI disposition; leave PR unarmed per owner."
branch: "codex-desk-starting-pose/15522-qtest-e2e"
paths: "src/tools/starting_pose_matcher/gui.py,tests/tools/starting_pose_matcher/test_gui_e2e_15522.py"
---

Refs #15522. TDD: close-playback assertion failed on unchanged origin/main. Real FK skeletons, CSV parser, Qt timer and JSON export; only private capture discovery and file pickers replaced. No physics qualification claim. No TURNOVER edits or PR arm/merge.
