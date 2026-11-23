---
issue: 15522
summary: "Add headless QTest receipt-to-load-heatmap E2E coverage for Spine Viewer and clear stale study results after rejected runs."
dl_state: "in_review"
next_step: "Independent review and CI disposition; owner requires an unarmed PR."
branch: "codex-desk/15522-spine-viewer-qtest"
paths: "tests/tools/spine_viewer/test_gui_e2e_15522.py, src/tools/spine_viewer/gui.py"
---

Native offscreen PyQt6 QTest drives Open, quantity selection, invalid Run and recovery. Only the OS file picker is replaced; verification, load-model conversion and plotting are real. Invented synthetic receipts do not qualify an OpenSim simulation. No merge/arm or TURNOVER changes.
