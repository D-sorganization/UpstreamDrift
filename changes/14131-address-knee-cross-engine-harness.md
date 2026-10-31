---
issue: 14131
summary: "Add cross-engine address knee harness contract test in test_address_knee_flexion.py."
---

Wire the committed-receipt cross-engine comparator into
``test_address_knee_flexion.py`` (#14131 harness scope). Refs #12684 and #12692;
does not close the strong parent or regenerate canonical receipts.
