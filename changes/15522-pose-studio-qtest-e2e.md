---
issue: 15522
summary: "Exercise Pose Studio with headless QTest and rebind the viewport when switching physics engines."
---

Native Qt keyboard and mouse events cover joint editing, undo/redo, engine
selection and an engine-native pose file round trip. The viewport now follows
the selected kinematics service immediately. Optional engine fallback remains
visible; these tests qualify GUI wiring and pose interchange, not dynamics.
