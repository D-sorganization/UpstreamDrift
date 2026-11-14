---
issue: 14310
summary: "Headless Qt/matplotlib teardown after each test and MuJoCo offscreen renderer release (#14310)"
---

- Add `tests/helpers/headless_gui_teardown.py` and call it from the root `pytest_runtest_teardown` hook so xdist workers drain pending matplotlib idle draws and Qt events after GUI tests.
- Release `MuJoCoOffscreenRenderer` GL resources via `release_renderer` on reload, `clear`, and widget `closeEvent`; tighten `test_drop_refreshes_the_editor_window_3d_preview` teardown.