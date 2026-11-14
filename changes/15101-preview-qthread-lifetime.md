---
issue: 15101
summary: Join or retain retiring BinderThread and CameraWorker instances until Qt finishes thread teardown so Capture Rig preview never aborts in ~QThread.
---

Join retiring `BinderThread` and `CameraWorker` instances before dropping Python references in `PreviewPanel`, retain any thread that outlives its join window until `QThread.finished` fires on the GUI thread, and stop active preview threads in `PreviewPanel.closeEvent`.
