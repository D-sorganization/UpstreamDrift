---
issue: 15522
summary: "Cover the Training Controller dashboard with headless QTest E2E (submit, run, pause/resume/cancel, close) and surface submit-dialog validation and scheduler errors in the dialog instead of only the log."
---

Real `MainWindow` and `SubmitDialog` are driven with QTest against the real
controller, scheduler, in-process driver and job registry at desktop and phone
viewports. A blank entry point or a framework with no registered runner now
shows a visible error in the dialog (previously logged only) and the dialog
stays usable. This does not certify training numerics or native rendering.
