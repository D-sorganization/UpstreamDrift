---
issue: 15522
summary: "Exercise Tour Matching Viewer archive loading, model rendering, playback, error recovery and close cleanup with native headless QTest."
---

Closing an embedded viewer now pauses its playback timer while retaining the
loaded scene for reuse. Synthetic replay fixtures exercise real archive loading
and model forward kinematics without private data or physics-engine mocks.
