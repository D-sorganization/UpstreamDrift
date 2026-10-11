---
issue: 11719
summary: "GCV-13: the OpenSim native export now hides the decorative ball after impact. The worker hangs the sphere on a tiny render-only body on a slider joint from ground and slides it 1 km out of view once the ball is hidden, because the simbody visualizer redraws body poses every frame but caches fixed geometry. The OpenSim ball limitation note is removed from the export result."
title: "GCV-13: OpenSim Ball Hidden After Impact"
branch: "claude/gcv-13-opensim-ball-hide"
---
