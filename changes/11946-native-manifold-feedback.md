---
issue: 11946
summary: "Connect F02 feedback to native floating-root manifold state and independently replay post-limit torques."
branch: "feat/f02-native-manifold-11946"
---

Run F02 TVLQR against an admitted native MuJoCo floating-root two-hinge
fixture with quaternion-aware error and physical inverse mass. Save only
executed post-limit motor torques in the Tools bundle and reproduce the
complete native integration-state trajectory in a fresh replay.
