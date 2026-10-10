---
issue: 12018
summary: "Keep F09 replay contracts on one private Tools module identity"
---

Route F09 runtime replay-contract access through the pinned private package
loader so execution and observation consumers share the same T01 class and
enum objects without extending `sidekick.lab`.
