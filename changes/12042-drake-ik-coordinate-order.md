---
issue: 12042
summary: "Fix Drake full-body IK/model spec-vs-plant coordinate order: the torso twist was solved under the SpineInputX bound; capture-A Drake IK 47.2 -> 34.6 mm, FD 382 -> 60.1 mm, twist now matches MuJoCo"
branch: "claude/drake-ik-twist"
paths: "src/engines/physics_engines/drake/python/full_body_model.py,src/engines/physics_engines/drake/python/full_body_ik.py,tests/unit/motion_matching/test_drake_coordinate_order.py"
---
