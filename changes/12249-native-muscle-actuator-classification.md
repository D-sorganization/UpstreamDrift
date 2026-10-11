---
issue: 12249
summary: "Use MuJoCo's named muscle enum for native MyoSuite actuator classification."
dl_state: "in_review"
next_step: "Root will publish the reviewed branch; retain physiology and replay qualification gates."
title: "MyoSuite Native Muscle Actuator Classification"
branch: "fix/12249-native-muscle-actuator-enum"
paths: "src/engines/physics_engines/myosuite/python/muscle_analysis.py;tests/unit/engines/myosuite/test_muscle_analysis.py;manuals/upstreamdrift/chapters/31-myosuite-native-excitation-replay.qmd;docs/development/feedback_controls/F09Q-MYOSUITE-ACTUATOR-CLASSIFICATION-TURNOVER.md"
---

Commit fc17a35c8f72b7d42e79a8c73fc1aebb08e3bba5; native MuJoCo 3.8.0 mixed muscle/filter/motor test passes without Gym. See the F09q turnover for validation; this fixes dynamic-type classification only.
