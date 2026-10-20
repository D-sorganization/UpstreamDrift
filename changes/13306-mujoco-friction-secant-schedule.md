---
issue: 13306
summary: "MuJoCo topology grip schedules each node friction row's solreffriction time to the shared law's secant slip gain every step (friction_secant_gain); ten_finger 0.04 s hold net_force 2.866% -> 0.413% RMS, ten_finger 0.10 s hold and swing pass all ten parity quantities, overlap/interlock per-hand forces 10-18% -> under 1% RMS"
dl_state: "in_review"
next_step: "Review and merge after #13442; overlap/interlock 0.10 s trail-hand deflection (4.60% hold, 3.39% swing RMS) stays a strict xfail; OpenSim ten_finger 0.04 s gaps remain"
branch: "fix/13306-mujoco-friction-law"
---
