## 2025-02-12 - MuJoCo Python Loop Optimization
**Learning:** Manual iteration over MuJoCo bodies in Python to compute system-wide quantities (like COM velocity) is extremely slow due to Python overhead and repeated C-API calls (`mj_jacBodyCom`).
**Action:** Always check for native MuJoCo functions (like `mj_subtreeVel`) which perform these calculations in C. The speedup can be massive (>50x).
