---
issue: 13896
summary: "Render workers restore the interpreter LIBDIR on LD_LIBRARY_PATH (opensim's Linux __init__ overwrites it with PATH) and lift render PYTHONPATH uses shared worker_env(), fixing the requires_opensim _ctypes ABI failure"
---
