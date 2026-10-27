---
issue: 13899
summary: "Render workers put the interpreter's <base_prefix>/lib first on LD_LIBRARY_PATH, because opensim's Linux __init__ overwrites it with PATH; this fixes the requires_opensim _ctypes ABI failure"
---
