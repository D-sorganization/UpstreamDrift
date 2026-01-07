# Sentinel's Journal 🛡️

## 2025-12-11 - Prevent Code Injection in MATLAB Subprocess Calls
**Vulnerability:** Found a Code Injection vulnerability in `matlab_quality_check.py` where a user-controlled file path containing single quotes (`'`) was inserted directly into a MATLAB command string passed to `subprocess.run`. This allowed breaking out of the string literal and executing arbitrary MATLAB code (and potentially shell commands via `system()`).
**Learning:** Even when using `subprocess.run(..., shell=False)`, passing arguments to an interpreter (like `matlab -batch "..."`) requires understanding that interpreter's syntax and escaping rules. The shell isn't the only place injection can happen; the target program's argument parser is also a surface.
**Prevention:** Always sanitize or escape variable content before embedding it into command strings for other interpreters. For MATLAB, replace `'` with `''`.
