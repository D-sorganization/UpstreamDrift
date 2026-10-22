---
issue: 13626
summary: "Prevent duplicate Xvfb startup in the motion-matching smoke command."
---

# Motion-Matching Smoke Display Startup

Disable automatic pytest-xvfb startup for the motion-matching smoke command so
it does not conflict with the script's EGL/OSMesa selection or explicit
`xvfb-run` fallback. A shell-execution regression test covers each display
route, best-effort behavior when no backend or Xvfb is available, forwarded
arguments, and pytest exit-status propagation.
