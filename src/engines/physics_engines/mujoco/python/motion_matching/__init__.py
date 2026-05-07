"""MuJoCo motion-matching package (Python side).

Houses the forward simulator (``simulate.py``), torque driver, and the
visualization sub-package (``viz``). See ``MUJOCO_PARITY_SPEC.md``
§2 for the full surface.

This file intentionally avoids re-exporting heavy modules so that
``import ...mujoco.python.motion_matching`` stays cheap; callers should
reach into the explicit submodules they need.
"""

from __future__ import annotations

__all__: list[str] = []
