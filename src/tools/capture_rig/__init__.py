"""Capture Rig: camera controller, recorder and pose-overlay player (#9619).

Desktop front end over ``python -m src.motion_capture.rig``. The pure parts
(:mod:`.commands`, :mod:`.session`, :mod:`.overlay`, :mod:`.player`,
:mod:`.layout_model`, :mod:`.layout_presets`) import no Qt and are what the
tests exercise; :mod:`.gui` arranges them.
"""

from src.shared.python.launcher_embed import register_embeddable_tool

from ._embed_adapter import CaptureRigAdapter

register_embeddable_tool(CaptureRigAdapter())

__all__ = ["CaptureRigAdapter"]
