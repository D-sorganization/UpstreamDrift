"""Decorative address-ball MJCF geom, shared by the MuJoCo-based workers.

GCV-13 slice 3 (#11719): the MuJoCo appearance layer
(``model_appearance/ball.py`` via ``visual_layer._attach_decorative_ball``)
can already draw a ``visual_ball`` geom, but only from the exporter's
*static reference pose*, not the swing's actual resolved address frame
(:func:`src.tools.native_viewer_export.ball.resolve_address_ball`). Both
MuJoCo-based render workers (``mujoco_worker``, ``myosuite_worker``) build
their own MJCF and call :func:`set_decorative_ball` after it is built: this
drops whatever the static fallback drew -- it is never trusted here -- and,
when a position is given, attaches the one already-resolved ball instead.
No placement math lives here, only the MJCF plumbing that would otherwise be
written twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.shared.python.model_appearance.ball import BALL_RADIUS_M

_BALL_NAME = "visual_ball"
_BALL_RGBA = "0.95 0.95 0.95 1"


def set_decorative_ball(root: Any, position_m: Sequence[float] | None) -> None:
    """Replace ``root``'s ``visual_ball`` worldbody geom with one at ``position_m``.

    ``root`` is an already-parsed MJCF document's root element. Any existing
    ``visual_ball`` geom is removed first; when ``position_m`` is ``None``
    nothing more happens, so the scene draws no ball. The attached geom is
    massless and non-colliding (``contype="0" conaffinity="0" mass="0"``),
    so it changes no physics -- visual only, like the club and head meshes.

    Raises ``ValueError`` when ``root`` has no ``worldbody`` element.
    """
    world = root.find("worldbody")
    if world is None:
        raise ValueError("MJCF document has no worldbody")
    for geom in list(world.findall("geom")):
        if geom.get("name") == _BALL_NAME:
            world.remove(geom)
    if position_m is None:
        return
    x, y, z = (float(v) for v in position_m)
    geom = world.makeelement(
        "geom",
        {
            "name": _BALL_NAME,
            "type": "sphere",
            "pos": f"{x:.6g} {y:.6g} {z:.6g}",
            "size": f"{BALL_RADIUS_M:.6g}",
            "rgba": _BALL_RGBA,
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
            "mass": "0",
        },
    )
    world.append(geom)
