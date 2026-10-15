"""Spec file for the NSX-55 Simscape planar test rig (#12583).

The MATLAB exporter (``nsx_simscape_closed_loop_export.m``) builds a Simscape
Multibody model of the NSX-00 planar two-arm golfer from this JSON and probes
it.  The spec holds only *inputs* (rig parameters, states, declared inertia and
topology); every dynamic quantity in the resulting export comes from Simscape.
This is a test rig for the exporter and loader, never the golf model.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from src.shared.python.loop_nullspace.fixtures import (
    PlanarGolferConfig,
    load_golden,
    planar_golfer_snapshot,
)
from src.shared.python.loop_nullspace.planar_model import PlanarGolferParams
from src.shared.python.loop_nullspace.simscape_export import (
    export_from_snapshot,
    export_to_dict,
)

__all__ = ["planar_rig_spec"]

NOTE = (
    "Planar two-arm golfer rig (NSX-00) built as an open tree in Simscape "
    "Multibody; M, h, J_c, Jdot_v and the grip origin were probed from Simscape "
    "by forward dynamics.  Not the golf model."
)


def planar_rig_spec(config: PlanarGolferConfig | None = None) -> dict[str, Any]:
    """Return the JSON-serialisable rig spec for ``config`` (default: generic).

    Postcondition: ``rig`` holds the ten NSX-00 parameters, ``states`` the
    requested ``(q, v, a)``, ``topology`` and ``inertias`` the planar fixture's.
    """
    case = planar_golfer_snapshot(config)
    params = PlanarGolferParams.from_golden(load_golden("planar_two_arm_full"))
    layout = export_to_dict(export_from_snapshot(case.snapshot))
    snap = case.snapshot
    return {
        "rig": dataclasses.asdict(params),
        "closure_rows": [1, 2],
        "topology": layout["topology"],
        "inertias": layout["inertias"],
        "states": [{"q": snap.q.tolist(), "v": snap.v.tolist(), "a": snap.a.tolist()}],
        "note": NOTE,
    }
