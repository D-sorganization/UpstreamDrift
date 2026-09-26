"""Argparse builders for the coaching, model and variant rig commands.

Split from ``motion_capture.rig.__main__`` to keep the operator CLI under the
file-size budget; these functions only declare arguments and hold no logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _add_coaching_parsers(sub: Any) -> None:
    """Printable boards, annotated clips and take comparison (#9679-#9681)."""
    brd = sub.add_parser("board", help="write a printable ChArUco board image")
    brd.add_argument("--board", default="charuco:7x5:0.04:0.03")
    brd.add_argument("--width", type=int, default=2100, help="pixels (A4 @ 254 dpi)")
    brd.add_argument("--out", type=Path, required=True)
    clp = sub.add_parser("clip", help="trimmed clip with overlay and slow motion")
    clp.add_argument("--session", type=Path, required=True)
    clp.add_argument("--view", required=True)
    clp.add_argument(
        "--from", dest="start", default="address-30", metavar="EVENT|FRAME"
    )
    clp.add_argument("--to", dest="end", default="finish+30", metavar="EVENT|FRAME")
    clp.add_argument("--speed", type=float, default=0.25, help="1 = real time")
    clp.add_argument("--set", default=None, help="observation set for the overlay")
    clp.add_argument("--out", type=Path, required=True)
    cmt = sub.add_parser("compare-takes", help="two takes side by side on an event")
    cmt.add_argument("--session", type=Path, required=True)
    cmt.add_argument("--view", required=True)
    cmt.add_argument("--other-session", type=Path, required=True)
    cmt.add_argument("--other-view", required=True)
    cmt.add_argument(
        "--align", default="top", choices=("address", "top", "peak", "finish")
    )
    cmt.add_argument("--speed", type=float, default=0.5)
    cmt.add_argument("--out", type=Path, required=True)
    mp = sub.add_parser("multipicture", help="composite multiview video via a layout")
    mp.add_argument("--session", type=Path, required=True)
    mp.add_argument("--layout", required=True, help="preset, saved name or JSON path")
    mp.add_argument("--variants", nargs="*", default=[], help="drawn on overlay tiles")
    mp.add_argument("--set", default="observations", help="observation set to draw")
    mp.add_argument("--from", dest="start", type=int, default=0, metavar="FRAME")
    mp.add_argument("--to", dest="stop", type=int, default=None, metavar="FRAME")
    mp.add_argument("--speed", type=float, default=1.0, help="1 = real time")
    mp.add_argument("--size", default=None, metavar="WxH", help="canvas pixels")
    mp.add_argument("--out", type=Path, required=True)


def _add_model_parsers(sub: Any) -> None:
    """Articulated-model fit, comparison and kinetics (#9709)."""
    fm = sub.add_parser(
        "fit-model", help="articulated golfer (scapula) fit, continuous"
    )
    fm.add_argument("--session", type=Path, required=True)
    fm.add_argument(
        "--sigma-accel",
        type=float,
        default=300.0,
        help="acceleration prior on every joint angle, rad/s^2 (smaller = stiffer)",
    )
    fm.add_argument(
        "--max-velocity",
        type=float,
        default=25.0,
        help="report joint-angle speeds above this, rad/s",
    )
    fm.add_argument("--sigma-landmark", type=float, default=0.01, help="metres")
    fm.add_argument("--model", default="golfer", help="registered model name")
    fm.add_argument(
        "--fit-lengths",
        action="store_true",
        help="learn the model's learnable segment lengths from the data",
    )
    fm.add_argument("--max-iterations", type=int, default=60, help="solver budget")
    _add_variant_args(fm, observations=True)
    fm.add_argument(
        "--from-views",
        default="",
        metavar="a,b",
        help="image-space fit to these views' 2-D keypoints (1..N), no triangulation",
    )
    fm.add_argument(
        "--cameras-from",
        default="",
        metavar="VARIANT",
        help="variant whose reconstruction supplies the cameras for --from-views",
    )
    fm.add_argument("--sigma-px", type=float, default=4.0, help="pixel noise")
    cmm = sub.add_parser("compare-models", help="fit several models, rank them")
    cmm.add_argument("--session", type=Path, required=True)
    cmm.add_argument("--models", default=None, help="comma list; default: all")
    cmm.add_argument("--fit-lengths", action="store_true")
    cmm.add_argument("--sigma-accel", type=float, default=300.0)
    cmm.add_argument("--max-iterations", type=int, default=60, help="solver budget")
    _add_variant_args(cmm)
    kin = sub.add_parser("kinetics", help="inverse dynamics + replay check of a fit")
    kin.add_argument("--session", type=Path, required=True)
    kin.add_argument("--model", default="golfer", help="registered model name")
    kin.add_argument("--body-mass", type=float, required=True, help="kg")
    _add_variant_args(kin)
    _add_variant_tools_parsers(sub)
    lin = sub.add_parser("lineage", help="provenance chain of a pipeline output")
    lin.add_argument("--session", type=Path, required=True)
    lin.add_argument("--path", type=Path, required=True, help="file inside the session")
    lin.add_argument("--json", action="store_true", help="machine-readable")


def _add_variant_tools_parsers(sub: Any) -> None:
    """overlay, compare-variants, annotations-to-observations (#9795/#9796/#9801)."""
    ov = sub.add_parser("overlay", help="draw variants' 3-D results on a view")
    ov.add_argument("--session", type=Path, required=True)
    ov.add_argument("--view", required=True)
    ov.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME",
        help="variant to draw, repeatable (default: the session's default match)",
    )
    ov.add_argument("--out", type=Path, required=True, help="mp4 to write")
    ov.add_argument("--from", dest="start", type=int, default=0, metavar="FRAME")
    ov.add_argument("--to", dest="stop", type=int, default=None, metavar="FRAME")
    ov.add_argument("--speed", type=float, default=1.0)
    ov.add_argument("--observations", default="observations", metavar="SET")
    ov.add_argument("--no-legend", action="store_true")
    cv = sub.add_parser("compare-variants", help="held-out and 3-D error per variant")
    cv.add_argument("--session", type=Path, required=True)
    cv.add_argument("--reference", default="", metavar="NAME")
    cv.add_argument("--observations", default="observations", metavar="SET")
    a2o = sub.add_parser(
        "annotations-to-observations",
        help="manual clicks as an observation set (optionally over a detector set)",
    )
    a2o.add_argument("--session", type=Path, required=True)
    a2o.add_argument("--view", action="append", default=[], help="repeatable")
    a2o.add_argument("--merge-with", default="", metavar="SET")
    a2o.add_argument("--out", default="", metavar="SET")


def _add_variant_args(
    parser: Any, *, observations: bool = False, views: bool = False
) -> None:
    """``--variant`` (and for reconstruct: ``--observations``, ``--views``), #9793."""
    parser.add_argument(
        "--variant",
        default="",
        metavar="NAME",
        help="named match: outputs under variants/NAME/ (default: the session)",
    )
    if observations:
        parser.add_argument(
            "--observations",
            default="observations",
            metavar="SET",
            help="observation-set directory to use (observations, observations_x)",
        )
    if views:
        parser.add_argument(
            "--views",
            default="",
            metavar="a,b",
            help="subset and order of the cameras to use (default: all with cameras)",
        )
