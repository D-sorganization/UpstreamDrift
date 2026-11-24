"""Command line for the native viewer export tool (NV-5, #11678)."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import partial
import json
import logging
from pathlib import Path
import sys
from typing import Any

from src.shared.python.body_part_viz.force_colors import (
    BLEND_MODES,
    DEFAULT_COMPRESSION_COLOR,
    DEFAULT_TENSION_COLOR,
    ForceColorScale,
)
from src.shared.python.body_part_viz.skin_load_settings import (
    load_skin_load_settings,
)
from src.shared.python.force_overlay.palette import ELLIPSOID_COLOR_MODES
from src.shared.python.golf_view_presets import VIEW_ORDER
from src.shared.python.model_appearance.appearance_selection import (
    AppearanceOverrides,
    resolve_appearance,
)
from src.shared.python.model_appearance.schema import (
    GLOVE_CHOICES,
    document_from_dict,
    document_to_dict,
)
from src.shared.python.motion_matching.execution.grip_selection import (
    add_grip_arguments,
    apply_visual_picks,
    migrate_sidecar_grip,
    overrides_from_args,
    refuse_physical_in_sidecar,
)
from src.tools.native_viewer_export.aim_line import load_aim_settings
from src.tools.native_viewer_export.backends._scene_lighting import (
    LIGHTING_CHOICES,
    LightingSettings,
    load_lighting,
)
from src.tools.native_viewer_export.core import (
    DEFAULT_SPEEDS,
    ELLIPSOID_KINDS,
    ENGINES,
    ExportResult,
    ExportSettings,
    parse_ellipsoid_scale,
)
from src.tools.native_viewer_export.display_units import (
    DISTANCE_UNITS,
    SPEED_UNITS,
    DisplayUnits,
    load_display_units,
)
from src.tools.native_viewer_export.ellipsoid_panel import load_panel_settings
from src.tools.native_viewer_export.ellipsoid_visibility import (
    VisibilitySettings,
    load_visibility_settings,
    panel_note,
)
from src.tools.native_viewer_export.flight_tracer import load_tracer_settings
from src.tools.native_viewer_export.hole_view import load_hole_view_settings
from src.tools.native_viewer_export.impact_factors import load_factor_settings
from src.tools.native_viewer_export.overlay import OverlayOptions, build_overlay_feed
from src.tools.native_viewer_export.runner import ExportJob, run_export
from src.tools.native_viewer_export import camera_framing, shot_setup

logger = logging.getLogger(__name__)


def _add_ellipsoid_arguments(p: argparse.ArgumentParser) -> None:
    """Add interface capability and mobility ellipsoid options (CAP-6, #14038, #14046)."""
    p.add_argument(
        "--interface-ellipsoid",
        action="append",
        choices=ELLIPSOID_KINDS,
        default=None,
        metavar="KIND",
        help="whole-multibody capability or mobility ellipsoid at segment interfaces "
        "(repeatable; choices: mobility, effective-mass, acceleration-capacity, "
        "force-capacity; CAP-6, #14038, #14046)",
    )
    p.add_argument(
        "--ellipsoid-scale",
        default="auto",
        metavar="SCALE",
        help="ellipsoid scaling: 'auto' (clip-wide 95th-percentile scale) or 'fixed:<m>' "
        "in metres (default 'auto'; CAP-6, #14038)",
    )
    p.add_argument(
        "--drift-arrow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="draw drift acceleration arrow at the interface ellipsoid centre "
        "(default: off; CAP-6, #14038)",
    )
    p.add_argument(
        "--hand-mobility",
        nargs="?",
        const="velocity",
        default=None,
        choices=("velocity", "force", "both", "mobility"),
        help="hand mobility overlay: 'velocity'/'mobility' (velocity ellipsoid), "
        "'force' (force ellipsoid), 'both' (both ellipsoids), or flag alias for "
        "'--interface-ellipsoid mobility' (HCM-3, #14004, #14046)",
    )
    p.add_argument(
        "--hand-mobility-mode",
        choices=("constrained", "unconstrained"),
        default="constrained",
        help="kinematic closure mode for hand mobility ellipsoids: "
        "'constrained' (both hands on club) or 'unconstrained' (hand alone) "
        "(default: constrained; HCM-3, #14004)",
    )
    p.add_argument(
        "--mobility-points",
        default="hands",
        metavar="SELECTOR",
        help="points for mobility overlays: comma-separated groups (hands, wrists, "
        "elbows, shoulders, pelvis, feet, club) or point names such as clubhead "
        "(default: hands; HCM-5, #14006, #15150)",
    )
    p.add_argument(
        "--max-mobility-points",
        type=int,
        default=8,
        metavar="N",
        help="max points to display for clutter control (default: 8; HCM-5, #14006)",
    )
    p.add_argument(
        "--mobility-condition-cutoff",
        type=float,
        default=None,
        metavar="MAX_COND",
        help="condition threshold to suppress ill-conditioned ellipsoids (HCM-5, #14006)",
    )
    p.add_argument(
        "--ellipsoid-color-by",
        choices=ELLIPSOID_COLOR_MODES,
        default=ELLIPSOID_COLOR_MODES[0],
        help="what an ellipsoid colour encodes: its point (default) or its kind, "
        "so mobility and effective-mass differ at one point (#15150)",
    )
    p.add_argument(
        "--ellipsoid-visibility-settings",
        type=Path,
        default=None,
        metavar="JSON",
        help="ellipsoid scale grouping (per_kind/per_point), nested alpha and "
        "minimum semi-axis (default: ellipsoid_visibility_settings.json beside "
        "the tool; #15150)",
    )


def _add_hand_force_arguments(p: argparse.ArgumentParser) -> None:
    """Add the hand-on-club force overlays (tip-to-tail split, palm on thumb)."""
    p.add_argument(
        "--ztcf-zvcf-hand-force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="draw the trail hand's force on the club tip to tail: ZTCF drift "
        "(tau = 0) then ZVCF control (the applied torques), with the total; "
        "needs a dynamics bundle with efforts (default: off) (#15685)",
    )
    p.add_argument(
        "--palm-thumb-force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="draw the allocated trail-palm-on-lead-thumb normal force at the "
        "contact patch (an equivalent of the grip closure force, not a "
        "measured contact; unloaded frames draw nothing; default: off)",
    )


def _add_overlay_arguments(p: argparse.ArgumentParser) -> None:
    """Add the overlay and segment-shading options."""
    p.add_argument(
        "--no-overlay",
        action="store_true",
        help="disable every overlay (force and torque vectors, segment shading)",
    )
    p.add_argument(
        "--force-vectors",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="draw straight force arrows (default: on, except on a kinematic "
        "replay with zero efforts, where only this flag draws them)",
    )
    p.add_argument(
        "--torque-vectors",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="draw curling joint-torque arcs (default: on, except on a kinematic "
        "replay with zero efforts, where only this flag draws them)",
    )
    _add_hand_force_arguments(p)
    _add_ellipsoid_arguments(p)
    p.add_argument(
        "--axial-impact-blackout",
        action=argparse.BooleanOptionalAction,
        default=load_skin_load_settings().axial_impact_blackout,
        help="leave segments unshaded around ball impact, whose force is not "
        "modelled (default: skin_load_settings.json; off shades them from the "
        "motion alone)",
    )
    shade = p.add_argument_group(
        "segment shading",
        "Translucent tension/compression shading of the golfer's segments "
        "(MuJoCo viewer only; axial loads from inverse dynamics, tension "
        "positive). Loads beyond the limit saturate.",
    )
    shade.add_argument(
        "--segment-shading",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="shade segments by axial load (default: off)",
    )
    shade.add_argument(
        "--shading-opacity",
        type=float,
        default=0.6,
        metavar="0..1",
        help="shading strength, 0 (invisible) to 1 (full colour; default 0.6)",
    )
    shade.add_argument(
        "--shading-mode",
        choices=BLEND_MODES,
        default="tint",
        help="tint: unloaded segments keep their colour (default); "
        "replace: every loaded segment blends to the load colour",
    )
    shade.add_argument(
        "--tension-color",
        default=DEFAULT_TENSION_COLOR,
        metavar="#RRGGBB",
        help="tension colour (default %(default)s)",
    )
    shade.add_argument(
        "--compression-color",
        default=DEFAULT_COMPRESSION_COLOR,
        metavar="#RRGGBB",
        help="compression colour (default %(default)s)",
    )
    shade.add_argument(
        "--shading-limit-n",
        type=float,
        default=1000.0,
        metavar="N",
        help="load in newtons that saturates the colour, both signs (default 1000)",
    )
    shade.add_argument(
        "--tension-limit-n",
        type=float,
        default=None,
        metavar="N",
        help="tension saturation load (default: --shading-limit-n)",
    )
    shade.add_argument(
        "--compression-limit-n",
        type=float,
        default=None,
        metavar="N",
        help="compression saturation load (default: --shading-limit-n)",
    )


def _add_impact_factor_arguments(p: argparse.ArgumentParser) -> None:
    """Impact factors panel flags (#15150)."""
    p.add_argument(
        "--impact-factors",
        action="store_true",
        help="draw the impact factors panel (club speed, path, face, face to "
        "path, attack angle, loft, strike point) on every frame; the values "
        "freeze at impact and stay to the end of the clip (default: off)",
    )
    p.add_argument(
        "--impact-factors-settings",
        type=Path,
        default=None,
        metavar="PATH",
        help="settings file of the impact factors panel (turns it on; default "
        "impact_factors_settings.json beside the tool)",
    )


def _add_ball_flight_arguments(p: argparse.ArgumentParser) -> None:
    """Ball flight and shot tracer flags (#15150)."""
    p.add_argument(
        "--ball-flight",
        action="store_true",
        help="from impact on fly the ball along the model-predicted carry "
        "trajectory with a shot tracer and hold the clip until it lands; the "
        "down_the_line_flight view keeps the flight in the picture (default: off)",
    )
    p.add_argument(
        "--ball-flight-settings",
        type=Path,
        default=None,
        metavar="PATH",
        help="settings file of the ball flight and tracer (turns it on; default "
        "flight_tracer_settings.json beside the tool)",
    )


def _add_hole_view_arguments(p: argparse.ArgumentParser) -> None:
    """Overhead hole view flags (#15150, track A4)."""
    p.add_argument(
        "--hole-view",
        action="store_true",
        help="also write a plan-view still of the flight on a hole (fairway, "
        "green, trace, landing, carry); turns the ball flight on (default: off)",
    )
    p.add_argument(
        "--hole-preset",
        default=None,
        metavar="NAME",
        help="hole of --hole-view: par3, par4 or par5 (default: hole_view_settings.json)",
    )


def _add_unit_arguments(p: argparse.ArgumentParser) -> None:
    """Display unit flags of the flight labels and the impact card (#15443)."""
    p.add_argument(
        "--distance-unit",
        dest="ball_flight_distance_unit",
        choices=DISTANCE_UNITS,
        default=None,
        help="unit of flight distance labels (default: display_units_settings.json)",
    )
    p.add_argument(
        "--speed-unit",
        dest="ball_flight_speed_unit",
        choices=SPEED_UNITS,
        default=None,
        help="unit of speed labels, flight HUD and impact card "
        "(default: display_units_settings.json); the manifest stays SI",
    )


def _display_units(args: argparse.Namespace) -> DisplayUnits:
    """Units of the settings file with the command line overrides applied."""
    units = load_display_units()
    distance = getattr(args, "ball_flight_distance_unit", None)
    speed = getattr(args, "ball_flight_speed_unit", None)
    return replace(
        units,
        distance_unit=distance or units.distance_unit,
        speed_unit=speed or units.speed_unit,
    )


def _add_lighting_arguments(p: argparse.ArgumentParser) -> None:
    """Scene lighting flags of the MuJoCo export (#15150)."""
    p.add_argument(
        "--lighting",
        choices=LIGHTING_CHOICES,
        default=LIGHTING_CHOICES[0],
        help="scene lighting of the MuJoCo export: the studio lights, or the "
        "showpiece lights, shadow and ground tone (default: %(default)s)",
    )
    p.add_argument(
        "--lighting-settings",
        type=Path,
        default=None,
        metavar="PATH",
        help="settings file of the showpiece lighting (turns it on; default "
        "backends/showpiece_lighting_settings.json beside the tool)",
    )


def _lighting_for(args: argparse.Namespace) -> LightingSettings | None:
    """The opted-in lighting settings, or ``None`` for the studio scene."""
    if args.lighting == LIGHTING_CHOICES[0] and args.lighting_settings is None:
        return None
    return load_lighting(args.lighting_settings)


def _add_shot_setup_arguments(p: argparse.ArgumentParser) -> None:
    """Opt-in shot setup flags: club roll in the hands and ball position."""
    p.add_argument(
        "--club-roll-in-hands",
        type=float,
        default=0.0,
        metavar="DEG",
        help="rigidly turn the club about its shaft relative to the hands, same "
        "body motion replayed (ignores shaft bend; positive opens the face: on "
        "the matched right-handed driver swing +1 degree turns the face about "
        "0.73 degrees right and adds about 0.67 degrees of loft; |DEG| <= "
        "max_abs_roll_deg of shot_setup_settings.json; default: %(default)s)",
    )
    p.add_argument(
        "--ball-position-shift",
        type=float,
        default=0.0,
        metavar="M",
        help="move the computed address ball M metres along the target line, "
        "positive toward the target; the ball is still placed where the face "
        "centre passes. Forward of the captured ball the measured club was "
        "already slowed by the real ball, so only the first few centimetres "
        "are valid (|M| <= max_abs_ball_shift_m of shot_setup_settings.json; "
        "default: %(default)s)",
    )
    p.add_argument(
        "--shot-shape",
        choices=shot_setup.SHAPES,
        default=None,
        help="solve the club roll in the hands from the flight model instead "
        "of taking a number: straight lands on the target line, draw / fade "
        "are the straight roll minus / plus shape_roll_step_deg (named by the "
        "ball curve of a right-handed golfer: a draw curves left, a fade "
        "right). Excludes a non-zero --club-roll-in-hands; combines with "
        "--ball-position-shift (default: off)",
    )
    p.add_argument(
        "--aim-line",
        choices=("finish", "address"),
        default=None,
        help="reference line of the plan view, side figure and wide-view ground "
        "line: finish (ball to where the ball finishes, so a draw or fade that "
        "ends on target reads as on target) or address (the address target "
        "line). Default: aim_line_settings.json, which is finish with "
        "--shot-shape and otherwise leaves the picture unchanged",
    )


def _add_clip_overlay_arguments(p: argparse.ArgumentParser) -> None:
    """Optional clip flags: impact factors panel, ball flight and lighting."""
    _add_impact_factor_arguments(p)
    _add_ball_flight_arguments(p)
    _add_hole_view_arguments(p)
    _add_unit_arguments(p)
    _add_lighting_arguments(p)
    _add_shot_setup_arguments(p)
    p.add_argument(
        "--contact-hold-frames",
        type=int,
        default=None,
        metavar="N",
        help="repeat the contact-instant frame N more times "
        "(default: contact_frame_settings.json)",
    )


def _add_camera_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--camera-lookat",
        default=None,
        metavar="X,Y,Z",
        help="look-at point in metres (world frame); beats the automatic "
        "centre-of-mass look-at",
    )
    p.add_argument(
        "--camera-height",
        type=float,
        default=None,
        help="look-at height in metres; keeps the automatic centre-of-mass x/y "
        "(lower it for a short club on the turf)",
    )
    p.add_argument(
        "--camera-distance",
        type=float,
        default=None,
        help="camera distance in metres (default: each view's own distance)",
    )


def _add_grip_and_ball_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--grip",
        action="store_true",
        help="add the grip overlay (per-hand, net and couple; GCV-10) and write a "
        "<swing>_<engine>_grip_wrench.json plot payload; pair with --views "
        "hands_closeup",
    )
    p.add_argument(
        "--no-ball",
        action="store_true",
        help="do not draw the decorative golf ball (shown until impact)",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.tools.native_viewer_export",
        description=(
            "Render a same-input swing in each engine's native viewer (Drake MeshCat, "
            "Pinocchio MeshCat, OpenSim simbody under xvfb, MyoSuite arena, MuJoCo Renderer) as mp4 "
            "clips: one per camera view plus a labelled 2x2."
        ),
    )
    p.add_argument("--bundle", type=Path, required=True, help="same-input bundle .npz")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--swing", required=True, help="swing name used in file names")
    p.add_argument("--club", default="Driver", help="club label for the HUD")
    p.add_argument(
        "--engines", default=",".join(ENGINES), help="comma-separated engines"
    )
    for engine in ENGINES:
        p.add_argument(
            f"--receipt-{engine}",
            type=Path,
            help=f"{engine} same-input receipt .json (sibling .npz holds q)",
        )
    p.add_argument(
        "--views", default=",".join(VIEW_ORDER), help="comma-separated views"
    )
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="DEPRECATED alias: render every Nth state (use --speeds and --fps)",
    )
    p.add_argument("--fps", type=int, default=60, help="video frame rate")
    p.add_argument(
        "--speeds",
        default=",".join(f"{v:g}" for v in DEFAULT_SPEEDS),
        help="comma-separated playback speeds, each in (0, 4]; one clip set per "
        "speed with _1x / _0p5x suffixes (default 1,0.5,0.25: full, half, quarter)",
    )
    p.add_argument(
        "--impact-window",
        type=float,
        default=None,
        metavar="SECONDS",
        help="also write a clip of this many swing seconds centred on impact "
        "at 0.1x speed (suffix _impact_0p1x)",
    )
    p.add_argument(
        "--impact-speed",
        type=float,
        default=None,
        help="playback speed of the --impact-window clip (default 0.1)",
    )
    p.add_argument(
        "--no-hud",
        action="store_true",
        help="clean frames: no view label, HUD or legend text",
    )
    p.add_argument(
        "--event-captions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="draw an anatomical swing event caption line ('Event: <name>') in the "
        "HUD (default: off; HRG-E4, #13175)",
    )
    p.add_argument(
        "--impact-time",
        type=float,
        default=None,
        metavar="SECONDS",
        help="impact time in the swing; default: bundle provenance impact_time_s, "
        "else the last sample",
    )
    p.add_argument(
        "--preset",
        choices=("hq", "preview"),
        default="hq",
        help="hq: 1280x720 tiles, libx264 yuv420p CRF 18 (default); "
        "preview: 640x544, CRF 23",
    )
    p.add_argument(
        "--size", default=None, help="tile size WIDTHxHEIGHT (overrides the preset)"
    )
    _add_overlay_arguments(p)
    _add_grip_and_ball_arguments(p)
    _add_clip_overlay_arguments(p)
    _add_camera_arguments(p)
    p.add_argument("--no-grid", action="store_true", help="skip the 2x2 clip")
    add_appearance_arguments(p)
    add_grip_arguments(p, physical=True)
    return p


def add_appearance_arguments(p: argparse.ArgumentParser) -> None:
    """Appearance flags (HRG-B2, #13167). Precedence: flag, then ``--appearance``
    document, then the contract defaults; nothing is read from GUI settings."""
    group = p.add_argument_group(
        "appearance",
        "explicit flag > --appearance document or sidecar > contract defaults",
    )
    group.add_argument(
        "--appearance",
        type=Path,
        default=None,
        metavar="PATH",
        help="appearance-v1 document or <spec>.appearance.json sidecar",
    )
    group.add_argument(
        "--skin-tone",
        default=None,
        metavar="ID_OR_HEX",
        help="skin palette id (skin_1..skin_8) or a #rrggbb colour",
    )
    group.add_argument(
        "--hair",
        default=None,
        metavar="ID_OR_HEX",
        help="hair palette id (hair_brown, ...) or a #rrggbb colour",
    )
    group.add_argument(
        "--clothing", default=None, metavar="ID", help="clothing preset id"
    )
    group.add_argument(
        "--glove",
        default=None,
        choices=GLOVE_CHOICES,
        help="golf glove by role (default none: bare skin-coloured hands); the "
        "lead hand is the left one for a right-handed golfer",
    )
    group.add_argument(
        "--top-color", default=None, metavar="HEX", help="shirt colour #rrggbb"
    )
    group.add_argument(
        "--bottom-color",
        default=None,
        metavar="HEX",
        help="trousers or shorts colour #rrggbb",
    )


def load_appearance_document(
    path: Path, document: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Parse an appearance-v1 document file (``ValueError`` if unreadable/invalid)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read --appearance {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"--appearance {path} must hold a JSON object")
    if "grip" in data:
        if not isinstance(data["grip"], Mapping):
            raise ValueError("appearance.json 'grip' must be a JSON object")
        migrated = migrate_sidecar_grip(data["grip"], document)
        if migrated:
            data["grip"] = migrated
        else:
            data.pop("grip")
    document_from_dict(data)  # validates (fail closed)
    return data


def build_appearance(args: argparse.Namespace) -> dict[str, Any] | None:
    """Resolved appearance document, or ``None`` when no appearance flag is given
    (backends then render the contract default, exactly as before)."""
    overrides = AppearanceOverrides(
        skin_tone=args.skin_tone,
        hair=args.hair,
        clothing=args.clothing,
        top_color=args.top_color,
        bottom_color=args.bottom_color,
        glove=args.glove,
    )
    grip = overrides_from_args(args)
    refuse_physical_in_sidecar(grip)  # never silently applied picture-only
    if args.appearance is None and overrides.is_empty and not grip.has_visual:
        return None
    document = (
        None if args.appearance is None else load_appearance_document(args.appearance)
    )
    resolved = resolve_appearance(overrides, document)
    return document_to_dict(document_from_dict(apply_visual_picks(resolved, grip)))


def parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise ValueError(f"--size must look like 640x544, got {text!r}") from None
    return w, h


def parse_speeds(text: str) -> tuple[float, ...]:
    try:
        return tuple(float(part) for part in text.split(",") if part.strip())
    except ValueError:
        raise ValueError(f"--speeds must look like 1,0.5,0.25, got {text!r}") from None


def build_shading(args: argparse.Namespace) -> ForceColorScale | None:
    """The one ``ForceColorScale`` of the shading options (``ValueError`` if invalid).

    Postcondition: ``None`` when shading is off or ``--no-overlay`` is set.
    """
    if not args.segment_shading or args.no_overlay:
        return None
    tension, compression = (
        args.shading_limit_n if own is None else own
        for own in (args.tension_limit_n, args.compression_limit_n)
    )
    return ForceColorScale(
        enabled=True,
        tension_limit_n=tension,
        compression_limit_n=compression,
        tension_color=args.tension_color,
        compression_color=args.compression_color,
        opacity=args.shading_opacity,
        blend_mode=args.shading_mode,
    )


def _visibility_of(args: argparse.Namespace) -> VisibilitySettings | None:
    """Visibility settings of ``--ellipsoid-visibility-settings``, else ``None``."""
    path = getattr(args, "ellipsoid_visibility_settings", None)
    return None if path is None else load_visibility_settings(path)


def _panel_for(args: argparse.Namespace) -> Any:
    """Panel settings whose footer follows the visibility file; ``None`` without it."""
    visibility = _visibility_of(args)
    if visibility is None:
        return None
    return replace(load_panel_settings(), note=panel_note(visibility))


def _camera_changes(args: argparse.Namespace) -> dict[str, Any]:
    """Camera fields of ``ExportSettings`` from the flags (``ValueError``)."""
    changes: dict[str, Any] = {}
    if args.camera_lookat is not None:
        changes["lookat_m"] = camera_framing.parse_lookat(args.camera_lookat)
        changes["lookat_explicit"] = True
    if args.camera_height is not None:
        changes["lookat_height_m"] = camera_framing.check_finite(
            "height", args.camera_height
        )
    if args.camera_distance is not None:
        changes["distance_m"] = camera_framing.check_distance(args.camera_distance)
    return changes


def _ball_flight_for(args: argparse.Namespace, units: DisplayUnits) -> Any:
    """Tracer settings of the flag combination, or ``None`` when no flight is wanted."""
    hole = getattr(args, "hole_view", False)
    if not (args.ball_flight or args.ball_flight_settings is not None or hole):
        return None
    tracer = load_tracer_settings(args.ball_flight_settings, units)
    aim = tracer.aim or load_aim_settings()
    mode = aim.resolved(getattr(args, "shot_shape", None), args.aim_line)
    tracer = replace(tracer, aim=replace(aim, aim_line=mode))
    if not hole:
        return tracer
    view = load_hole_view_settings(preset=args.hole_preset)
    return replace(tracer, hole_view=view)


def build_settings(args: argparse.Namespace) -> ExportSettings:
    """``ExportSettings`` from parsed arguments (raises ``ValueError``)."""
    base = ExportSettings.preview() if args.preset == "preview" else ExportSettings()
    units = _display_units(args)
    width, height = (
        parse_size(args.size) if args.size is not None else (base.width, base.height)
    )
    ellipsoids = list(args.interface_ellipsoid or [])
    hand_mob = getattr(args, "hand_mobility", None)
    if hand_mob:
        if hand_mob in ("velocity", "mobility", True):
            if "mobility" not in ellipsoids:
                ellipsoids.append("mobility")
        elif hand_mob == "force":
            if "effective-mass" not in ellipsoids:
                ellipsoids.append("effective-mass")
        elif hand_mob == "both":
            if "mobility" not in ellipsoids:
                ellipsoids.append("mobility")
            if "effective-mass" not in ellipsoids:
                ellipsoids.append("effective-mass")
    ellipsoid_scale = parse_ellipsoid_scale(getattr(args, "ellipsoid_scale", "auto"))
    hand_mobility_mode = getattr(args, "hand_mobility_mode", "constrained")
    mobility_points = getattr(args, "mobility_points", "hands")
    max_mobility_points = getattr(args, "max_mobility_points", 8)
    mobility_condition_cutoff = getattr(args, "mobility_condition_cutoff", None)
    return replace(
        base,
        **_camera_changes(args),
        ellipsoid_panel=_panel_for(args),
        views=tuple(v for v in args.views.split(",") if v),
        width=width,
        height=height,
        fps=args.fps,
        speeds=parse_speeds(args.speeds),
        stride=args.stride,
        impact_time_s=args.impact_time,
        impact_window_s=args.impact_window,
        contact_hold_frames=getattr(args, "contact_hold_frames", None),
        overlays=not args.no_overlay,
        shading=build_shading(args),
        force_vectors=args.force_vectors is not False,
        torque_vectors=args.torque_vectors is not False,
        vectors_requested=args.force_vectors is True or args.torque_vectors is True,
        palm_thumb_force=args.palm_thumb_force,
        interface_ellipsoids=tuple(ellipsoids),
        ellipsoid_scale=ellipsoid_scale,
        drift_arrow=bool(getattr(args, "drift_arrow", False)),
        hand_mobility_mode=hand_mobility_mode,
        mobility_points=mobility_points,
        max_mobility_points=max_mobility_points,
        mobility_condition_cutoff=mobility_condition_cutoff,
        ellipsoid_color_by=getattr(args, "ellipsoid_color_by", "point"),
        multiview=not args.no_grid,
        grip=args.grip,
        hud=not args.no_hud,
        event_captions=bool(getattr(args, "event_captions", False)),
        ball=not args.no_ball,
        impact_factors=load_factor_settings(args.impact_factors_settings)
        if args.impact_factors or args.impact_factors_settings is not None
        else None,
        appearance=build_appearance(args),
        lighting=_lighting_for(args),
        club_roll_in_hands_deg=getattr(args, "club_roll_in_hands", 0.0),
        ball_position_shift_m=getattr(args, "ball_position_shift", 0.0),
        shot_shape=getattr(args, "shot_shape", None),
        ball_flight=_ball_flight_for(args, units),
        speed_unit=units.speed_unit if args.ball_flight_speed_unit else None,
        **(
            {"impact_speed": args.impact_speed} if args.impact_speed is not None else {}
        ),
    )


def main(
    argv: Sequence[str] | None = None,
    on_results: Callable[[list[ExportResult]], None] | None = None,
) -> int:
    """Run the exporter; ``on_results`` receives the results (#15150 G8)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.shot_shape is not None and args.club_roll_in_hands != 0.0:
        parser.error("--shot-shape solves the roll: omit --club-roll-in-hands")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.stride is not None:
        logger.warning(
            "--stride is deprecated: playback is time-based; use --speeds and --fps"
        )
    try:
        settings = build_settings(args)
        receipts = {
            e: getattr(args, f"receipt_{e}")
            for e in ENGINES
            if getattr(args, f"receipt_{e}") is not None
        }
        job = ExportJob(
            args.bundle,
            args.out,
            args.swing,
            args.club,
            tuple(e for e in args.engines.split(",") if e),
            receipts,
        )
        results = run_export(
            job,
            settings,
            overlay_factory=partial(
                build_overlay_feed,
                grip=settings.grip,
                axial=settings.shading_enabled,
                options=OverlayOptions(
                    forces_requested=settings.vectors_requested,
                    show_force=settings.force_vectors,
                    show_torque=settings.torque_vectors,
                    palm_thumb=settings.palm_thumb_force,
                    interface_ellipsoids=settings.interface_ellipsoids,
                    ellipsoid_scale=settings.ellipsoid_scale,
                    hand_mobility_mode=settings.hand_mobility_mode,
                    mobility_points=settings.mobility_points,
                    max_mobility_points=settings.max_mobility_points,
                    mobility_condition_cutoff=settings.mobility_condition_cutoff,
                    ellipsoid_color_by=settings.ellipsoid_color_by,
                    visibility=_visibility_of(args),
                    axial_impact_blackout=args.axial_impact_blackout,
                    ztcf_zvcf_hand_force=args.ztcf_zvcf_hand_force,
                ),
            ),
        )
    except (ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    if on_results is not None:
        on_results(results)
    for r in results:
        if r.skipped:
            sys.stdout.write(f"{r.engine}: SKIPPED ({r.skipped_reason})\n")
        else:
            if r.ball is not None:
                sys.stdout.write(f"{r.engine}: ball {r.ball}\n")
            if r.flight is not None:
                sys.stdout.write(f"{r.engine}: flight {r.flight}\n")
            sys.stdout.write(
                f"{r.engine}: "
                + ", ".join(
                    f"{n} frames ({s.lstrip('_')})"
                    for s, n in r.frames_by_suffix.items()
                )
                + " -> "
                + ", ".join(p.name for p in r.paths.values())
                + "\n"
            )
    return 0 if any(not r.skipped for r in results) else 1
