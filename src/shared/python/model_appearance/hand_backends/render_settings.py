"""Settings of the shared grip-hand renderer (#15150).

Every number the renderer uses lives in ``scripts/config/hand_render.json``.
This module loads that file, validates it, and hands back an immutable
:class:`HandRenderSettings`. Nothing here imports a GL library.

Contract: a missing file, a missing key, a non-finite number, a value outside
its range, or an unknown view name raises ``ValueError``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.shared.python.contracts import ensure
from src.shared.python.model_appearance.library import MATERIALS

#: The six views every backend is judged from.
REQUIRED_VIEWS: tuple[str, ...] = (
    "palm",
    "back",
    "thumb_side",
    "pinky_side",
    "butt",
    "eye",
)
#: Light roles the rig must define.
REQUIRED_LIGHTS: tuple[str, ...] = ("key", "fill", "rim")
#: Scene part kinds, in mask-id order (background is zero).
PART_KINDS: tuple[str, ...] = ("skin", "grip", "shaft")
#: Engines that need a MuJoCo GL context, and the software fallback.
GL_ENGINES: tuple[str, ...] = ("osmesa", "egl")
FALLBACK_ENGINE = "matplotlib"

_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SETTINGS_PATH = _REPO_ROOT / "scripts" / "config" / "hand_render.json"

Vec3 = tuple[float, float, float]
Rgba = tuple[float, float, float, float]


@dataclass(frozen=True)
class ViewSpec:
    """One camera view in the anatomical ``(thumb, dorsum, shaft)`` basis.

    ``eye`` points from the hand toward the camera; the camera looks along
    ``-eye``. ``up`` is the image-up hint in the same basis.
    """

    name: str
    eye: Vec3
    up: Vec3


@dataclass(frozen=True)
class LightSpec:
    """A directional light. ``direction`` points toward the light in camera
    space (``x`` right, ``y`` up, ``z`` toward the viewer)."""

    name: str
    direction: Vec3
    color: Vec3
    intensity: float
    specular: float
    cast_shadow: bool


@dataclass(frozen=True)
class SurfaceSpec:
    """Specular response of one part kind."""

    specular: float
    shininess: float


@dataclass(frozen=True)
class SheetSpec:
    """Contact-sheet layout."""

    cell_px: int
    label_height_px: int
    label_font_px: int
    margin_px: int
    label_rgb: tuple[int, int, int]


@dataclass(frozen=True)
class HandRenderSettings:
    """Validated renderer settings. Build with :func:`settings_from_mapping`."""

    frame_px: int
    fill_fraction: float
    fill_tolerance: float
    fov_deg: float
    frame_iterations: int
    supersample: int
    msaa_samples: int
    znear_ratio: float
    zfar_ratio: float
    backdrop_texture_px: int
    epsilon: float
    grip_radius_m: float
    shaft_radius_m: float
    grip_length_margin_m: float
    shaft_extension_m: float
    club_sides: int
    rolls_deg: tuple[float, ...]
    roll_range_deg: tuple[float, float]
    engine_order: tuple[str, ...]
    gl_probe_px: int
    gl_probe_timeout_s: float
    background_rgba: Rgba
    skin_rgba: Rgba
    grip_rgba: Rgba
    shaft_rgba: Rgba
    surface: Mapping[str, SurfaceSpec]
    views: Mapping[str, ViewSpec]
    lights: Mapping[str, LightSpec]
    ambient_rgb: Vec3
    wrap: float
    sss_tint: Vec3
    sss_strength: float
    shadows: bool
    shadow_map_px: int
    basis_outer_radius_quantile: float
    basis_thumb_height_quantile: float
    normal_weld_tolerance_m: float
    luminance_weights: Vec3
    min_luminance_std: float
    golden_px: int
    perceptual_mae_max: float
    matplotlib_dpi: float
    matplotlib_edge_width: float
    sheet: SheetSpec
    raw_json: str
    settings_hash: str

    def with_overrides(self, **overrides: Any) -> HandRenderSettings:
        """A re-validated copy with top-level JSON keys replaced.

        Example: ``settings.with_overrides(frame_px=1536, shadows=False)``.
        """
        raw = json.loads(self.raw_json)
        unknown = sorted(set(overrides) - set(raw))
        if unknown:
            raise ValueError(f"unknown hand-render setting(s): {unknown}")
        raw.update(overrides)
        return settings_from_mapping(raw)

    def view(self, name: str) -> ViewSpec:
        """The named view. Raises ``ValueError`` for an unknown name."""
        if not isinstance(name, str) or name not in self.views:
            raise ValueError(f"unknown view {name!r}; known: {sorted(self.views)}")
        return self.views[name]

    def kind_rgba(self, kind: str) -> Rgba:
        """Default colour of a part kind."""
        table = {
            "skin": self.skin_rgba,
            "grip": self.grip_rgba,
            "shaft": self.shaft_rgba,
        }
        if kind not in table:
            raise ValueError(f"unknown part kind {kind!r}; known: {PART_KINDS}")
        return table[kind]


def _get(raw: Mapping[str, Any], key: str) -> Any:
    if not isinstance(raw, Mapping):
        raise ValueError(f"expected an object holding {key!r}, got {type(raw)}")
    if key not in raw:
        raise ValueError(f"hand-render settings are missing {key!r}")
    return raw[key]


def _number(
    raw: Mapping[str, Any],
    key: str,
    *,
    low: float = -math.inf,
    high: float = math.inf,
    low_open: bool = False,
) -> float:
    value = _get(raw, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key!r} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key!r} must be finite, got {value!r}")
    below = number <= low if low_open else number < low
    if below or number > high:
        raise ValueError(f"{key!r}={number} is outside its range ({low}, {high})")
    return number


def _integer(raw: Mapping[str, Any], key: str, *, low: int, high: int) -> int:
    value = _get(raw, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key!r} must be an integer, got {value!r}")
    if not low <= value <= high:
        raise ValueError(f"{key!r}={value} is outside [{low}, {high}]")
    return value


def _boolean(raw: Mapping[str, Any], key: str) -> bool:
    value = _get(raw, key)
    if not isinstance(value, bool):
        raise ValueError(f"{key!r} must be true or false, got {value!r}")
    return value


def _vector(
    raw: Mapping[str, Any],
    key: str,
    size: int,
    *,
    low: float = -math.inf,
    high: float = math.inf,
    nonzero: bool = False,
) -> tuple[float, ...]:
    value = _get(raw, key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{key!r} must be a list of {size} numbers")
    if len(value) != size:
        raise ValueError(f"{key!r} must hold {size} numbers, got {len(value)}")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{key!r} holds a non-number: {item!r}")
        number = float(item)
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"{key!r} holds {item!r}, outside [{low}, {high}]")
        out.append(number)
    if nonzero and not any(out):
        raise ValueError(f"{key!r} must not be the zero vector")
    return tuple(out)


def _views(raw: Mapping[str, Any]) -> dict[str, ViewSpec]:
    table = _get(raw, "views")
    if not isinstance(table, Mapping):
        raise ValueError("'views' must be an object")
    unknown = sorted(set(table) - set(REQUIRED_VIEWS))
    if unknown:
        raise ValueError(f"unknown view name(s) {unknown}; known: {REQUIRED_VIEWS}")
    missing = [name for name in REQUIRED_VIEWS if name not in table]
    if missing:
        raise ValueError(f"'views' is missing {missing}")
    views = {}
    for name in REQUIRED_VIEWS:
        eye = _vector(table[name], "eye", 3, nonzero=True)
        up = _vector(table[name], "up", 3, nonzero=True)
        cross = (
            eye[1] * up[2] - eye[2] * up[1],
            eye[2] * up[0] - eye[0] * up[2],
            eye[0] * up[1] - eye[1] * up[0],
        )
        if not any(cross):
            raise ValueError(f"view {name!r}: 'up' is parallel to 'eye'")
        views[name] = ViewSpec(name, eye, up)  # type: ignore[arg-type]
    return views


def _lights(raw: Mapping[str, Any]) -> dict[str, LightSpec]:
    table = _get(raw, "lights")
    if not isinstance(table, Mapping):
        raise ValueError("'lights' must be an object")
    missing = [name for name in REQUIRED_LIGHTS if name not in table]
    if missing:
        raise ValueError(f"'lights' is missing {missing}")
    lights = {}
    for name, entry in table.items():
        lights[name] = LightSpec(
            name=name,
            direction=_vector(entry, "direction", 3, nonzero=True),  # type: ignore[arg-type]
            color=_vector(entry, "color", 3, low=0.0, high=1.0),  # type: ignore[arg-type]
            intensity=_number(entry, "intensity", low=0.0),
            specular=_number(entry, "specular", low=0.0, high=1.0),
            cast_shadow=_boolean(entry, "cast_shadow"),
        )
    return lights


def _surfaces(raw: Mapping[str, Any]) -> dict[str, SurfaceSpec]:
    table = _get(raw, "surface")
    return {
        kind: SurfaceSpec(
            specular=_number(_get(table, kind), "specular", low=0.0, high=1.0),
            shininess=_number(_get(table, kind), "shininess", low=0.0, high=1.0),
        )
        for kind in PART_KINDS
    }


def _library_rgba(raw: Mapping[str, Any], key: str) -> Rgba:
    name = _get(raw, key)
    if not isinstance(name, str) or name not in MATERIALS:
        raise ValueError(f"{key!r}={name!r} is not a library material")
    return tuple(float(c) for c in MATERIALS[name].rgba)  # type: ignore[return-value]


def _engine_order(raw: Mapping[str, Any]) -> tuple[str, ...]:
    order = _get(raw, "engine_order")
    known = (*GL_ENGINES, FALLBACK_ENGINE)
    if (
        not isinstance(order, Sequence)
        or isinstance(order, str)
        or not order
        or any(name not in known for name in order)
        or len(set(order)) != len(order)
    ):
        raise ValueError(f"'engine_order' must list distinct engines from {known}")
    return tuple(order)


def _sheet(raw: Mapping[str, Any]) -> SheetSpec:
    table = _get(raw, "sheet")
    rgb = _vector(table, "label_rgb", 3, low=0.0, high=255.0)
    return SheetSpec(
        cell_px=_integer(table, "cell_px", low=16, high=4096),
        label_height_px=_integer(table, "label_height_px", low=1, high=512),
        label_font_px=_integer(table, "label_font_px", low=1, high=512),
        margin_px=_integer(table, "margin_px", low=0, high=512),
        label_rgb=tuple(int(c) for c in rgb),  # type: ignore[arg-type]
    )


def settings_from_mapping(raw: Mapping[str, Any]) -> HandRenderSettings:
    """Validate a decoded settings object.

    Precondition: ``raw`` is a mapping with every key of
    ``scripts/config/hand_render.json``. Postcondition: the six views and the
    key, fill, and rim lights are present and ``settings_hash`` is the SHA-256
    of the canonical JSON.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(f"hand-render settings must be an object, got {type(raw)}")
    try:
        raw_json = json.dumps(raw, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"hand-render settings are not finite JSON: {exc}") from exc
    rolls = _get(raw, "rolls_deg")
    if not isinstance(rolls, Sequence) or isinstance(rolls, str) or not rolls:
        raise ValueError("'rolls_deg' must be a non-empty list")
    roll_range = _vector(raw, "roll_range_deg", 2)
    if roll_range[0] >= roll_range[1]:
        raise ValueError("'roll_range_deg' must be increasing")
    rolls_deg = _vector(
        raw, "rolls_deg", len(rolls), low=roll_range[0], high=roll_range[1]
    )
    basis = _get(raw, "basis")
    probe = _get(raw, "gl_probe")
    mpl = _get(raw, "matplotlib")
    fill = _number(raw, "fill_fraction", low=0.0, high=1.0, low_open=True)
    settings = HandRenderSettings(
        frame_px=_integer(raw, "frame_px", low=16, high=8192),
        fill_fraction=fill,
        fill_tolerance=_number(
            raw, "fill_tolerance", low=0.0, high=fill, low_open=True
        ),
        fov_deg=_number(raw, "fov_deg", low=1.0, high=120.0),
        frame_iterations=_integer(raw, "frame_iterations", low=1, high=1000),
        supersample=_integer(raw, "supersample", low=1, high=4),
        msaa_samples=_integer(raw, "msaa_samples", low=0, high=16),
        znear_ratio=_number(raw, "znear_ratio", low=0.0, high=1.0, low_open=True),
        zfar_ratio=_number(raw, "zfar_ratio", low=1.0),
        backdrop_texture_px=_integer(raw, "backdrop_texture_px", low=1, high=4096),
        epsilon=_number(raw, "epsilon", low=0.0, high=1e-3, low_open=True),
        grip_radius_m=_number(raw, "grip_radius_m", low=0.0, low_open=True),
        shaft_radius_m=_number(raw, "shaft_radius_m", low=0.0, low_open=True),
        grip_length_margin_m=_number(raw, "grip_length_margin_m", low=0.0),
        shaft_extension_m=_number(raw, "shaft_extension_m", low=0.0, low_open=True),
        club_sides=_integer(raw, "club_sides", low=6, high=1024),
        rolls_deg=rolls_deg,
        roll_range_deg=roll_range,  # type: ignore[arg-type]
        engine_order=_engine_order(raw),
        gl_probe_px=_integer(probe, "frame_px", low=1, high=1024),
        gl_probe_timeout_s=_number(probe, "timeout_s", low=0.0, low_open=True),
        background_rgba=_vector(raw, "background_rgba", 4, low=0.0, high=1.0),  # type: ignore[arg-type]
        skin_rgba=_vector(raw, "skin_rgba", 4, low=0.0, high=1.0),  # type: ignore[arg-type]
        grip_rgba=_library_rgba(raw, "grip_material"),
        shaft_rgba=_library_rgba(raw, "shaft_material"),
        surface=_surfaces(raw),
        views=_views(raw),
        lights=_lights(raw),
        ambient_rgb=_vector(raw, "ambient_rgb", 3, low=0.0, high=1.0),  # type: ignore[arg-type]
        wrap=_number(raw, "wrap", low=0.0, high=1.0),
        sss_tint=_vector(raw, "sss_tint", 3, low=0.0, high=1.0),  # type: ignore[arg-type]
        sss_strength=_number(raw, "sss_strength", low=0.0, high=1.0),
        shadows=_boolean(raw, "shadows"),
        shadow_map_px=_integer(raw, "shadow_map_px", low=64, high=16384),
        basis_outer_radius_quantile=_number(
            basis, "outer_radius_quantile", low=0.0, high=1.0
        ),
        basis_thumb_height_quantile=_number(
            basis, "thumb_height_quantile", low=0.0, high=1.0
        ),
        normal_weld_tolerance_m=_number(
            raw, "normal_weld_tolerance_m", low=0.0, low_open=True
        ),
        luminance_weights=_vector(raw, "luminance_weights", 3, low=0.0, high=1.0),  # type: ignore[arg-type]
        min_luminance_std=_number(raw, "min_luminance_std", low=0.0, high=1.0),
        golden_px=_integer(raw, "golden_px", low=4, high=1024),
        perceptual_mae_max=_number(raw, "perceptual_mae_max", low=0.0, high=1.0),
        matplotlib_dpi=_number(mpl, "dpi", low=1.0),
        matplotlib_edge_width=_number(mpl, "edge_width", low=0.0),
        sheet=_sheet(raw),
        raw_json=raw_json,
        settings_hash=hashlib.sha256(raw_json.encode("utf-8")).hexdigest(),
    )
    if settings.shaft_radius_m >= settings.grip_radius_m:
        raise ValueError("'shaft_radius_m' must be thinner than 'grip_radius_m'")
    ensure(
        set(REQUIRED_VIEWS) <= set(settings.views)
        and set(REQUIRED_LIGHTS) <= set(settings.lights),
        "settings carry the six views and the key, fill, and rim lights",
    )
    return settings


def _reject_constant(token: str) -> float:
    raise ValueError(f"hand-render settings hold a non-finite number: {token}")


@lru_cache(maxsize=8)
def _load_cached(path: Path) -> HandRenderSettings:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read hand-render settings {path}: {exc}") from exc
    try:
        raw = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"hand-render settings {path} are not JSON: {exc}") from exc
    return settings_from_mapping(raw)


def load_hand_render_settings(path: str | Path | None = None) -> HandRenderSettings:
    """Load and validate the renderer settings (default: the repo JSON).

    Raises ``ValueError`` when the file is missing, is not JSON, holds a
    non-finite number, or fails validation.
    """
    return _load_cached(Path(DEFAULT_SETTINGS_PATH if path is None else path))


def resolve_settings(settings: HandRenderSettings | None) -> HandRenderSettings:
    """``settings`` itself, or the default file when ``None``."""
    if settings is None:
        return load_hand_render_settings()
    if not isinstance(settings, HandRenderSettings):
        raise TypeError(f"settings must be HandRenderSettings, got {settings!r}")
    return settings
