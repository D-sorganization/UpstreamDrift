"""Plot-style integration tests for the live-view controller (#4808).

Verifies:

1. Programmatic style change → renderer applies it on the next frame.
2. Session save / load round-trip preserves both body and club styles.
3. Backwards-compat: an old session JSON (without ``plot_styles``) loads
   with default styles, no error.

Headless via the matplotlib ``Agg`` backend and ``QT_QPA_PLATFORM=offscreen``
so the suite runs under ``PYTEST_QT_API=pyqt6`` without a display.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

# Headless GUI/drawing stack BEFORE matplotlib is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.shared.python.plot_style import (  # noqa: E402
    MarkerShape,
    MarkerStyle,
    PresetLibrary,
    StaticColor,
)
from src.tools.starting_pose_matcher.live_view_controller import (  # noqa: E402
    LiveViewController,
    StyledMarkerLayer,
    default_body_marker_style,
    default_club_marker_style,
)
from src.tools.starting_pose_matcher.session_schema import (  # noqa: E402
    SESSION_SCHEMA_VERSION,
    PlotStylesBlock,
    default_plot_styles,
    parse_plot_styles,
    serialize_plot_styles,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def axes_canvas():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    yield ax, fig.canvas
    plt.close(fig)


def _body(n_frames: int = 6, n_markers: int = 4) -> SimpleNamespace:
    rng = np.random.default_rng(0)
    marker_xyz = rng.standard_normal((n_frames, n_markers, 3))
    names = tuple(f"m{i}" for i in range(n_markers))
    return SimpleNamespace(marker_xyz=marker_xyz, marker_names=names)


def _club(n_frames: int = 6) -> SimpleNamespace:
    rng = np.random.default_rng(1)
    return SimpleNamespace(
        mid_hands=rng.standard_normal((n_frames, 3)),
        clubhead=rng.standard_normal((n_frames, 3)),
        clubface_triad=np.broadcast_to(np.eye(3), (n_frames, 3, 3)).copy(),
    )


# --------------------------------------------------------------------------- #
# 0. Defaults                                                                 #
# --------------------------------------------------------------------------- #


def test_controller_uses_default_styles_when_unspecified(axes_canvas) -> None:
    ax, canvas = axes_canvas
    controller = LiveViewController(ax, canvas)

    assert isinstance(controller.body_marker_style, MarkerStyle)
    assert isinstance(controller.club_marker_style, MarkerStyle)
    # Defaults match the preset-resolution helpers exactly.
    assert controller.body_marker_style == default_body_marker_style()
    assert controller.club_marker_style == default_club_marker_style()


def test_default_helpers_pull_from_preset_library() -> None:
    library = PresetLibrary.default()
    assert "default" in library
    preset = library["default"]
    names = {entry.name for entry in preset.entries}
    # Both default entry names referenced by the controller must exist
    # in the shipped preset, otherwise the helper fall-through hides a
    # genuine schema regression.
    assert {"left_hand", "club_head"}.issubset(names)


def test_constructor_rejects_non_marker_style(axes_canvas) -> None:
    ax, canvas = axes_canvas
    with pytest.raises(TypeError, match="body_marker_style must be MarkerStyle"):
        LiveViewController(ax, canvas, body_marker_style=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="club_marker_style must be MarkerStyle"):
        LiveViewController(ax, canvas, club_marker_style=object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 1. Programmatic style change applies on next frame                          #
# --------------------------------------------------------------------------- #


def test_set_body_style_applies_to_renderer(axes_canvas) -> None:
    ax, canvas = axes_canvas
    controller = LiveViewController(ax, canvas)
    controller.set_target(body=_body())

    layer = controller.layers()["body_markers"]
    assert isinstance(layer, StyledMarkerLayer)
    initial_artists = layer.artists()
    assert initial_artists, "body marker layer should be built with at least one artist"

    new_style = MarkerStyle(
        shape=MarkerShape.CUBE,
        size_px=12.0,
        edge_color="#ff0000",
        edge_width=1.0,
        fill_color=StaticColor("#00ff00"),
        opacity=0.8,
    )
    controller.set_body_style(new_style)
    assert controller.body_marker_style == new_style
    assert layer.style == new_style

    # The next frame update should not raise after a style swap.
    controller.set_frame(2)
    assert controller.current_frame == 2


def test_set_club_style_applies_to_renderer(axes_canvas) -> None:
    ax, canvas = axes_canvas
    controller = LiveViewController(ax, canvas)
    controller.set_target(body=_body(), club=_club())

    layer = controller.layers().get("club_markers")
    assert isinstance(layer, StyledMarkerLayer)

    new_style = MarkerStyle(
        shape=MarkerShape.STAR,
        size_px=10.0,
        fill_color=StaticColor("#abcdef"),
    )
    controller.set_club_style(new_style)
    assert controller.club_marker_style == new_style
    assert layer.style == new_style

    controller.set_frame(3)
    assert controller.current_frame == 3


def test_set_body_style_caches_when_no_target_loaded(axes_canvas) -> None:
    ax, canvas = axes_canvas
    controller = LiveViewController(ax, canvas)
    new_style = MarkerStyle(shape=MarkerShape.DIAMOND)
    controller.set_body_style(new_style)
    assert controller.body_marker_style == new_style

    # Loading a target afterwards must adopt the cached style.
    controller.set_target(body=_body())
    layer = controller.layers()["body_markers"]
    assert isinstance(layer, StyledMarkerLayer)
    assert layer.style == new_style


def test_set_body_style_rejects_wrong_type(axes_canvas) -> None:
    ax, canvas = axes_canvas
    controller = LiveViewController(ax, canvas)
    with pytest.raises(TypeError, match="style must be MarkerStyle"):
        controller.set_body_style("not-a-style")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="style must be MarkerStyle"):
        controller.set_club_style(42)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 2. Session save / load round-trips both styles                              #
# --------------------------------------------------------------------------- #


def _marker_style_to_jsonable(style: MarkerStyle) -> dict:
    """Tiny helper that mirrors the shape persisted by plot_style."""
    fill = style.fill_color
    assert isinstance(fill, StaticColor)
    return {
        "shape": style.shape.value,
        "size_px": float(style.size_px),
        "edge_color": style.edge_color,
        "edge_width": float(style.edge_width),
        "fill_color": {"kind": "static", "hex_value": fill.hex_value},
        "opacity": float(style.opacity),
    }


def test_session_round_trip_preserves_both_styles(tmp_path: Path) -> None:
    body_style = MarkerStyle(
        shape=MarkerShape.CUBE,
        size_px=8.0,
        fill_color=StaticColor("#112233"),
    )
    club_style = MarkerStyle(
        shape=MarkerShape.STAR,
        size_px=11.0,
        fill_color=StaticColor("#445566"),
    )
    block = PlotStylesBlock(
        body=_marker_style_to_jsonable(body_style),
        club=_marker_style_to_jsonable(club_style),
    )

    session = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "plot_styles": serialize_plot_styles(block),
    }
    path = tmp_path / "session.json"
    path.write_text(json.dumps(session), encoding="utf-8")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    parsed = parse_plot_styles(loaded.get("plot_styles"))
    assert parsed == block
    assert parsed.body is not None
    assert parsed.club is not None
    assert parsed.body["shape"] == "cube"
    assert parsed.club["shape"] == "star"
    assert parsed.body["fill_color"]["hex_value"] == "#112233"
    assert parsed.club["fill_color"]["hex_value"] == "#445566"


def test_serialize_and_parse_handle_none_entries() -> None:
    block = PlotStylesBlock(body=None, club=None)
    data = serialize_plot_styles(block)
    assert data == {"body": None, "club": None}
    assert parse_plot_styles(data) == block


# --------------------------------------------------------------------------- #
# 3. Backwards-compat: pre-v6 sessions load with defaults                     #
# --------------------------------------------------------------------------- #


def test_old_session_without_plot_styles_loads_with_defaults(tmp_path: Path) -> None:
    # A v5 session — no ``plot_styles`` field present at all.
    legacy_session = {
        "schema_version": 5,
        "playback": {"current_frame": 0, "speed": 1.0, "loop": True},
        "data_sources": {},
        "body_skeleton": {"style": "lines"},
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy_session), encoding="utf-8")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    parsed = parse_plot_styles(loaded.get("plot_styles"))
    assert parsed == default_plot_styles()
    assert parsed.body is None
    assert parsed.club is None


def test_parse_plot_styles_tolerates_unknown_keys() -> None:
    parsed = parse_plot_styles(
        {
            "body": {
                "shape": "cube",
                "fill_color": {"kind": "static", "hex_value": "#0a0b0c"},
            },
            "club": None,
            "future_field": "ignored",
        }
    )
    assert parsed.body is not None
    assert parsed.body["shape"] == "cube"
    assert parsed.club is None


def test_parse_plot_styles_empty_input_returns_default() -> None:
    assert parse_plot_styles(None) == default_plot_styles()
    assert parse_plot_styles({}) == default_plot_styles()


def test_session_schema_version_bumped_to_v6() -> None:
    assert SESSION_SCHEMA_VERSION >= 6


# --------------------------------------------------------------------------- #
# 4. End-to-end: load legacy → controller still constructs with defaults      #
# --------------------------------------------------------------------------- #


def test_legacy_session_drives_controller_with_default_styles(
    axes_canvas, tmp_path: Path
) -> None:
    legacy_session = {"schema_version": 5}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy_session), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    parsed = parse_plot_styles(loaded.get("plot_styles"))
    assert parsed.body is None and parsed.club is None

    ax, canvas = axes_canvas
    # No style overrides — controller falls back to its built-in defaults.
    controller = LiveViewController(ax, canvas)
    controller.set_target(body=_body())
    layer = controller.layers()["body_markers"]
    assert isinstance(layer, StyledMarkerLayer)
    assert layer.style == default_body_marker_style()
