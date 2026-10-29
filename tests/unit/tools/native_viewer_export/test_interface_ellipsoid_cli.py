"""CLI options and contract for interface capability and mobility ellipsoids (CAP-6, #14038, #14046)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.core import (
    ELLIPSOID_KINDS,
    ExportSettings,
    parse_ellipsoid_scale,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
BASE = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]


def _settings(*extra: str) -> ExportSettings:
    return cli.build_settings(cli.build_parser().parse_args([*BASE, *extra]))


def test_flags_default_values_and_help() -> None:
    settings = _settings()
    assert settings.interface_ellipsoids == ()
    assert settings.ellipsoid_scale == "auto"
    assert settings.drift_arrow is False

    help_text = cli.build_parser().format_help()
    assert "--interface-ellipsoid" in help_text
    assert "--ellipsoid-scale" in help_text
    assert "--drift-arrow" in help_text
    assert "--hand-mobility" in help_text


def test_interface_ellipsoid_repeatable_and_choices() -> None:
    s1 = _settings("--interface-ellipsoid", "mobility")
    assert s1.interface_ellipsoids == ("mobility",)

    s2 = _settings(
        "--interface-ellipsoid",
        "mobility",
        "--interface-ellipsoid",
        "effective-mass",
        "--interface-ellipsoid",
        "acceleration-capacity",
        "--interface-ellipsoid",
        "force-capacity",
    )
    assert s2.interface_ellipsoids == (
        "mobility",
        "effective-mass",
        "acceleration-capacity",
        "force-capacity",
    )

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([*BASE, "--interface-ellipsoid", "invalid-kind"])


def test_ellipsoid_scale_parsing() -> None:
    assert _settings("--ellipsoid-scale", "auto").ellipsoid_scale == "auto"
    assert _settings("--ellipsoid-scale", "fixed:0.15").ellipsoid_scale == "fixed:0.15"
    assert _settings("--ellipsoid-scale", "fixed:0.08").ellipsoid_scale == "fixed:0.08"

    with pytest.raises(ValueError, match="--ellipsoid-scale"):
        _settings("--ellipsoid-scale", "fixed:-0.1")

    with pytest.raises(ValueError, match="--ellipsoid-scale"):
        _settings("--ellipsoid-scale", "fixed:zero")

    with pytest.raises(ValueError, match="--ellipsoid-scale"):
        _settings("--ellipsoid-scale", "relative")


def test_drift_arrow_toggle() -> None:
    assert _settings("--drift-arrow").drift_arrow is True
    assert _settings("--no-drift-arrow").drift_arrow is False


def test_hand_mobility_backward_compatible_alias() -> None:
    s1 = _settings("--hand-mobility")
    assert s1.interface_ellipsoids == ("mobility",)

    s2 = _settings("--interface-ellipsoid", "effective-mass", "--hand-mobility")
    assert s2.interface_ellipsoids == ("effective-mass", "mobility")

    # Does not duplicate if mobility was already specified
    s3 = _settings("--interface-ellipsoid", "mobility", "--hand-mobility")
    assert s3.interface_ellipsoids == ("mobility",)


def test_ellipsoid_settings_require_overlays() -> None:
    with pytest.raises(ValueError, match="interface_ellipsoids requires overlays=True"):
        _settings("--no-overlay", "--interface-ellipsoid", "mobility")

    with pytest.raises(ValueError, match="drift_arrow requires overlays=True"):
        _settings("--no-overlay", "--drift-arrow")


def test_export_settings_contracts_and_dbc() -> None:
    with pytest.raises(ValueError, match="interface_ellipsoids"):
        ExportSettings(interface_ellipsoids=("unknown-kind",))

    with pytest.raises(ValueError, match="--ellipsoid-scale"):
        ExportSettings(ellipsoid_scale="bad-scale")

    with pytest.raises(TypeError, match="drift_arrow"):
        ExportSettings(drift_arrow="not-a-bool")  # type: ignore[arg-type]


def test_feature_parity_json_entry_updated() -> None:
    path = (
        Path(__file__).resolve().parents[4] / "src" / "config" / "feature_parity.json"
    )
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    feat = data["features"]["tools.native_viewer_export"]
    notes = feat.get("notes", "")
    assert "--interface-ellipsoid" in notes
    assert "--hand-mobility" in notes
    assert "--ellipsoid-scale" in notes
    assert "--drift-arrow" in notes
