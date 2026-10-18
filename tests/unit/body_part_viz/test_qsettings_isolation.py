"""Tests must not touch the user's real QSettings file."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_qsettings_is_redirected_away_from_home() -> None:
    try:
        from PyQt6.QtCore import QSettings
    except (ImportError, OSError) as exc:  # pragma: no cover
        pytest.skip(f"PyQt6 not loadable: {exc}")
    real = Path.home() / ".config" / "UpstreamDrift" / "UpstreamDrift.conf"
    before = real.stat().st_mtime_ns if real.exists() else None
    settings = QSettings("UpstreamDrift", "UpstreamDrift")
    settings.setValue("force_color/isolation_probe", "x")
    settings.sync()
    assert Path(settings.fileName()).resolve() != real.resolve()
    assert not str(Path(settings.fileName())).startswith(str(Path.home() / ".config"))
    after = real.stat().st_mtime_ns if real.exists() else None
    assert before == after
