"""Tests for the release support matrix exposed at runtime."""

from __future__ import annotations

import sys
import warnings

import pytest


def test_supported_platforms_has_canonical_artifact_rows() -> None:
    """The runtime matrix must expose exactly the canonical production artifacts."""
    from src.api._version import SUPPORTED_PLATFORMS

    assert set(SUPPORTED_PLATFORMS) == {
        "python_wheel",
        "docker_api",
        "tauri_desktop",
        "rust_crate",
    }


def test_supported_python_wheel_matrix_covers_declared_versions() -> None:
    """Python wheel support must match the documented release matrix."""
    from src.api._version import SUPPORTED_PLATFORMS

    python_wheel = SUPPORTED_PLATFORMS["python_wheel"]
    assert python_wheel["python"] == ("3.10", "3.11", "3.12", "3.13")
    assert "Windows 10+ x86_64" in python_wheel["os"]
    assert python_wheel["hardware"] == ("CPU",)


def test_warn_if_current_platform_supported_emits_no_warning() -> None:
    """Known supported Python versions should not warn on launcher startup."""
    from src.api._version import warn_if_unsupported_platform

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        warn_if_unsupported_platform(system_name="Linux", python_version=(3, 11))

    assert len(captured_warnings) == 0


def test_warn_if_unsupported_platform_emits_user_warning() -> None:
    """Out-of-matrix platform combinations must surface a UserWarning."""
    from src.api._version import __version__, warn_if_unsupported_platform

    with pytest.warns(UserWarning, match="not supported"):
        warn_if_unsupported_platform(system_name="Plan9", python_version=(3, 99))

    with pytest.warns(UserWarning, match=__version__):
        warn_if_unsupported_platform(system_name="Linux", python_version=(3, 9))


def test_launcher_checks_support_matrix_before_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public console script must check support before launching services."""
    import launch_golf_suite

    calls: list[str] = []
    monkeypatch.setattr(launch_golf_suite, "parse_arguments", lambda: object())
    monkeypatch.setattr(
        launch_golf_suite,
        "warn_if_unsupported_platform",
        lambda: calls.append("checked"),
    )
    monkeypatch.setattr(launch_golf_suite, "route_launch", lambda _args: None)

    launch_golf_suite.main()

    assert calls == ["checked"]
