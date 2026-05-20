"""Unit tests for top-level launch_golf_suite.py."""

from __future__ import annotations

import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest

import launch_golf_suite as lgs


class TestParseArguments:
    def test_default_args(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", ["upstream-drift"])
        args = lgs.parse_arguments()
        assert args.classic is False
        assert args.api_only is False
        assert args.engine is None
        assert args.port == 8000
        assert args.no_browser is False

    def test_classic_flag(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", ["upstream-drift", "--classic"])
        args = lgs.parse_arguments()
        assert args.classic is True

    def test_api_only_flag(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", ["upstream-drift", "--api-only"])
        args = lgs.parse_arguments()
        assert args.api_only is True

    def test_engine_choice(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", ["upstream-drift", "--engine", "mujoco"])
        args = lgs.parse_arguments()
        assert args.engine == "mujoco"

    def test_invalid_engine_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr(
            sys, "argv", ["upstream-drift", "--engine", "definitely-not-real"]
        )
        with pytest.raises(SystemExit):
            lgs.parse_arguments()

    def test_custom_port(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", ["upstream-drift", "--port", "9090"])
        args = lgs.parse_arguments()
        assert args.port == 9090

    def test_no_browser(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", ["upstream-drift", "--no-browser"])
        args = lgs.parse_arguments()
        assert args.no_browser is True

    def test_engine_fallback_when_import_fails(self, monkeypatch) -> None:
        # Force ImportError on the EngineType import path
        real_import = (
            __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__
        )

        def fake_import(name, *a, **kw):
            if "engine_manager" in name:
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(sys, "argv", ["upstream-drift", "--engine", "pendulum"])
        with patch("builtins.__import__", side_effect=fake_import):
            args = lgs.parse_arguments()
        assert args.engine == "pendulum"


def _ns(**kwargs) -> argparse.Namespace:
    defaults = {
        "classic": False,
        "api_only": False,
        "engine": None,
        "port": 8000,
        "no_browser": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestRouteLaunchValidation:
    def test_none_args_raises(self) -> None:
        with pytest.raises(ValueError):
            lgs.route_launch(None)  # type: ignore[arg-type]

    def test_non_namespace_args_raises(self) -> None:
        with pytest.raises(ValueError):
            lgs.route_launch({"engine": "mujoco"})  # type: ignore[arg-type]


class TestRouteLaunchEngine:
    def test_engine_routes_to_launch_engine_directly(self) -> None:
        mock_factory = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "src.shared.python.launcher_factory": MagicMock(
                    launch_engine_directly=mock_factory
                )
            },
        ):
            lgs.route_launch(_ns(engine="mujoco"))
        mock_factory.assert_called_once_with("mujoco")

    def test_web_only_engine_falls_back_to_server(self, monkeypatch) -> None:
        mock_server = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "src.api.local_server": MagicMock(main=mock_server),
                "src.shared.python.launcher_factory": MagicMock(
                    launch_engine_directly=MagicMock()
                ),
            },
        ):
            lgs.route_launch(_ns(engine="matlab_2d"))
        mock_server.assert_called_once()
        assert __import__("os").environ.get("GOLF_DEFAULT_ENGINE") == "matlab_2d"


class TestRouteLaunchClassic:
    def test_classic_launches_pyqt(self) -> None:
        mock_classic = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "src.launchers.upstream_drift_launcher": MagicMock(main=mock_classic),
            },
        ):
            lgs.route_launch(_ns(classic=True))
        mock_classic.assert_called_once()

    def test_classic_import_error_exits(self, capsys) -> None:
        # Force ImportError when classic main is loaded
        bad = MagicMock()
        bad.main.side_effect = ImportError("nope")
        with (
            patch.dict(
                sys.modules,
                {"src.launchers.upstream_drift_launcher": bad},
            ),
            pytest.raises(SystemExit) as ei,
        ):
            lgs.route_launch(_ns(classic=True))
        assert ei.value.code == 1


class TestRouteLaunchApiOnly:
    def test_api_only_sets_env_and_runs(self, monkeypatch) -> None:
        monkeypatch.delenv("GOLF_NO_BROWSER", raising=False)
        monkeypatch.delenv("GOLF_PORT", raising=False)
        mock_main = MagicMock()
        with patch.dict(
            sys.modules,
            {"src.api.local_server": MagicMock(main=mock_main)},
        ):
            lgs.route_launch(_ns(api_only=True, port=7777))
        import os

        assert os.environ.get("GOLF_NO_BROWSER") == "true"
        assert os.environ.get("GOLF_PORT") == "7777"
        mock_main.assert_called_once()


class TestRouteLaunchDefault:
    def test_default_launches_web(self, monkeypatch) -> None:
        monkeypatch.delenv("GOLF_NO_BROWSER", raising=False)
        mock_main = MagicMock()
        with patch.dict(
            sys.modules,
            {"src.api.local_server": MagicMock(main=mock_main)},
        ):
            lgs.route_launch(_ns(port=8123))
        import os

        assert os.environ.get("GOLF_PORT") == "8123"
        mock_main.assert_called_once()

    def test_no_browser_sets_env(self, monkeypatch) -> None:
        monkeypatch.delenv("GOLF_NO_BROWSER", raising=False)
        mock_main = MagicMock()
        with patch.dict(
            sys.modules,
            {"src.api.local_server": MagicMock(main=mock_main)},
        ):
            lgs.route_launch(_ns(no_browser=True))
        import os

        assert os.environ.get("GOLF_NO_BROWSER") == "true"


class TestMain:
    def test_main_calls_warn_and_route(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", ["upstream-drift", "--api-only"])
        with (
            patch.object(lgs, "warn_if_unsupported_platform") as warn,
            patch.object(lgs, "route_launch") as route,
        ):
            lgs.main()
        warn.assert_called_once()
        route.assert_called_once()
