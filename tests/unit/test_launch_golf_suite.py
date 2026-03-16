import argparse
from unittest.mock import patch

import launch_golf_suite


def test_parse_arguments():
    with patch("sys.argv", ["launch_golf_suite.py", "--engine", "mujoco"]):
        args = launch_golf_suite.parse_arguments()
        assert args.engine == "mujoco"
        assert args.port == 8000

    with patch("sys.argv", ["launch_golf_suite.py", "--classic", "--port", "9000"]):
        args = launch_golf_suite.parse_arguments()
        assert args.classic is True
        assert args.port == 9000


@patch("src.shared.python.launcher_factory.launch_engine_directly")
def test_route_launch_engine(mock_launch, monkeypatch):
    args = argparse.Namespace(engine="mujoco", classic=False, api_only=False)
    launch_golf_suite.route_launch(args)
    mock_launch.assert_called_once_with("mujoco")


@patch("src.api.local_server.main")
def test_route_launch_web_engine(mock_server_main, monkeypatch):
    args = argparse.Namespace(
        engine="matlab_2d", classic=False, api_only=False, port=8000, no_browser=False
    )
    launch_golf_suite.route_launch(args)
    mock_server_main.assert_called_once()


@patch("src.launchers.golf_launcher.main")
def test_route_launch_classic(mock_classic_main):
    args = argparse.Namespace(engine=None, classic=True, api_only=False)
    launch_golf_suite.route_launch(args)
    mock_classic_main.assert_called_once()


@patch("src.api.local_server.main")
def test_route_launch_api_only(mock_server_main):
    args = argparse.Namespace(engine=None, classic=False, api_only=True, port=8080)
    launch_golf_suite.route_launch(args)
    mock_server_main.assert_called_once()


@patch("src.api.local_server.main")
def test_route_launch_default(mock_server_main):
    args = argparse.Namespace(
        engine=None, classic=False, api_only=False, port=8000, no_browser=True
    )
    launch_golf_suite.route_launch(args)
    mock_server_main.assert_called_once()


@patch("launch_golf_suite.parse_arguments")
@patch("launch_golf_suite.route_launch")
def test_main(mock_route, mock_parse):
    mock_args = argparse.Namespace()
    mock_parse.return_value = mock_args
    launch_golf_suite.main()
    mock_route.assert_called_once_with(mock_args)
