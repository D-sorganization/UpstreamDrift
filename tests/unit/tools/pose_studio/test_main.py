from __future__ import annotations

from unittest.mock import patch

from src.tools.pose_studio.__main__ import main


@patch("src.tools.pose_studio.__main__.sys.stderr.write")
def test_main_import_error(mock_stderr_write) -> None:
    # Simulate an ImportError when trying to import gui
    with patch.dict("sys.modules", {"src.tools.pose_studio.gui": None}):
        result = main()
        assert result == 1
        mock_stderr_write.assert_called_once()


def test_main_success_import() -> None:
    with patch("src.tools.pose_studio.gui.main", return_value=0) as mock_gui_main:
        result = main()
        assert result == 0
        mock_gui_main.assert_called_once()
