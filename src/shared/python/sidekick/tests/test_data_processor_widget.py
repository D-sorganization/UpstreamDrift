from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from PyQt6.QtCore import QPoint
from sidekick.data_processing.core import DataProcessorEngine
from sidekick.ui.widgets.data_processor_widget import DataProcessorWidget


@pytest.fixture
def empty_widget(qapp) -> Any:
    widget = DataProcessorWidget()
    return widget


@pytest.fixture
def populated_widget(qapp) -> Any:
    widget = DataProcessorWidget()
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "c", "d", "e"],
            "C": [1.1, 2.2, 3.3, 4.4, np.nan],
        }
    )
    widget.engine.data = df
    widget._update_column_selectors()
    widget._update_table()
    widget.refresh_statistics()
    return widget


def test_initialization(empty_widget) -> None:
    assert empty_widget.calculator_name == "DataProcessor"
    assert isinstance(empty_widget.engine, DataProcessorEngine)
    assert empty_widget.current_file is None


def test_setup_state_management(qapp) -> None:
    widget = DataProcessorWidget()
    # It schedules setup_state_management on event loop. Call directly for test:
    widget.setup_state_management()
    # verify splitters and tables registered via CalculatorStateMixin
    split_names = [s["name"] for s in widget.splitters]
    assert "data_processor_splitter" in split_names

    copy_types = [c["type"] for c in widget.copyable_widgets]
    assert "table" in copy_types


def test_open_file_success(empty_widget) -> None:
    with (
        patch(
            "PyQt6.QtWidgets.QFileDialog.getOpenFileName", return_value=("test.csv", "")
        ),
        patch.object(empty_widget.engine, "load_file") as mock_load,
    ):
        res = MagicMock()
        res.success = True
        mock_load.return_value = res

        with patch.object(empty_widget, "_update_table") as mock_upd:
            empty_widget.open_file()
            assert empty_widget.current_file == "test.csv"
            assert mock_upd.call_count >= 1
            assert "Loaded" in empty_widget.status_label.text()


def test_open_file_failure(empty_widget) -> None:
    with (
        patch(
            "PyQt6.QtWidgets.QFileDialog.getOpenFileName", return_value=("test.csv", "")
        ),
        patch.object(empty_widget.engine, "load_file") as mock_load,
    ):
        res = MagicMock()
        res.success = False
        res.message = "Failed to load"
        mock_load.return_value = res

        with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warn:
            empty_widget.open_file()
            mock_warn.assert_called_once_with(empty_widget, "Error", "Failed to load")


def test_open_file_cancel(empty_widget) -> None:
    with patch("PyQt6.QtWidgets.QFileDialog.getOpenFileName", return_value=("", "")):
        with patch.object(empty_widget.engine, "load_file") as mock_load:
            empty_widget.open_file()
            mock_load.assert_not_called()


def test_save_file_no_data(empty_widget) -> None:
    with patch.object(empty_widget.engine, "export_data") as mock_exp:
        empty_widget.save_file()
        mock_exp.assert_not_called()


def test_save_file_existing(populated_widget) -> None:
    populated_widget.current_file = "test.csv"
    with patch.object(populated_widget.engine, "export_data") as mock_exp:
        res = MagicMock()
        res.success = True
        mock_exp.return_value = res

        populated_widget.save_file()
        mock_exp.assert_called_once_with("test.csv")
        assert populated_widget.status_label.text() == "Saved"


def test_save_file_existing_error(populated_widget) -> None:
    populated_widget.current_file = "test.csv"
    with patch.object(populated_widget.engine, "export_data") as mock_exp:
        res = MagicMock()
        res.success = False
        res.message = "Error saving"
        mock_exp.return_value = res

        with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warn:
            populated_widget.save_file()
            mock_warn.assert_called_once_with(populated_widget, "Error", "Error saving")


def test_save_file_no_file(populated_widget) -> None:
    populated_widget.current_file = None
    with patch.object(populated_widget, "export_file") as mock_export:
        populated_widget.save_file()
        mock_export.assert_called_once()


def test_export_file(populated_widget) -> None:
    with (
        patch(
            "PyQt6.QtWidgets.QFileDialog.getSaveFileName", return_value=("out.csv", "")
        ),
        patch.object(populated_widget.engine, "export_data") as mock_exp,
    ):
        res = MagicMock()
        res.success = True
        mock_exp.return_value = res

        populated_widget.export_file()
        mock_exp.assert_called_once_with("out.csv")
        assert populated_widget.status_label.text() == "Exported"


def test_undo_redo_reset(populated_widget) -> None:
    with patch.object(populated_widget.engine, "undo") as mock_undo:
        res = MagicMock()
        res.success = True
        mock_undo.return_value = res
        populated_widget.undo()
        assert populated_widget.status_label.text() == "Undo"

    with patch.object(populated_widget.engine, "redo") as mock_redo:
        res = MagicMock()
        res.success = True
        mock_redo.return_value = res
        populated_widget.redo()
        assert populated_widget.status_label.text() == "Redo"

    with patch.object(populated_widget.engine, "reset") as mock_reset:
        res = MagicMock()
        res.success = True
        mock_reset.return_value = res
        populated_widget.reset_data()
        assert populated_widget.status_label.text() == "Reset"


def test_update_table_pagination(populated_widget) -> None:
    populated_widget.rows_per_page.setRange(1, 1000)
    populated_widget.rows_per_page.setValue(2)
    populated_widget._update_table()  # manually trigger
    # Should trigger _update_table and set to 3 pages
    assert populated_widget.total_pages == 3
    assert populated_widget.current_page == 0
    assert populated_widget.data_table.rowCount() == 2

    populated_widget._next_page()
    assert populated_widget.current_page == 1

    populated_widget._prev_page()
    assert populated_widget.current_page == 0


def test_refresh_statistics_empty(empty_widget) -> None:
    # Should do nothing
    empty_widget.refresh_statistics()
    assert empty_widget.row_count_label.text() == "0"


def test_update_column_stats(populated_widget) -> None:
    populated_widget.column_selector.setCurrentText("A")
    populated_widget._update_column_stats()
    text = populated_widget.stats_text.toPlainText()
    assert "A" in text
    assert "Count: 5" in text


def test_show_table_context_menu(populated_widget) -> None:
    with patch("PyQt6.QtWidgets.QMenu.exec") as mock_exec:
        populated_widget._show_table_context_menu(QPoint(0, 0))
        mock_exec.assert_called_once()


def test_copy_selected(populated_widget) -> None:
    from PyQt6.QtWidgets import QTableWidgetSelectionRange

    populated_widget.data_table.setRangeSelected(
        QTableWidgetSelectionRange(0, 0, 1, 1), True
    )

    with patch("PyQt6.QtWidgets.QApplication.clipboard") as mock_clip:
        clip_obj = MagicMock()
        mock_clip.return_value = clip_obj

        populated_widget._copy_selected()
        clip_obj.setText.assert_called_once()
        text = clip_obj.setText.call_args[0][0]
        assert "1\ta" in text
        assert "2\tb" in text


def test_on_data_loaded_modified(empty_widget) -> None:
    empty_widget._on_data_loaded("test")
    assert empty_widget.current_page == 0

    empty_widget._on_data_modified()  # passes silently
