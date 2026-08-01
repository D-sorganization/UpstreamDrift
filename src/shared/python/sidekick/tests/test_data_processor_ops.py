from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from PyQt6.QtWidgets import QMessageBox
from sidekick.data_processing.core import AggregationType, FitType
from sidekick.ui.widgets.data_processor_widget import DataProcessorWidget


@pytest.fixture
def test_widget(qapp) -> None:
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
    return widget


def test_apply_filter(test_widget) -> None:
    test_widget.filter_column.setCurrentText("A")
    test_widget.filter_operator.setCurrentText(">")
    test_widget.filter_value.setText("2")

    with patch.object(test_widget.engine, "filter_data") as mock_filter:
        res = MagicMock()
        res.success = True
        res.message = "Filtered"
        mock_filter.return_value = res

        test_widget._apply_filter()
        mock_filter.assert_called_once_with("A", ">", 2.0)
        assert test_widget.status_label.text() == "Filtered"


def test_apply_filter_empty(test_widget) -> None:
    test_widget.filter_column.setCurrentText("")
    test_widget.filter_value.setText("")
    with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warn:
        test_widget._apply_filter()
        mock_warn.assert_called_once()


def test_apply_filter_string_val(test_widget) -> None:
    test_widget.filter_column.setCurrentText("B")
    test_widget.filter_operator.setCurrentText("==")
    test_widget.filter_value.setText("a")

    with patch.object(test_widget.engine, "filter_data") as mock_filter:
        res = MagicMock()
        res.success = False
        res.message = "Err"
        mock_filter.return_value = res

        with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warn:
            test_widget._apply_filter()
            mock_filter.assert_called_once_with("B", "==", "a")
            mock_warn.assert_called_once()


def test_execute_query(test_widget) -> None:
    test_widget.query_input.setText("A > 2")
    with patch.object(test_widget.engine, "query") as mock_query:
        res = MagicMock()
        res.success = True
        res.message = "Queried"
        mock_query.return_value = res

        test_widget._execute_query()
        mock_query.assert_called_once_with("A > 2")
        assert test_widget.status_label.text() == "Queried"

    # empty query
    test_widget.query_input.setText("")
    with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warn:
        test_widget._execute_query()
        mock_warn.assert_called_once()


def test_aggregate_data(test_widget) -> None:
    test_widget.agg_group_by.setCurrentText("(None)")
    test_widget.agg_column.setCurrentText("A")
    test_widget.agg_type.setCurrentText("sum")

    with patch.object(test_widget.engine, "aggregate") as mock_agg:
        res = MagicMock()
        res.success = True
        res.message = "Aggregated"
        mock_agg.return_value = res

        test_widget._aggregate_data()
        mock_agg.assert_called_once_with(None, "A", AggregationType("sum"))

    test_widget.agg_group_by.setCurrentText("B")
    test_widget.agg_type.setCurrentText("mean")
    with patch.object(test_widget.engine, "aggregate") as mock_agg:
        res = MagicMock()
        res.success = False
        res.message = "Failed agg"
        mock_agg.return_value = res

        with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warn:
            test_widget._aggregate_data()
            mock_warn.assert_called_once()


def test_add_column(test_widget) -> None:
    test_widget.new_col_name.setText("D")
    test_widget.new_col_expr.setText("A * 2")

    with patch.object(test_widget.engine, "add_calculated_column") as mock_add:
        res = MagicMock()
        res.success = True
        res.message = "Added D"
        mock_add.return_value = res

        test_widget._add_column()
        mock_add.assert_called_once_with("D", "A * 2")

    # Empty
    test_widget.new_col_name.setText("")
    with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warn:
        test_widget._add_column()
        mock_warn.assert_called_once()


def test_transform_column(test_widget) -> None:
    test_widget.transform_column.setCurrentText("C")
    test_widget.transform_type.setCurrentText("round")
    test_widget.transform_param.setValue(2)

    with patch.object(test_widget.engine, "transform_column") as mock_trans:
        res = MagicMock()
        res.success = True
        res.message = "Transformed"
        mock_trans.return_value = res

        test_widget._transform_column()
        mock_trans.assert_called_once_with("C", "round", decimals=2)

    test_widget.transform_type.setCurrentText("fillna")
    test_widget.transform_param.setValue(0.5)
    with patch.object(test_widget.engine, "transform_column") as mock_trans:
        res = MagicMock()
        res.success = False
        res.message = "Fail"
        mock_trans.return_value = res

        with patch("PyQt6.QtWidgets.QMessageBox.warning"):
            test_widget._transform_column()
            mock_trans.assert_called_once_with("C", "fillna", value=0.5)


def test_rename_column(test_widget) -> None:
    test_widget.rename_column.setCurrentText("A")
    test_widget.rename_to.setText("Alpha")

    with patch.object(test_widget.engine, "rename_column") as mock_rename:
        res = MagicMock()
        res.success = True
        res.message = "Renamed"
        mock_rename.return_value = res

        test_widget._rename_column()
        mock_rename.assert_called_once_with("A", "Alpha")

    test_widget.rename_to.setText("")
    with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warn:
        test_widget._rename_column()
        mock_warn.assert_called_once()


def test_drop_column(test_widget) -> None:
    test_widget.rename_column.setCurrentText("A")

    with (
        patch(
            "PyQt6.QtWidgets.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ),
        patch.object(test_widget.engine, "drop_columns") as mock_drop,
    ):
        res = MagicMock()
        res.success = True
        res.message = "Dropped"
        mock_drop.return_value = res

        test_widget._drop_column()
        mock_drop.assert_called_once_with(["A"])


def test_fit_curve(test_widget) -> None:
    test_widget.fit_x_column.setCurrentText("A")
    test_widget.fit_y_column.setCurrentText("C")
    test_widget.fit_type.setCurrentText("linear")
    test_widget.fit_degree.setValue(1)

    with patch.object(test_widget.engine, "fit_curve") as mock_fit:
        res = MagicMock()
        res.equation = "y = mx + b"
        res.r_squared = 0.99
        res.coefficients = [1.0, 0.0]
        res.residuals = np.array([0.1, 0.2])
        mock_fit.return_value = res

        test_widget._fit_curve()
        mock_fit.assert_called_once_with("A", "C", FitType("linear"), 1)
        assert "Equation:" in test_widget.fit_results_text.toHtml()

    # failure
    with patch.object(test_widget.engine, "fit_curve", return_value=None):
        test_widget._fit_curve()
        assert "failed" in test_widget.fit_results_text.toPlainText()

    # empty col
    test_widget.fit_x_column.clear()
    with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warn:
        test_widget._fit_curve()
        mock_warn.assert_called_once()
