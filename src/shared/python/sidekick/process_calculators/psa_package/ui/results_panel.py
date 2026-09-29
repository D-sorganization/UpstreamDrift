import numpy as np
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..psa_model import PSAResults, get_flammability_status

__all__ = [
    "ResultsPanel",
]


class ResultsPanel(QWidget):
    """Panel for displaying calculation results."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Key Metrics Group
        metrics_group = QGroupBox("Key Performance Metrics")
        metrics_layout = QGridLayout()

        self.h2_recovery_label = QLabel("--")
        self.h2_purity_label = QLabel("--")
        self.net_product_label = QLabel("--")
        self.exhaust_label = QLabel("--")
        self.mass_balance_label = QLabel("--")

        font = QFont()
        font.setPointSize(12)
        font.setBold(True)

        for label in [
            self.h2_recovery_label,
            self.h2_purity_label,
            self.net_product_label,
        ]:
            label.setFont(font)

        metrics_layout.addWidget(QLabel("H2 Recovery:"), 0, 0)
        metrics_layout.addWidget(self.h2_recovery_label, 0, 1)
        metrics_layout.addWidget(QLabel("H2 Purity:"), 1, 0)
        metrics_layout.addWidget(self.h2_purity_label, 1, 1)
        metrics_layout.addWidget(QLabel("Net Product:"), 2, 0)
        metrics_layout.addWidget(self.net_product_label, 2, 1)
        metrics_layout.addWidget(QLabel("Exhaust:"), 3, 0)
        metrics_layout.addWidget(self.exhaust_label, 3, 1)
        metrics_layout.addWidget(QLabel("Mass Balance:"), 4, 0)
        metrics_layout.addWidget(self.mass_balance_label, 4, 1)

        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        # Safety Metrics Group
        safety_group = QGroupBox("Safety Metrics")
        safety_layout = QGridLayout()

        self.s2_tail_h2_label = QLabel("--")
        self.s2_tail_o2_label = QLabel("--")
        self.flammability_label = QLabel("--")
        self.flammability_label.setStyleSheet("font-weight: bold;")

        safety_layout.addWidget(QLabel("S2 Tail H2:"), 0, 0)
        safety_layout.addWidget(self.s2_tail_h2_label, 0, 1)
        safety_layout.addWidget(QLabel("S2 Tail O2:"), 1, 0)
        safety_layout.addWidget(self.s2_tail_o2_label, 1, 1)
        safety_layout.addWidget(QLabel("Status:"), 2, 0)
        safety_layout.addWidget(self.flammability_label, 2, 1)

        safety_group.setLayout(safety_layout)
        layout.addWidget(safety_group)

        # Stream Flows Table
        flows_group = QGroupBox("Stream Flows (SCFM)")
        flows_layout = QVBoxLayout()

        self.flows_table = QTableWidget()
        self.flows_table.setColumnCount(9)
        self.flows_table.setHorizontalHeaderLabels(
            [
                "Component",
                "Fresh Feed",
                "Mixed Feed",
                "Exhaust",
                "Interstage",
                "S2 Tail",
                "S2 Tail Recy",
                "Gross Prod",
                "Net Prod",
            ]
        )
        flows_layout.addWidget(self.flows_table)

        flows_group.setLayout(flows_layout)
        layout.addWidget(flows_group)

        # Compositions Table
        comp_group = QGroupBox("Stream Compositions (%)")
        comp_layout = QVBoxLayout()

        self.comp_table = QTableWidget()
        self.comp_table.setColumnCount(7)
        self.comp_table.setHorizontalHeaderLabels(
            [
                "Component",
                "Fresh Feed",
                "Mixed Feed",
                "Exhaust",
                "Interstage",
                "S2 Tail",
                "Net Prod",
            ]
        )
        comp_layout.addWidget(self.comp_table)

        comp_group.setLayout(comp_layout)
        layout.addWidget(comp_group)

    def update_results(self, results: PSAResults) -> None:
        """Update display with calculation results."""
        if results is None:
            raise ValueError("results must be provided")
        self._update_key_metrics(results)
        self._update_safety_metrics(results)
        self._update_flows_table(results)
        self._update_compositions_table(results)

    def _update_key_metrics(self, results: PSAResults) -> None:
        """Update key performance metric labels."""
        if results is None:
            raise ValueError("results must be provided")
        self.h2_recovery_label.setText(f"{results.h2_recovery_pct:.2f}%")
        self.h2_purity_label.setText(f"{results.h2_purity_pct:.5f}%")
        self.net_product_label.setText(f"{results.total_net_product_scfm:.2f} SCFM")
        self.exhaust_label.setText(f"{results.total_exhaust_scfm:.2f} SCFM")
        self.mass_balance_label.setText(f"{results.mass_balance_error:.2e}")

    def _update_safety_metrics(self, results: PSAResults) -> None:
        """Update safety/flammability metric labels and styling."""
        if results is None:
            raise ValueError("results must be provided")
        self.s2_tail_h2_label.setText(f"{results.s2_tail_h2_pct:.2f}%")
        self.s2_tail_o2_label.setText(f"{results.s2_tail_o2_pct:.2f}%")

        status = get_flammability_status(results.s2_tail_h2_pct, results.s2_tail_o2_pct)
        self.flammability_label.setText(status)

        if "CRITICAL" in status or "FLAMMABLE" in status or "DANGEROUS" in status:
            self.flammability_label.setStyleSheet(
                "font-weight: bold; color: red; background-color: #ffcccc;"
            )
        elif "Caution" in status:
            self.flammability_label.setStyleSheet(
                "font-weight: bold; color: orange; background-color: #ffffcc;"
            )
        else:
            self.flammability_label.setStyleSheet(
                "font-weight: bold; color: green; background-color: #ccffcc;"
            )

    def _update_flows_table(self, results: PSAResults) -> None:
        """Populate the flows table with component flow data and totals."""
        if results is None:
            raise ValueError("results must be provided")
        n_comp = len(results.component_names)
        self.flows_table.setRowCount(n_comp + 1)

        flow_columns = [
            results.flows.fresh_feed,
            results.flows.mixed_feed,
            results.flows.exhaust,
            results.flows.interstage,
            results.flows.s2_tail,
            results.flows.s2_tail_recycle,
            results.flows.gross_product,
            results.flows.net_product,
        ]

        for i, name in enumerate(results.component_names):
            self.flows_table.setItem(i, 0, QTableWidgetItem(name))
            for col_idx, col_data in enumerate(flow_columns):
                self.flows_table.setItem(
                    i, col_idx + 1, QTableWidgetItem(f"{col_data[i]:.4f}")
                )

        # Totals row
        self.flows_table.setItem(n_comp, 0, QTableWidgetItem("TOTAL"))
        totals = [
            results.total_feed_scfm,
            np.sum(results.flows.mixed_feed),
            results.total_exhaust_scfm,
            np.sum(results.flows.interstage),
            np.sum(results.flows.s2_tail),
            np.sum(results.flows.s2_tail_recycle),
            np.sum(results.flows.gross_product),
            results.total_net_product_scfm,
        ]
        for col_idx, total in enumerate(totals):
            self.flows_table.setItem(
                n_comp, col_idx + 1, QTableWidgetItem(f"{total:.2f}")
            )

        self.flows_table.resizeColumnsToContents()

    def _update_compositions_table(self, results: PSAResults) -> None:
        """Populate the compositions table with component percentage data."""
        if results is None:
            raise ValueError("results must be provided")
        n_comp = len(results.component_names)
        self.comp_table.setRowCount(n_comp + 1)

        comp_columns = [
            results.compositions.fresh_feed,
            results.compositions.mixed_feed,
            results.compositions.exhaust,
            results.compositions.interstage,
            results.compositions.s2_tail,
            results.compositions.net_product,
        ]

        for i, name in enumerate(results.component_names):
            self.comp_table.setItem(i, 0, QTableWidgetItem(name))
            for col_idx, col_data in enumerate(comp_columns):
                self.comp_table.setItem(
                    i, col_idx + 1, QTableWidgetItem(f"{col_data[i]:.4f}")
                )

        # Totals row
        self.comp_table.setItem(n_comp, 0, QTableWidgetItem("TOTAL"))
        for j in range(1, 7):
            self.comp_table.setItem(n_comp, j, QTableWidgetItem("100.00"))

        self.comp_table.resizeColumnsToContents()
