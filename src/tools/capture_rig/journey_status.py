"""Capture evidence and action-history trees, separate from navigation (#9913)."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem

from .capture_activity import display_status, read_activity
from .capture_evidence import observation_evidence
from .session import SessionMedia, VariantMedia, ViewMedia


def _view_row(view: ViewMedia, root: Path) -> QTreeWidgetItem:
    row = QTreeWidgetItem(
        [
            view.view,
            "Video available"
            if view.playable and view.playable.is_file()
            else "Video missing",
            str(view.playable or "Record or import this view"),
        ]
    )
    for name, path in (view.observation_sets or {}).items():
        detector, evidence = observation_evidence(root, path)
        child = QTreeWidgetItem(
            [
                f"{name} · {detector}",
                evidence,
                str(path),
            ]
        )
        child.setData(0, Qt.ItemDataRole.UserRole, path)
        child.setToolTip(0, "Double-click to review this output and its provenance")
        row.addChild(child)
    if row.childCount() == 0:
        row.addChild(
            QTreeWidgetItem(
                [
                    "Pose detection",
                    "Not available",
                    "Run Detect Pose or annotate points",
                ]
            )
        )
    return row


def _variant_row(variant: VariantMedia) -> QTreeWidgetItem:
    evidence = (
        ", ".join(
            name
            for present, name in (
                (variant.has_reconstruction, "3-D reconstruction"),
                (variant.has_model_fit, "Model fit"),
            )
            if present
        )
        or "No output available"
    )
    match_row = QTreeWidgetItem(
        [
            f"Match: {variant.label}",
            evidence,
            f"{variant.observation_set} · views: {', '.join(variant.views)}",
        ]
    )
    for name, path in (
        (
            "Reconstruction",
            variant.root / "reconstruct" / "session_reconstruction.json",
        ),
        ("Model Fit", variant.root / "model" / "joint_angles.json"),
    ):
        child = QTreeWidgetItem(
            [
                name,
                "Available — Review Provenance" if path.is_file() else "Not Available",
                str(path),
            ]
        )
        if path.is_file():
            child.setData(0, Qt.ItemDataRole.UserRole, path)
        match_row.addChild(child)
    return match_row


def evidence_tree(media: SessionMedia) -> QTreeWidget:
    """Build view and variant associations without loading video frames."""
    table = QTreeWidget()
    table.setHeaderLabels(["View / Model", "Available Evidence", "Association"])
    for view in media.views:
        table.addTopLevelItem(_view_row(view, media.root))
    for variant in media.variants:
        table.addTopLevelItem(_variant_row(variant))
    table.expandAll()
    table.setColumnWidth(0, 180)
    table.setColumnWidth(1, 230)
    return table


def history_tree(root: Path, active_id: str | None) -> QTreeWidget:
    """Show confirmed current activity separately from persisted unconfirmed work."""
    history_table = QTreeWidget()
    history_table.setHeaderLabels(
        ["Action / Started", "Status", "Detector / Model / Match"]
    )
    try:
        history = read_activity(root)
    except (ValueError, OSError) as exc:
        history_table.addTopLevelItem(
            QTreeWidgetItem(["Activity history", "Needs Attention", str(exc)])
        )
    else:
        for action in reversed(history.actions):
            history_table.addTopLevelItem(
                QTreeWidgetItem(
                    [
                        f"{action.action} · {action.started.astimezone():%Y-%m-%d %H:%M:%S}",
                        display_status(action, active_id),
                        action.context,
                    ]
                )
            )
        if not history.actions:
            history_table.addTopLevelItem(
                QTreeWidgetItem(
                    [
                        "No recorded action history",
                        "Unknown",
                        "Existing outputs are listed above",
                    ]
                )
            )
    return history_table
