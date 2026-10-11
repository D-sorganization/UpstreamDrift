"""Lazy Qt helpers shared by the two shot-pattern preview presenters."""

from collections.abc import Iterable
from typing import Any


def scale_previews(labels: Iterable[Any]) -> None:
    """Scale valid source pixmaps to label bounds without replacing the source."""
    from PyQt6 import QtCore, QtGui

    for label in labels:
        source = label.property("sourcePixmap")
        if isinstance(source, QtGui.QPixmap) and not source.isNull():
            label.setPixmap(
                source.scaled(
                    label.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            )
